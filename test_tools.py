"""
Unit tests for the refined Hermes Coding Agent components.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from terminal import TerminalSession
from skills import SkillStore, AutoSkillExtractor
from protocol import ToolProtocol
from storage import TrajectoryLogger
from compaction import ContextManager
from agent import HermesCodingAgent
from tools import registry


class TestHermesAgentComponents(unittest.TestCase):

    def setUp(self):
        from memory import user_profile_manager, project_memory_manager
        from tools import skill_store, terminal_session

        self._temporary_directory = tempfile.TemporaryDirectory()
        self.test_dir = self._temporary_directory.name
        self._previous_cwd = os.getcwd()
        self._previous_paths = (
            skill_store.storage_dir,
            user_profile_manager.storage_dir,
            user_profile_manager.file_path,
            project_memory_manager.storage_dir,
            project_memory_manager.file_path,
            terminal_session.cwd,
        )
        os.chdir(self.test_dir)
        skill_store.storage_dir = os.path.join(self.test_dir, ".agent_skills")
        user_profile_manager.storage_dir = os.path.join(self.test_dir, ".agent_memories")
        user_profile_manager.file_path = os.path.join(user_profile_manager.storage_dir, "USER.md")
        project_memory_manager.storage_dir = user_profile_manager.storage_dir
        project_memory_manager.file_path = os.path.join(project_memory_manager.storage_dir, "MEMORY.md")
        terminal_session.cwd = self.test_dir

    def tearDown(self):
        from memory import user_profile_manager, project_memory_manager
        from tools import skill_store, terminal_session

        (
            skill_store.storage_dir,
            user_profile_manager.storage_dir,
            user_profile_manager.file_path,
            project_memory_manager.storage_dir,
            project_memory_manager.file_path,
            terminal_session.cwd,
        ) = self._previous_paths
        os.chdir(self._previous_cwd)
        self._temporary_directory.cleanup()

    def test_stateful_terminal_cd(self):
        term = TerminalSession(cwd=self.test_dir)
        sub_dir = os.path.join(self.test_dir, "nested_folder")
        os.makedirs(sub_dir, exist_ok=True)

        result = term.execute("cd nested_folder")
        self.assertIn("[Directory changed to]", result)
        self.assertEqual(term.cwd, os.path.abspath(sub_dir))

        out = term.execute("echo InSubdir")
        self.assertIn("InSubdir", out)
        self.assertIn("[EXIT CODE]: 0", out)

    def test_terminal_destructive_check(self):
        term = TerminalSession()
        self.assertTrue(term.is_destructive("rm -rf /some/path"))
        self.assertTrue(term.is_destructive("git reset --hard HEAD~1"))
        self.assertFalse(term.is_destructive("git status"))
        self.assertFalse(term.is_destructive("ls -la"))

    def test_skill_store_and_search(self):
        skill_dir = os.path.join(self.test_dir, "skills")
        store = SkillStore(storage_dir=skill_dir)

        # Save
        store.save_skill(
            name="git_squash_commits",
            description="How to rebase and squash git commits together",
            instructions="Run git rebase -i HEAD~3"
        )
        store.save_skill(
            name="setup_pytest_env",
            description="Configure python virtual environment for pytest suites",
            instructions="python -m venv venv && pip install pytest"
        )

        # Test catalog formatting
        catalog = store.format_catalog_prompt()
        self.assertIn("git_squash_commits", catalog)
        self.assertIn("setup_pytest_env", catalog)

        # Test loading by plain name, .md, and .json
        load_plain = store.load_skill("git_squash_commits")
        self.assertIn("Run git rebase -i HEAD~3", load_plain)
        
        load_md = store.load_skill("git_squash_commits.md")
        self.assertIn("Run git rebase -i HEAD~3", load_md)

        # Test relevant skill retrieval
        matches = store.find_relevant_skills("I need to squash my last 3 commits in git")
        self.assertTrue(len(matches) > 0)
        self.assertEqual(matches[0]["name"], "git_squash_commits")

        pytest_matches = store.find_relevant_skills("Run the pytest test suite")
        self.assertTrue(len(pytest_matches) > 0)
        self.assertEqual(pytest_matches[0]["name"], "setup_pytest_env")

    def test_auto_skill_extractor_deduplication(self):
        skill_dir = os.path.join(self.test_dir, "skills_auto")
        store = SkillStore(storage_dir=skill_dir)
        extractor = AutoSkillExtractor(skill_store=store)

        # Save initial skill
        store.save_skill(
            name="setup_pytest_env",
            description="Configure python virtual environment for pytest",
            instructions="1. python -m venv venv"
        )

        messages = [
            {"role": "user", "content": "Set up virtualenv with pytest and coverage."},
            {"role": "assistant", "content": "Running setup."},
            {"role": "tool", "content": "venv created and coverage added."},
            {"role": "assistant", "content": "Configured."}
        ]

        # Case A: LLM updates existing skill instead of duplicating
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content=json.dumps({
            "action": "UPDATE",
            "target_skill_name": "setup_pytest_env",
            "name": "setup_pytest_env",
            "description": "Configure python virtual environment with pytest and coverage",
            "instructions": "1. python -m venv venv\n2. pip install pytest pytest-cov"
        })))]
        mock_client.chat.completions.create.return_value = mock_resp

        result = extractor.extract_and_save(
            client=mock_client,
            model="Qwen-32b",
            messages=messages,
            task_summary="Set up venv with coverage"
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["action"], "SKIP")
        self.assertEqual(result["name"], "setup_pytest_env")
        
        # Verify only 1 skill file exists (no duplicates)
        all_skills = store.get_all_skills()
        self.assertEqual(len(all_skills), 1)
        self.assertNotIn("pytest-cov", all_skills[0]["instructions"])

        # Case B: LLM action NONE (trivial / duplicate)
        mock_resp.choices = [MagicMock(message=MagicMock(content=json.dumps({"action": "NONE"})))]
        result_none = extractor.extract_and_save(
            client=mock_client,
            model="Qwen-32b",
            messages=messages,
            task_summary="Routine test"
        )
        self.assertIsNone(result_none)

    def test_hermes_xml_protocol_parsing(self):
        sample_model_response = """
I will check the files in the directory.
<tool_call>
{"name": "run_terminal_command", "arguments": {"command": "ls -la"}}
</tool_call>
"""
        thought, calls = ToolProtocol.extract_tool_calls(sample_model_response)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "run_terminal_command")
        self.assertEqual(calls[0]["arguments"]["command"], "ls -la")
        self.assertIn("I will check the files", thought)

    def test_trajectory_logger(self):
        db_path = os.path.join(self.test_dir, "test_history.db")
        logger = TrajectoryLogger(db_path=db_path)
        
        session_id = "test_s1"
        logger.start_session(session_id, "Test task")
        logger.log_step(session_id, 1, "user", content="Build a calculator")
        logger.log_step(
            session_id,
            2,
            "assistant",
            content="Writing code",
            tool_calls=[{"name": "write_file", "args": {"file_path": "calc.py"}}]
        )
        logger.end_session(session_id, "COMPLETED")

        jsonl_path = os.path.join(self.test_dir, "trajectory.jsonl")
        logger.export_jsonl(session_id, jsonl_path)
        self.assertTrue(os.path.exists(jsonl_path))

        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 2)

    def test_interactive_command_prompt(self):
        agent = HermesCodingAgent(confirm_all_terminal_commands=True)

        with patch("builtins.input", return_value=""):
            should_run, cmd, feedback = agent.prompt_user_for_command("echo hello", {})
            self.assertTrue(should_run)
            self.assertEqual(cmd, "echo hello")
            self.assertIsNone(feedback)

        with patch("builtins.input", return_value="n"):
            should_run, cmd, feedback = agent.prompt_user_for_command("echo hello", {})
            self.assertFalse(should_run)
            self.assertIn("denied by user", feedback)

    def test_context_checkpoint_compaction(self):
        cm = ContextManager(max_context_tokens=40960, trigger_threshold=0.7, keep_recent_turns=2)

        messages = [
            {"role": "system", "content": "System prompt instructions."},
            {"role": "user", "content": "Inspect file C:/src/main.py and fix bug."},
            {"role": "assistant", "content": "Inspecting main.py"},
            {"role": "tool", "content": "Error at line 45: NoneType error\n" + ("A" * 1200)},
            {"role": "assistant", "content": "Found the bug."},
            {"role": "user", "content": "Now patch it."},
            {"role": "assistant", "content": "Patching code."},
        ]

        pruned = cm.prune_tool_outputs(messages)
        self.assertLess(len(pruned[3]["content"]), 500)
        self.assertIn("PRUNED TOOL OUTPUT", pruned[3]["content"])

        anchors = cm.extract_exact_anchors(messages)
        self.assertIn("main.py", anchors)

        mock_checkpoint_output = """[CONTEXT COMPACTION — REFERENCE ONLY]
## Historical Task Snapshot
"Inspect file C:/src/main.py and fix bug."

## Goal
Fix NoneType bug in main.py.

--- END OF CONTEXT SUMMARY ---"""

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content=mock_checkpoint_output))]
        mock_client.chat.completions.create.return_value = mock_resp

        compacted, was_compacted, msg = cm.compact(
            client=mock_client,
            model="Qwen-32b",
            messages=messages,
            current_step=10,
            force=True
        )

        self.assertTrue(was_compacted)
        self.assertIn("[CONTEXT COMPACTION — REFERENCE ONLY]", compacted[1]["content"])
        self.assertEqual(cm.previous_checkpoint, mock_checkpoint_output)

    def test_agent_skill_auto_injection(self):
        from tools import skill_store as global_store
        # Save a skill into the store
        global_store.save_skill(
            name="fastapi_endpoint_pattern",
            description="How to write a standard FastAPI router with Pydantic schemas",
            instructions="from fastapi import APIRouter\nrouter = APIRouter()"
        )

        agent = HermesCodingAgent(auto_learn_skills=False)
        mock_response = MagicMock(choices=[MagicMock(message=MagicMock(content="Done building endpoint.", tool_calls=None))])
        agent.client.chat.completions.create = MagicMock(return_value=mock_response)

        agent.run("Create a new FastAPI router endpoint")

        # Skill instructions are an ephemeral provider projection, not durable user authorship.
        self.assertEqual(agent.messages[1]["content"], "Create a new FastAPI router endpoint")
        injected = agent.client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        self.assertIn("fastapi_endpoint_pattern", injected)
        self.assertIn("from fastapi import APIRouter", injected)

    def test_direct_skill_name_as_tool_call(self):
        from tools import registry as reg, skill_store as global_store
        global_store.save_skill(
            name="git_squash_commits",
            description="Rebase commits",
            instructions="git rebase -i HEAD~3"
        )

        # Model calls the skill name directly as a tool: [Tool request]: git_squash_commits Arguments: {}
        result = reg.execute("git_squash_commits", {})
        self.assertIn("=== SKILL: git_squash_commits ===", result)
        self.assertIn("git rebase -i HEAD~3", result)
        self.assertNotIn("Error: Tool", result)

    def test_user_profile_management(self):
        from memory import UserProfileManager
        mem_dir = os.path.join(self.test_dir, "memories")
        mgr = UserProfileManager(storage_dir=mem_dir)

        # 1. Load initial profile
        initial = mgr.load_profile()
        self.assertIn("User Profile & Preferences", initial)
        self.assertIn("Communication Preferences", initial)

        # 2. Append new preference under section
        update_res = mgr.update_preference("Technical Preferences & Conventions", "Prefers strict type hints and Pydantic v2")
        self.assertIn("Successfully updated USER.md", update_res)

        updated = mgr.load_profile()
        self.assertIn("Prefers strict type hints and Pydantic v2", updated)

        # 3. System prompt XML block formatting
        block = mgr.format_system_prompt_block()
        self.assertTrue(block.startswith("<user_profile>"))
        self.assertTrue(block.endswith("</user_profile>"))

    def test_user_profile_tools(self):
        from tools import registry as reg
        read_res = reg.execute("read_user_profile", {})
        self.assertIn("User Profile", read_res)

        # Test updating with a new preference
        unique_pref = f"Format terminal outputs with clean headers (run_id_{id(self)})"
        update_res = reg.execute("update_user_profile", {
            "category": "Communication Preferences",
            "preference": unique_pref
        })
        self.assertIn("Successfully updated USER.md", update_res)

        # Test deduplication when adding the exact same preference again
        repeat_res = reg.execute("update_user_profile", {
            "category": "Communication Preferences",
            "preference": unique_pref
        })
        self.assertIn("Preference already recorded in USER.md", repeat_res)

    def test_project_memory_management(self):
        from memory import ProjectMemoryManager
        mem_dir = os.path.join(self.test_dir, "proj_mem")
        mgr = ProjectMemoryManager(storage_dir=mem_dir)

        # 1. Load initial MEMORY.md
        initial = mgr.load_memory()
        self.assertIn("Project Memory & Architecture Facts", initial)
        self.assertIn("Codebase Architecture & Tech Stack", initial)

        # 2. Append fact
        update_res = mgr.update_fact("Codebase Architecture & Tech Stack", "PostgreSQL database running on port 5432")
        self.assertIn("Successfully updated MEMORY.md", update_res)

        updated = mgr.load_memory()
        self.assertIn("PostgreSQL database running on port 5432", updated)

        # 3. System prompt XML block formatting
        block = mgr.format_system_prompt_block()
        self.assertTrue(block.startswith("<project_memory>"))
        self.assertTrue(block.endswith("</project_memory>"))

    def test_session_resumption_and_listing(self):
        db_path = os.path.join(self.test_dir, "resume_test.db")
        logger = TrajectoryLogger(db_path=db_path)

        # Create sample session
        s_id = "sess_res_001"
        logger.start_session(s_id, "Build authentication system")
        logger.log_step(s_id, 1, "user", content="Create user login API")
        logger.log_step(s_id, 2, "assistant", content="Working on auth.py")
        logger.log_step(s_id, 3, "tool", content="auth.py written successfully")
        logger.end_session(s_id, "COMPLETED")

        # 1. Test listing sessions
        sessions = logger.list_sessions()
        self.assertTrue(len(sessions) > 0)
        self.assertEqual(sessions[0]["session_id"], s_id)
        self.assertEqual(sessions[0]["status"], "COMPLETED")
        self.assertEqual(sessions[0]["step_count"], 3)

        # 2. Test session resumption trajectory loading
        task, msgs = logger.load_session_messages(s_id)
        self.assertEqual(task, "Build authentication system")
        self.assertEqual(len(msgs), 3)
        self.assertEqual(msgs[0]["role"], "user")
        self.assertEqual(msgs[0]["content"], "Create user login API")

        # 3. Test agent resume_session
        agent = HermesCodingAgent(auto_learn_skills=False)
        agent.logger = logger
        resumed = agent.resume_session(s_id)
        self.assertTrue(resumed)
        self.assertEqual(agent.session_id, s_id)
        self.assertTrue(len(agent.messages) >= 4)  # 1 system + 3 turns

    def test_codebase_search_grep_and_find(self):
        from tools import registry as reg, terminal_session as ts
        ts.cwd = self.test_dir

        # Create mock file structure
        src_dir = os.path.join(self.test_dir, "src")
        os.makedirs(src_dir, exist_ok=True)
        
        file1 = os.path.join(src_dir, "auth_service.py")
        with open(file1, "w", encoding="utf-8") as f:
            f.write("def authenticate_user(token: str):\n    if not token:\n        return False\n    return True\n")

        file2 = os.path.join(src_dir, "database.py")
        with open(file2, "w", encoding="utf-8") as f:
            f.write("DB_HOST = 'localhost'\nDB_PORT = 5432\n")

        # 1. Test find_files_by_pattern
        find_res = reg.execute("find_files_by_pattern", {"pattern": "*.py", "search_path": "."})
        self.assertIn("auth_service.py", find_res)
        self.assertIn("database.py", find_res)

        # 2. Test grep_search literal
        grep_res = reg.execute("grep_search", {"query": "authenticate_user", "search_path": "."})
        self.assertIn("auth_service.py:1: def authenticate_user", grep_res)

        # 3. Test grep_search regex
        regex_res = reg.execute("grep_search", {"query": r"DB_\w+\s*=", "is_regex": True, "search_path": "."})
        self.assertIn("DB_HOST", regex_res)
        self.assertIn("DB_PORT", regex_res)

    def test_auto_memory_extractor(self):
        from memory import UserProfileManager, ProjectMemoryManager, AutoMemoryExtractor
        mem_dir = os.path.join(self.test_dir, "auto_mem")
        u_mgr = UserProfileManager(storage_dir=mem_dir)
        p_mgr = ProjectMemoryManager(storage_dir=mem_dir)
        extractor = AutoMemoryExtractor(user_manager=u_mgr, project_manager=p_mgr)

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps({
                        "user_profile_update": {
                            "category": "Technical Preferences & Conventions",
                            "preference": "Prefers uv instead of pip"
                        },
                        "project_memory_update": {
                            "category": "Environment & Configuration",
                            "fact": "Redis cache running on port 6379"
                        }
                    })
                )
            )
        ]
        mock_client.chat.completions.create.return_value = mock_resp

        messages = [
            {"role": "user", "content": "Please remember that I prefer uv over pip, and our redis runs on port 6379."},
            {"role": "assistant", "content": "Understood, I will use uv and redis on 6379."}
        ]

        result = extractor.extract_and_update(
            client=mock_client,
            model="Qwen-32b",
            messages=messages,
            task_summary="Configure package manager and redis"
        )

        self.assertIsNotNone(result.get("user_updated"))
        self.assertIn("Prefers uv instead of pip", u_mgr.load_profile())

        self.assertIsNotNone(result.get("project_updated"))
        self.assertIn("Redis cache running on port 6379", p_mgr.load_memory())

    def test_testing_modes_read_only_and_stateless(self):
        from tools import registry as reg

        # 1. Test Read-Only Mode
        agent_ro = HermesCodingAgent(read_only=True)
        self.assertTrue(agent_ro.read_only)

        save_res = reg.execute("save_skill", {
            "name": "test_ro_skill",
            "description": "desc",
            "instructions": "inst"
        }, read_only=agent_ro.read_only)
        self.assertIn("Read-only active", save_res)

        user_res = reg.execute("update_user_profile", {
            "category": "Communication Preferences",
            "preference": "Test pref"
        }, read_only=agent_ro.read_only)
        self.assertIn("Read-only active", user_res)

        # 2. Test Stateless Benchmark Mode
        agent_stateless = HermesCodingAgent(enable_skills=False, enable_memory=False, read_only=True)
        sys_prompt = agent_stateless.messages[0]["content"]
        self.assertIn("Skills disabled for testing", sys_prompt)
        self.assertIn("Default testing profile", sys_prompt)
if __name__ == "__main__":
    unittest.main()



