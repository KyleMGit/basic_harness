"""
Unit tests for the refined Hermes Coding Agent components.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
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
            terminal_session.read_only_roots,
        )
        os.chdir(self.test_dir)
        skill_store.storage_dir = os.path.join(self.test_dir, ".agent_skills")
        user_profile_manager.storage_dir = os.path.join(self.test_dir, ".agent_memories")
        user_profile_manager.file_path = os.path.join(user_profile_manager.storage_dir, "USER.md")
        project_memory_manager.storage_dir = user_profile_manager.storage_dir
        project_memory_manager.file_path = os.path.join(project_memory_manager.storage_dir, "MEMORY.md")
        terminal_session.cwd = self.test_dir
        terminal_session.set_read_only_roots(())

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
            read_only_roots,
        ) = self._previous_paths
        terminal_session.set_read_only_roots(read_only_roots)
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

    def test_terminal_auto_approves_common_reads_in_read_only_and_workspace_roots(self):
        with tempfile.TemporaryDirectory() as external:
            read_only_root = os.path.join(external, "database_information")
            os.makedirs(read_only_root)
            schema_file = os.path.join(read_only_root, "orders.sql")
            workspace_file = os.path.join(self.test_dir, "workspace.sql")
            for path in (schema_file, workspace_file):
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write("order_id")
            term = TerminalSession(cwd=self.test_dir)
            term.set_read_only_roots([read_only_root])

            commands = (
                f'head -n 20 "{schema_file}"',
                f'tail -n 5 "{schema_file}"',
                f'grep -rin "order_id" "{read_only_root}"',
                f'grep -r order_id "{read_only_root}"',
                f'ls -la "{read_only_root}"',
                f'cat "{schema_file}"',
                f'type "{schema_file}"',
                f'wc -l "{schema_file}"',
                f'stat "{schema_file}"',
                f'file "{schema_file}"',
                f'du -sh "{read_only_root}"',
                f'dir "{read_only_root}"',
                f'find "{read_only_root}" -type f -name "*.sql"',
                f'sha256sum "{schema_file}"',
                f'md5sum "{schema_file}"',
                f'cmp "{schema_file}" "{schema_file}"',
                f'diff "{schema_file}" "{schema_file}"',
                f'readlink "{schema_file}"',
                f'realpath "{schema_file}"',
                f'cat "{workspace_file}"',
                "pwd",
            )
            for command in commands:
                with self.subTest(command=command):
                    self.assertTrue(term.is_auto_approved_read_only_command(command))

    def test_terminal_auto_approval_fails_closed_for_unsafe_or_outside_commands(self):
        with tempfile.TemporaryDirectory() as external, tempfile.TemporaryDirectory() as forbidden:
            read_only_root = os.path.join(external, "database_information")
            os.makedirs(read_only_root)
            schema_file = os.path.join(read_only_root, "orders.sql")
            sensitive_file = os.path.join(read_only_root, ".env")
            outside_file = os.path.join(forbidden, "outside.sql")
            for path in (schema_file, sensitive_file, outside_file):
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write("order_id")
            list_file = os.path.join(self.test_dir, "input-list.txt")
            checksum_file = os.path.join(self.test_dir, "checksums.txt")
            existing_output = os.path.join(self.test_dir, "existing-output.txt")
            for path in (list_file, checksum_file, existing_output):
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(outside_file)
            expansion_dir = os.path.join(self.test_dir, "%USERPROFILE%")
            os.makedirs(expansion_dir)
            expansion_decoy = os.path.join(expansion_dir, "outside.sql")
            with open(expansion_decoy, "w", encoding="utf-8") as handle:
                handle.write("decoy")
            term = TerminalSession(cwd=self.test_dir)
            term.set_read_only_roots([read_only_root])

            commands = (
                f'head "{outside_file}"',
                f'head "{sensitive_file}"',
                f'head "{schema_file}" > "{os.path.join(read_only_root, "copy.sql")}"',
                f'grep order_id "{read_only_root}"; rm -rf "{read_only_root}"',
                f'grep "$(touch bad)" "{read_only_root}"',
                f'find "{read_only_root}" -delete',
                f'find "{read_only_root}" -exec cat {{}} +',
                f'file -C "{schema_file}"',
                f'file --compile "{schema_file}"',
                f'file -Cm "{schema_file}"',
                f'file -f "{list_file}"',
                f'md5sum -c "{checksum_file}"',
                f'sha256sum --check "{checksum_file}"',
                f'wc --files0-from "{list_file}"',
                f'du --files0-from "{list_file}"',
                f'diff --output "{existing_output}" "{schema_file}" "{schema_file}"',
                f'grep -r order_id "{read_only_root}"',
                f'grep -R order_id "{read_only_root}"',
                f'du -L "{read_only_root}"',
                f'ls -RL "{read_only_root}"',
                'cat "%USERPROFILE%\\outside.sql"',
                "grep order_id",
                "python -c pass",
            )
            for command in commands:
                with self.subTest(command=command):
                    self.assertFalse(term.is_auto_approved_read_only_command(command))

    @unittest.skipUnless(os.name == "nt", "Windows command shadowing rule")
    def test_terminal_auto_approval_rejects_workspace_command_shadow(self):
        workspace_file = os.path.join(self.test_dir, "workspace.sql")
        shadow_command = os.path.join(self.test_dir, "cat.bat")
        for path, content in (
            (workspace_file, "order_id"),
            (shadow_command, "@echo off\necho shadowed"),
        ):
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(content)
        term = TerminalSession(cwd=self.test_dir)

        self.assertFalse(
            term.is_auto_approved_read_only_command(f'cat "{workspace_file}"')
        )

    def test_changing_terminal_cwd_preserves_read_only_roots(self):
        first = os.path.join(self.test_dir, "first")
        second = os.path.join(self.test_dir, "second")
        os.makedirs(first)
        os.makedirs(second)
        term = TerminalSession(cwd=first)
        term.set_read_only_roots([self.test_dir])

        term.cwd = second

        self.assertEqual(term.read_only_roots, (os.path.realpath(self.test_dir),))

    def test_read_only_root_denial_survives_workspace_reconfiguration(self):
        from tools import export_teradata_csv, patch_file, terminal_session, write_file

        nested_workspace = os.path.join(self.test_dir, "nested-workspace")
        os.makedirs(nested_workspace)
        existing = os.path.join(nested_workspace, "existing.txt")
        with open(existing, "w", encoding="utf-8") as handle:
            handle.write("unchanged")
        terminal_session.set_read_only_roots([self.test_dir])
        terminal_session.cwd = nested_workspace

        self.assertIn(
            "read-only",
            write_file(os.path.join(nested_workspace, "new.txt"), "blocked").lower(),
        )
        self.assertIn("read-only", patch_file(existing, "unchanged", "changed").lower())
        with self.assertRaisesRegex(ValueError, "read-only"):
            export_teradata_csv(
                "SELECT 1",
                os.path.join(nested_workspace, "blocked.csv"),
            )
        self.assertFalse(os.path.exists(os.path.join(nested_workspace, "new.txt")))
        with open(existing, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "unchanged")

    def test_file_discovery_tools_read_external_roots_but_mutations_stay_in_workspace(self):
        from tools import (
            find_files_by_pattern,
            grep_search,
            list_directory,
            patch_file,
            read_file,
            terminal_session,
            write_file,
        )

        with tempfile.TemporaryDirectory() as external:
            schema_dir = os.path.join(external, "database_information")
            os.makedirs(os.path.join(schema_dir, "teradata"))
            schema_file = os.path.join(schema_dir, "teradata", "orders.sql")
            with open(schema_file, "w", encoding="utf-8") as handle:
                handle.write("CREATE TABLE orders (order_id INTEGER);\n")
            terminal_session.set_read_only_roots([schema_dir])

            self.assertIn("CREATE TABLE orders", read_file(schema_file))
            self.assertIn("orders.sql", list_directory(os.path.dirname(schema_file)))
            self.assertIn("orders.sql:1", grep_search("order_id", schema_dir))
            self.assertIn("orders.sql", find_files_by_pattern("*.sql", schema_dir))

            denied_write = write_file(os.path.join(schema_dir, "new.sql"), "SELECT 1")
            denied_patch = patch_file(schema_file, "orders", "changed")
            self.assertIn("read-only", denied_write.lower())
            self.assertIn("read-only", denied_patch.lower())
            self.assertFalse(os.path.exists(os.path.join(schema_dir, "new.sql")))
            with open(schema_file, encoding="utf-8") as handle:
                self.assertIn("CREATE TABLE orders", handle.read())

    def test_read_only_root_denies_parent_traversal(self):
        from tools import read_file, terminal_session

        with tempfile.TemporaryDirectory() as external:
            read_only_root = Path(external, "database_information")
            outside_file = Path(external, "secret.txt")
            read_only_root.mkdir()
            outside_file.write_text("secret", encoding="utf-8")
            terminal_session.set_read_only_roots([str(read_only_root)])

            traversal = read_file(str(read_only_root / ".." / "secret.txt"))
            self.assertIn("outside", traversal.lower())

    def test_read_only_root_denies_symlink_escape(self):
        from tools import read_file, terminal_session

        with tempfile.TemporaryDirectory() as external:
            read_only_root = Path(external, "database_information")
            outside_file = Path(external, "secret.txt")
            read_only_root.mkdir()
            outside_file.write_text("secret", encoding="utf-8")
            terminal_session.set_read_only_roots([str(read_only_root)])

            link = read_only_root / "outside-link.txt"
            try:
                link.symlink_to(outside_file)
            except OSError:
                self.skipTest("file symlinks unavailable")
            symlink_escape = read_file(str(link))
            self.assertIn("outside", symlink_escape.lower())

    def test_database_exports_reject_read_only_destinations_before_querying(self):
        from tools import export_impala_csv, export_teradata_csv, terminal_session

        with tempfile.TemporaryDirectory() as external:
            read_only_root = Path(external, "database_information")
            read_only_root.mkdir()
            terminal_session.set_read_only_roots([str(read_only_root)])

            for export in (export_teradata_csv, export_impala_csv):
                with self.subTest(export=export.__name__), self.assertRaisesRegex(
                    ValueError, "read-only"
                ):
                    export("SELECT 1", str(read_only_root / "blocked.csv"))

    def test_external_search_results_use_absolute_paths_across_drives(self):
        from tools import find_files_by_pattern, grep_search, terminal_session

        with tempfile.TemporaryDirectory() as external:
            read_only_root = Path(external, "database_information")
            schema_file = read_only_root / "orders.sql"
            read_only_root.mkdir()
            schema_file.write_text("order_id", encoding="utf-8")
            terminal_session.set_read_only_roots([str(read_only_root)])
            real_relpath = os.path.relpath

            def cross_drive_relpath(path, start):
                if os.path.realpath(start) == os.path.realpath(terminal_session.cwd):
                    raise ValueError("path is on a different drive")
                return real_relpath(path, start)

            with patch("tools.os.path.relpath", side_effect=cross_drive_relpath):
                grep_result = grep_search("order_id", str(read_only_root))
                find_result = find_files_by_pattern("*.sql", str(read_only_root))

            self.assertIn(str(schema_file.resolve()), grep_result)
            self.assertIn(str(schema_file.resolve()), find_result)

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
        self.assertEqual(result["action"], "UPDATE")
        self.assertEqual(result["name"], "setup_pytest_env")
        
        # Verify only 1 skill file exists (no duplicates)
        all_skills = store.get_all_skills()
        self.assertEqual(len(all_skills), 1)
        self.assertIn("pytest-cov", all_skills[0]["instructions"])

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
        self.assertTrue(cm.previous_checkpoint.startswith(mock_checkpoint_output))
        self.assertIn("## Exact Recovery Anchors", cm.previous_checkpoint)
        self.assertIn("## Verbatim Historical User Messages", cm.previous_checkpoint)

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


