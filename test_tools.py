"""
Unit tests for the refined Hermes Coding Agent components.
"""

import json
import os
import shutil
import sys
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
        self.test_dir = os.path.join(os.path.dirname(__file__), "_test_tmp")
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

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

        # Verify that the skill was retrieved and pre-injected into messages
        injected_turns = [m for m in agent.messages if "[RELEVANT LEARNED SKILLS AUTO-INJECTED]" in str(m.get("content"))]
        self.assertEqual(len(injected_turns), 1)
        self.assertIn("fastapi_endpoint_pattern", injected_turns[0]["content"])
        self.assertIn("from fastapi import APIRouter", injected_turns[0]["content"])


if __name__ == "__main__":
    unittest.main()

