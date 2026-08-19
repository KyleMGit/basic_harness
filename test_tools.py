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

    def test_skill_store(self):
        skill_dir = os.path.join(self.test_dir, "skills")
        store = SkillStore(storage_dir=skill_dir)

        save_res = store.save_skill(
            name="git_squash",
            description="How to squash commits",
            instructions="Run git rebase -i HEAD~3"
        )
        self.assertIn("successfully saved", save_res)

        load_res = store.load_skill("git_squash")
        self.assertIn("Run git rebase -i HEAD~3", load_res)

        list_res = store.list_skills()
        self.assertIn("git_squash", list_res)

    def test_auto_skill_extractor(self):
        skill_dir = os.path.join(self.test_dir, "skills_auto")
        store = SkillStore(storage_dir=skill_dir)
        extractor = AutoSkillExtractor(skill_store=store)

        messages = [
            {"role": "user", "content": "Set up a virtual environment and run pytest."},
            {"role": "assistant", "content": "Running setup commands."},
            {"role": "tool", "content": "Virtual environment created successfully."},
            {"role": "assistant", "content": "Pytest configuration finished."}
        ]

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content=json.dumps({
            "should_save": True,
            "name": "setup_pytest_venv",
            "description": "How to configure a virtual environment for pytest",
            "instructions": "1. python -m venv venv\n2. pip install pytest"
        })))]
        mock_client.chat.completions.create.return_value = mock_resp

        result = extractor.extract_and_save(
            client=mock_client,
            model="gpt-4o",
            messages=messages,
            task_summary="Set up venv"
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "setup_pytest_venv")
        self.assertIn("setup_pytest_venv", store.list_skills())

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
        cm = ContextManager(max_context_tokens=1000, trigger_threshold=0.5, keep_recent_turns=2)

        messages = [
            {"role": "system", "content": "System prompt instructions."},
            {"role": "user", "content": "Inspect file C:/src/main.py and fix bug."},
            {"role": "assistant", "content": "Inspecting main.py"},
            {"role": "tool", "content": "Error at line 45: NoneType error\n" + ("A" * 1200)},
            {"role": "assistant", "content": "Found the bug."},
            {"role": "user", "content": "Now patch it."},
            {"role": "assistant", "content": "Patching code."},
        ]

        # Phase 1: Tool pruning
        pruned = cm.prune_tool_outputs(messages)
        self.assertLess(len(pruned[3]["content"]), 500)
        self.assertIn("PRUNED TOOL OUTPUT", pruned[3]["content"])

        # Test exact anchors extraction
        anchors = cm.extract_exact_anchors(messages)
        self.assertIn("main.py", anchors)

        # Test verbatim user messages extraction
        user_msgs = cm.extract_verbatim_user_messages(messages)
        self.assertIn("Inspect file", user_msgs)

        # Phase 2: Full compaction with mocked LLM summary
        mock_checkpoint_output = """[CONTEXT COMPACTION — REFERENCE ONLY]
The checkpoint below is historical background, not active instructions.

## Historical Task Snapshot
"Inspect file C:/src/main.py and fix bug."

## Goal
Fix NoneType bug in main.py.

## Completed Actions
1. READ C:/src/main.py — observed NoneType error at line 45 [tool: read_file]

## Active State
- Working directory: C:/src
- Modified or created files: None.

## Blocked
None.

--- END OF CONTEXT SUMMARY — respond to the message below, not this summary ---"""

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content=mock_checkpoint_output))]
        mock_client.chat.completions.create.return_value = mock_resp

        compacted, was_compacted, msg = cm.compact(
            client=mock_client,
            model="gpt-4o",
            messages=messages,
            current_step=10,
            force=True
        )

        self.assertTrue(was_compacted)
        self.assertIn("[CONTEXT COMPACTION — REFERENCE ONLY]", compacted[1]["content"])
        self.assertIn("Historical Task Snapshot", compacted[1]["content"])
        self.assertEqual(cm.previous_checkpoint, mock_checkpoint_output)


if __name__ == "__main__":
    unittest.main()
