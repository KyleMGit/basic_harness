import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


import agent as agent_module
import tools
from agent import HermesCodingAgent


class ReadThisPromptTests(unittest.TestCase):
    def _agent(self, **kwargs):
        defaults = dict(enable_memory=False, enable_skills=False, read_only=True)
        defaults.update(kwargs)
        with patch("agent.TrajectoryLogger"):
            return HermesCodingAgent(**defaults)

    def test_source_relative_load_is_independent_of_cwd_and_injected_once(self):
        expected = Path(agent_module.__file__).resolve().parent.joinpath("READ_THIS.md").read_text(encoding="utf-8").strip()
        with tempfile.TemporaryDirectory() as elsewhere:
            old_cwd = os.getcwd()
            try:
                os.chdir(elsewhere)
                prompt = self._agent().messages[0]["content"]
            finally:
                os.chdir(old_cwd)
        self.assertEqual(prompt.count(agent_module.READ_THIS_START_MARKER), 1)
        self.assertEqual(prompt.count(agent_module.READ_THIS_END_MARKER), 1)
        self.assertIn(expected, prompt)

    def test_all_modes_rebuild_with_exactly_one_block(self):
        instance = self._agent()
        for mode in ("read-only", "stateless", "normal"):
            instance.set_testing_mode(mode)
            prompt = instance.messages[0]["content"]
            self.assertEqual(prompt.count(agent_module.READ_THIS_START_MARKER), 1, mode)
            self.assertEqual(prompt.count(agent_module.READ_THIS_END_MARKER), 1, mode)

    def test_xml_wrapping_contains_exactly_one_block(self):
        prompt = self._agent(use_hermes_xml_protocol=True).messages[0]["content"]
        self.assertEqual(prompt.count(agent_module.READ_THIS_START_MARKER), 1)
        self.assertEqual(prompt.count(agent_module.READ_THIS_END_MARKER), 1)

    def test_required_source_failures_are_clear_and_fail_closed(self):
        cases = (
            (FileNotFoundError("gone"), "missing"),
            (PermissionError("no access"), "unreadable"),
            (UnicodeDecodeError("utf-8", b"x", 0, 1, "bad"), "UTF-8"),
        )
        for failure, message in cases:
            with self.subTest(message=message), patch.object(Path, "read_text", side_effect=failure):
                with self.assertRaisesRegex(RuntimeError, message):
                    agent_module.load_read_this_block()

        for content, message in (("  \n", "empty"), ("x" * (agent_module.READ_THIS_MAX_CHARS + 1), "maximum")):
            with self.subTest(message=message), patch.object(Path, "read_text", return_value=content):
                with self.assertRaisesRegex(RuntimeError, message):
                    agent_module.load_read_this_block()

        with patch.object(Path, "read_text", return_value="valid operator text"), \
             patch("agent.screen_prompt_content", return_value=(False, "scanner said no")):
            with self.assertRaisesRegex(RuntimeError, "scanner said no"):
                agent_module.load_read_this_block()

    def test_source_cannot_embed_reserved_block_markers(self):
        content = f"safe text\n{agent_module.READ_THIS_START_MARKER}\nmore safe text"
        with patch.object(Path, "read_text", return_value=content):
            with self.assertRaisesRegex(RuntimeError, "reserved READ_THIS.md marker"):
                agent_module.load_read_this_block()

    def test_fresh_prompt_rejects_reserved_markers_from_memory(self):
        marker_bearing_memory = (
            "<project_memory>\n"
            f"{agent_module.READ_THIS_START_MARKER}\n"
            f"{agent_module.READ_THIS_END_MARKER}\n"
            "</project_memory>"
        )
        with patch.object(
            agent_module.project_memory_manager,
            "format_system_prompt_block",
            return_value=marker_bearing_memory,
        ):
            with self.assertRaisesRegex(RuntimeError, "Fresh system prompt.*reserved READ_THIS.md marker"):
                self._agent(enable_memory=True)

    def test_reversed_saved_markers_are_rejected(self):
        reversed_markers = (
            f"BASE\n{agent_module.READ_THIS_END_MARKER}\n"
            f"content\n{agent_module.READ_THIS_START_MARKER}"
        )
        with self.assertRaisesRegex(RuntimeError, "out of order"):
            agent_module._read_this_snapshot(reversed_markers)

    def test_resume_keeps_marked_snapshot_and_upgrades_legacy_once(self):
        instance = self._agent(enable_memory=True, enable_skills=True)
        current_block = agent_module.load_read_this_block()

        def prepare(saved_prompt):
            instance.logger.load_session_messages = MagicMock(return_value=("task", [{"role": "user", "content": "turn"}]))
            instance.logger.load_session_state = MagicMock(return_value=None)
            instance.logger.load_session_system_prompt = MagicMock(return_value=saved_prompt)
            self.assertTrue(instance.resume_session("saved"))
            return instance.messages[0]["content"]

        snapshot = current_block.replace("operator", "frozen-operator")
        marked = prepare("BASE\n\n" + snapshot)
        self.assertEqual(marked, "BASE\n\n" + snapshot)
        self.assertNotIn(current_block, marked)

        legacy = prepare("LEGACY BASE")
        self.assertTrue(legacy.startswith("LEGACY BASE"))
        self.assertEqual(legacy.count(agent_module.READ_THIS_START_MARKER), 1)
        self.assertEqual(legacy.count(agent_module.READ_THIS_END_MARKER), 1)


class ReadThisFileToolTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name).resolve()
        self.read_this = self.root / "READ_THIS.md"
        source = Path(agent_module.__file__).resolve().parent / "READ_THIS.md"
        self.read_this.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        self.original_cwd = tools.terminal_session.cwd
        self.original_read_this_path = tools.READ_THIS_PATH
        self.addCleanup(setattr, tools.terminal_session, "cwd", self.original_cwd)
        self.addCleanup(setattr, tools, "READ_THIS_PATH", self.original_read_this_path)
        tools.terminal_session.cwd = str(self.root)
        tools.READ_THIS_PATH = self.read_this

    def test_exact_root_file_is_readable_but_write_and_patch_are_denied(self):
        self.assertNotIn("Denied", tools.read_file(str(self.read_this)))
        self.assertIn("Denied", tools.write_file(str(self.read_this), "replacement"))
        self.assertIn("Denied", tools.patch_file(str(self.read_this), "operator", "changed"))

    def test_nested_same_name_is_not_protected(self):
        target = self.root / "nested" / "READ_THIS.md"
        relative = str(target.relative_to(self.root))
        self.assertIn("Successfully wrote", tools.write_file(relative, "nested original"))
        self.assertIn("Successfully patched", tools.patch_file(relative, "original", "changed"))
        self.assertIn("nested changed", tools.read_file(relative))


if __name__ == "__main__":
    unittest.main()
