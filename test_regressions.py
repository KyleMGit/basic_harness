"""Regression tests for confirmed harness defects."""

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


class TestReadOnlyIsolation(unittest.TestCase):
    def test_read_only_agent_initialization_creates_no_runtime_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = f"""
import os, sys
os.chdir({tmp!r})
sys.path.insert(0, {str(ROOT)!r})
from agent import HermesCodingAgent
HermesCodingAgent(
    read_only=True,
    auto_learn_skills=False,
    auto_learn_memory=False,
)
print(sorted(os.listdir('.')))
"""
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(result.stdout.strip(), "[]")

    def test_read_only_dispatch_blocks_every_write_capable_tool(self):
        from tools import registry, terminal_session

        with tempfile.TemporaryDirectory() as tmp:
            previous_cwd = terminal_session.cwd
            terminal_session.cwd = tmp
            try:
                cases = [
                    ("write_file", {"file_path": "blocked.txt", "content": "x"}),
                    (
                        "patch_file",
                        {
                            "file_path": "existing.txt",
                            "search_content": "old",
                            "replace_content": "new",
                        },
                    ),
                    ("run_terminal_command", {"command": "echo x > terminal-write.txt"}),
                    (
                        "save_skill",
                        {"name": "blocked", "description": "d", "instructions": "i"},
                    ),
                    (
                        "update_user_profile",
                        {"category": "Communication Preferences", "preference": "blocked"},
                    ),
                    (
                        "update_project_memory",
                        {"category": "Environment & Configuration", "fact": "blocked"},
                    ),
                ]
                Path(tmp, "existing.txt").write_text("old", encoding="utf-8")
                for name, arguments in cases:
                    with self.subTest(name=name):
                        result = registry.execute(name, arguments, read_only=True)
                        self.assertIn("Read-only active", result)
                self.assertFalse(Path(tmp, "blocked.txt").exists())
                self.assertFalse(Path(tmp, "terminal-write.txt").exists())
                self.assertEqual(Path(tmp, "existing.txt").read_text(encoding="utf-8"), "old")
            finally:
                terminal_session.cwd = previous_cwd

    def test_read_only_agent_does_not_log_trajectory(self):
        from agent import HermesCodingAgent

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp, "history.db")
            agent = HermesCodingAgent(
                read_only=True,
                enable_skills=False,
                enable_memory=False,
                auto_learn_skills=False,
                auto_learn_memory=False,
            )
            agent.logger.db_path = str(db_path)
            response = MagicMock()
            response.choices = [MagicMock(message=MagicMock(content="done", tool_calls=None))]
            agent.client.chat.completions.create = MagicMock(return_value=response)
            agent.run("read-only task")
            self.assertFalse(db_path.exists())


class TestProtocolPersistence(unittest.TestCase):
    def test_native_tool_call_round_trips_with_protocol_ids(self):
        from storage import TrajectoryLogger

        canonical_call = {
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": json.dumps({"file_path": "x.txt"}),
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            logger = TrajectoryLogger(str(Path(tmp, "history.db")))
            logger.start_session("s1", "task")
            logger.log_step("s1", 1, "assistant", "", [canonical_call])
            logger.log_step("s1", 2, "tool", "contents", tool_call_id="call_123")
            _, messages = logger.load_session_messages("s1")
            export_path = Path(tmp, "trajectory.jsonl")
            logger.export_jsonl("s1", str(export_path))
            exported = [json.loads(line) for line in export_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(messages[0]["tool_calls"], [canonical_call])
        self.assertEqual(messages[1]["role"], "tool")
        self.assertEqual(messages[1]["tool_call_id"], "call_123")
        self.assertEqual(exported[1]["tool_call_id"], "call_123")

    def test_xml_tool_response_preserves_call_id(self):
        from protocol import ToolProtocol

        response = ToolProtocol.format_hermes_tool_response(
            "read_file", "contents", tool_call_id="hermes_call_7"
        )
        self.assertIn('"tool_call_id": "hermes_call_7"', response)


class TestCompactionTransactional(unittest.TestCase):
    @staticmethod
    def _messages_with_tool_boundary():
        return [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "old request " * 20},
            {"role": "assistant", "content": "old answer " * 20},
            {"role": "user", "content": "prepare tool"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_boundary",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_boundary", "content": "tool result"},
            {"role": "user", "content": "latest request"},
        ]

    def test_summarizer_failure_keeps_original_messages(self):
        from compaction import ContextManager

        messages = self._messages_with_tool_boundary()
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("offline")
        manager = ContextManager(max_context_tokens=100, keep_recent_turns=2)

        result, compacted, status = manager.compact(
            client, "model", messages, current_step=10, force=True
        )

        self.assertFalse(compacted)
        self.assertEqual(result, messages)
        self.assertIn("failed", status.lower())
        self.assertIsNone(manager.previous_checkpoint)

    def test_compaction_never_splits_assistant_tool_group(self):
        from compaction import ContextManager

        messages = self._messages_with_tool_boundary()
        manager = ContextManager(max_context_tokens=1000, keep_recent_turns=2)
        checkpoint = "[CONTEXT COMPACTION — REFERENCE ONLY]\nshort checkpoint"
        with patch.object(manager, "summarize_history", return_value=checkpoint):
            result, compacted, _ = manager.compact(
                MagicMock(), "model", messages, current_step=10, force=True
            )

        self.assertTrue(compacted)
        tool_index = next(i for i, msg in enumerate(result) if msg.get("role") == "tool")
        assistant = result[tool_index - 1]
        self.assertEqual(assistant.get("role"), "assistant")
        self.assertEqual(assistant["tool_calls"][0]["id"], "call_boundary")

    def test_compaction_never_splits_xml_tool_group(self):
        from compaction import ContextManager

        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "old request " * 20},
            {"role": "assistant", "content": "old answer " * 20},
            {"role": "user", "content": "prepare XML tool"},
            {
                "role": "assistant",
                "content": '<tool_call>{"name":"read_file","arguments":{}}</tool_call>',
            },
            {
                "role": "user",
                "content": '<tool_response>{"tool_call_id":"hermes_call_0","content":"ok"}</tool_response>',
            },
            {"role": "user", "content": "latest request"},
        ]
        manager = ContextManager(max_context_tokens=1000, keep_recent_turns=2)
        checkpoint = "[CONTEXT COMPACTION — REFERENCE ONLY]\nshort checkpoint"
        with patch.object(manager, "summarize_history", return_value=checkpoint):
            result, compacted, _ = manager.compact(
                MagicMock(), "model", messages, current_step=10, force=True
            )

        self.assertTrue(compacted)
        response_index = next(
            i for i, msg in enumerate(result) if "<tool_response>" in msg.get("content", "")
        )
        self.assertEqual(result[response_index - 1]["role"], "assistant")
        self.assertIn("<tool_call>", result[response_index - 1]["content"])

    def test_ineffective_checkpoint_is_rejected_without_state_mutation(self):
        from compaction import ContextManager

        messages = self._messages_with_tool_boundary()
        manager = ContextManager(max_context_tokens=1000, keep_recent_turns=2)
        huge_checkpoint = "[CONTEXT COMPACTION — REFERENCE ONLY]\n" + ("x" * 10000)
        with patch.object(manager, "summarize_history", return_value=huge_checkpoint):
            result, compacted, status = manager.compact(
                MagicMock(), "model", messages, current_step=10, force=True
            )

        self.assertFalse(compacted)
        self.assertEqual(result, messages)
        self.assertIn("ineffective", status.lower())
        self.assertIsNone(manager.previous_checkpoint)


class TestDurableSessionState(unittest.TestCase):
    def test_compacted_transcript_and_context_state_round_trip(self):
        from storage import TrajectoryLogger

        active_messages = [
            {"role": "user", "content": "[CONTEXT COMPACTION — REFERENCE ONLY]\ncheckpoint B"},
            {"role": "assistant", "content": "Checkpoint ingested."},
            {"role": "user", "content": "active request"},
        ]
        context_state = {
            "previous_checkpoint": "checkpoint B",
            "last_compaction_step": 8,
            "compaction_count": 2,
        }
        with tempfile.TemporaryDirectory() as tmp:
            logger = TrajectoryLogger(str(Path(tmp, "history.db")))
            logger.start_session("s1", "task")
            logger.save_session_state("s1", active_messages, 9, context_state)
            loaded = logger.load_session_state("s1")

        self.assertEqual(loaded["messages"], active_messages)
        self.assertEqual(loaded["step_counter"], 9)
        self.assertEqual(loaded["context_state"], context_state)

    def test_resume_rehydrates_own_checkpoint_not_previous_session(self):
        from agent import HermesCodingAgent
        from storage import TrajectoryLogger

        with tempfile.TemporaryDirectory() as tmp:
            logger = TrajectoryLogger(str(Path(tmp, "history.db")))
            logger.start_session("session_b", "task B")
            logger.save_session_state(
                "session_b",
                [{"role": "user", "content": "active B"}],
                4,
                {
                    "previous_checkpoint": "checkpoint B",
                    "last_compaction_step": 3,
                    "compaction_count": 1,
                },
            )
            agent = HermesCodingAgent(
                enable_skills=False,
                enable_memory=False,
                auto_learn_skills=False,
                auto_learn_memory=False,
            )
            agent.logger = logger
            agent.context_manager.previous_checkpoint = "checkpoint A"
            self.assertTrue(agent.resume_session("session_b"))

        self.assertEqual(agent.context_manager.previous_checkpoint, "checkpoint B")
        self.assertEqual(agent.context_manager.last_compaction_step, 3)
        self.assertEqual(agent.step_counter, 4)
        self.assertEqual(agent.messages[-1], {"role": "user", "content": "active B"})


class TestMemoryBudget(unittest.TestCase):
    def test_user_profile_oversize_update_is_rejected_without_truncation(self):
        from memory import UserProfileManager

        with tempfile.TemporaryDirectory() as tmp:
            manager = UserProfileManager(tmp)
            manager.load_profile()
            self.assertIn("Successfully updated", manager.save_profile("original profile"))
            result = manager.save_profile("x" * (manager.MAX_CHAR_BUDGET + 1))
            self.assertIn("exceeds", result.lower())
            self.assertEqual(Path(manager.file_path).read_text(encoding="utf-8").strip(), "original profile")

    def test_project_memory_oversize_update_is_rejected_without_truncation(self):
        from memory import ProjectMemoryManager

        with tempfile.TemporaryDirectory() as tmp:
            manager = ProjectMemoryManager(tmp)
            manager.load_memory()
            self.assertIn("Successfully updated", manager.save_memory("original memory"))
            result = manager.save_memory("x" * (manager.MAX_CHAR_BUDGET + 1))
            self.assertIn("exceeds", result.lower())
            self.assertEqual(Path(manager.file_path).read_text(encoding="utf-8").strip(), "original memory")


class TestMessageSequencingAndCommandProvenance(unittest.TestCase):
    @staticmethod
    def _response(message):
        response = MagicMock()
        response.choices = [MagicMock(message=message)]
        return response

    @staticmethod
    def _native_message(content, calls=None):
        message = MagicMock()
        message.content = content
        message.tool_calls = calls
        payload = {"role": "assistant", "content": content or ""}
        if calls:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in calls
            ]
        message.model_dump.return_value = payload
        return message

    def test_skill_injection_is_combined_with_active_user_turn(self):
        from agent import HermesCodingAgent

        agent = HermesCodingAgent(
            max_iterations=1,
            enable_skills=True,
            enable_memory=False,
            auto_learn_skills=False,
            auto_learn_memory=False,
            read_only=True,
        )
        final = self._native_message("done")
        agent.client.chat.completions.create = MagicMock(return_value=self._response(final))
        matched = [{"name": "testing", "description": "d", "instructions": "i"}]
        with patch("agent.skill_store.find_relevant_skills", return_value=matched):
            agent.run("active task")

        roles = [message["role"] for message in agent.messages]
        self.assertNotIn(("user", "user"), list(zip(roles, roles[1:])))
        user_message = next(message for message in agent.messages if message["role"] == "user")
        self.assertEqual(user_message["content"], "active task")
        provider_messages = agent.client.chat.completions.create.call_args.kwargs["messages"]
        self.assertIn("RELEVANT LEARNED SKILLS", provider_messages[1]["content"])
        self.assertIn("active task", provider_messages[1]["content"])

    def test_parallel_xml_results_are_one_correlated_user_turn(self):
        from agent import HermesCodingAgent

        agent = HermesCodingAgent(
            max_iterations=2,
            enable_skills=False,
            enable_memory=False,
            auto_learn_skills=False,
            auto_learn_memory=False,
            read_only=True,
            use_hermes_xml_protocol=True,
        )
        first = MagicMock()
        first.content = (
            '<tool_call>{"name":"read_file","arguments":{"file_path":"a"}}</tool_call>'
            '<tool_call>{"name":"read_file","arguments":{"file_path":"b"}}</tool_call>'
        )
        first.tool_calls = None
        final = MagicMock()
        final.content = "done"
        final.tool_calls = None
        agent.client.chat.completions.create = MagicMock(
            side_effect=[self._response(first), self._response(final)]
        )
        agent.run("inspect both")

        roles = [message["role"] for message in agent.messages]
        self.assertNotIn(("user", "user"), list(zip(roles, roles[1:])))
        responses = [
            message["content"]
            for message in agent.messages
            if message["role"] == "user" and "<tool_response>" in message["content"]
        ]
        self.assertEqual(len(responses), 1)
        self.assertIn('"tool_call_id": "hermes_call_0"', responses[0])
        self.assertIn('"tool_call_id": "hermes_call_1"', responses[0])

    def test_edited_terminal_command_is_live_and_durably_accurate(self):
        from agent import HermesCodingAgent
        from storage import TrajectoryLogger

        call = MagicMock()
        call.id = "call_cmd"
        call.function.name = "run_terminal_command"
        call.function.arguments = json.dumps({"command": "echo proposed"})
        first = self._native_message("", [call])
        final = self._native_message("done")

        with tempfile.TemporaryDirectory() as tmp:
            agent = HermesCodingAgent(
                max_iterations=2,
                enable_skills=False,
                enable_memory=False,
                auto_learn_skills=False,
                auto_learn_memory=False,
            )
            agent.logger = TrajectoryLogger(str(Path(tmp, "history.db")))
            agent.client.chat.completions.create = MagicMock(
                side_effect=[self._response(first), self._response(final)]
            )
            agent.prompt_user_for_command = MagicMock(
                return_value=(True, "echo edited", None)
            )
            with patch(
                "agent.registry.execute",
                return_value="executed output containing the phrase Read-only active",
            ):
                agent.run("run it")
            _, durable = agent.logger.load_session_messages(agent.session_id)

        live_assistant = next(message for message in agent.messages if message.get("tool_calls"))
        live_args = json.loads(live_assistant["tool_calls"][0]["function"]["arguments"])
        durable_assistant = next(message for message in durable if message.get("tool_calls"))
        durable_args = json.loads(durable_assistant["tool_calls"][0]["function"]["arguments"])
        tool_result = next(message for message in durable if message["role"] == "tool")
        self.assertEqual(live_args["command"], "echo edited")
        self.assertEqual(durable_args["command"], "echo edited")
        self.assertIn("echo proposed", tool_result["content"])
        self.assertIn("echo edited", tool_result["content"])
        self.assertIn("Executed: echo edited", tool_result["content"])
        self.assertNotIn("Not executed", tool_result["content"])

    def test_second_edited_command_in_parallel_native_turn_is_live_and_durable(self):
        from agent import HermesCodingAgent
        from storage import TrajectoryLogger

        calls = []
        for call_id, command in (
            ("call_first", "echo first"),
            ("call_second", "echo second"),
        ):
            call = MagicMock()
            call.id = call_id
            call.function.name = "run_terminal_command"
            call.function.arguments = json.dumps({"command": command})
            calls.append(call)

        first = self._native_message("", calls)
        final = self._native_message("done")
        with tempfile.TemporaryDirectory() as tmp:
            agent = HermesCodingAgent(
                max_iterations=2,
                enable_skills=False,
                enable_memory=False,
                auto_learn_skills=False,
                auto_learn_memory=False,
            )
            agent.logger = TrajectoryLogger(str(Path(tmp, "history.db")))
            agent.client.chat.completions.create = MagicMock(
                side_effect=[self._response(first), self._response(final)]
            )
            agent.prompt_user_for_command = MagicMock(
                side_effect=[
                    (True, "echo first edited", None),
                    (True, "echo second edited", None),
                ]
            )
            with patch("agent.registry.execute", return_value="executed output"):
                agent.run("run both")
            _, durable = agent.logger.load_session_messages(agent.session_id)

        live_assistant = next(message for message in agent.messages if message.get("tool_calls"))
        durable_assistant = next(message for message in durable if message.get("tool_calls"))
        live_commands = [
            json.loads(call["function"]["arguments"])["command"]
            for call in live_assistant["tool_calls"]
        ]
        durable_commands = [
            json.loads(call["function"]["arguments"])["command"]
            for call in durable_assistant["tool_calls"]
        ]
        self.assertEqual(live_commands, ["echo first edited", "echo second edited"])
        self.assertEqual(durable_commands, live_commands)

    def test_xml_approved_terminal_command_does_not_persist_empty_tool_calls(self):
        from agent import HermesCodingAgent
        from storage import TrajectoryLogger

        first = MagicMock()
        first.content = (
            '<tool_call>{"name":"run_terminal_command",'
            '"arguments":{"command":"echo proposed"}}</tool_call>'
        )
        first.tool_calls = None
        final = MagicMock()
        final.content = "done"
        final.tool_calls = None

        with tempfile.TemporaryDirectory() as tmp:
            agent = HermesCodingAgent(
                max_iterations=2,
                enable_skills=False,
                enable_memory=False,
                auto_learn_skills=False,
                auto_learn_memory=False,
                use_hermes_xml_protocol=True,
            )
            agent.logger = TrajectoryLogger(str(Path(tmp, "history.db")))
            agent.client.chat.completions.create = MagicMock(
                side_effect=[self._response(first), self._response(final)]
            )
            agent.prompt_user_for_command = MagicMock(
                return_value=(True, "echo edited", None)
            )
            with patch("agent.registry.execute", return_value="executed output"):
                agent.run("run XML command")
            _, legacy_messages = agent.logger.load_session_messages(agent.session_id)

        xml_assistant = next(
            message
            for message in legacy_messages
            if message["role"] == "assistant" and "<tool_call>" in message["content"]
        )
        self.assertNotIn("tool_calls", xml_assistant)

    def test_read_only_blocked_edit_is_not_reported_as_executed(self):
        from agent import HermesCodingAgent

        call = MagicMock()
        call.id = "call_blocked"
        call.function.name = "run_terminal_command"
        call.function.arguments = json.dumps({"command": "echo proposed"})
        first = self._native_message("", [call])
        final = self._native_message("done")
        agent = HermesCodingAgent(
            max_iterations=2,
            enable_skills=False,
            enable_memory=False,
            auto_learn_skills=False,
            auto_learn_memory=False,
            read_only=True,
        )
        agent.client.chat.completions.create = MagicMock(
            side_effect=[self._response(first), self._response(final)]
        )
        agent.prompt_user_for_command = MagicMock(
            return_value=(True, "echo edited", None)
        )

        agent.run("run it read-only")

        tool_result = next(message for message in agent.messages if message["role"] == "tool")
        self.assertIn("Not executed", tool_result["content"])
        self.assertIn("echo edited", tool_result["content"])
        self.assertNotIn("Executed: echo edited", tool_result["content"])


class TestQualifiedGlobAndSkillConfinement(unittest.TestCase):
    def test_path_qualified_glob_matches_relative_paths(self):
        from tools import find_files_by_pattern, terminal_session

        with tempfile.TemporaryDirectory() as tmp:
            previous_cwd = terminal_session.cwd
            terminal_session.cwd = tmp
            try:
                Path(tmp, "src", "nested").mkdir(parents=True)
                Path(tmp, "src", "top.ts").write_text("top", encoding="utf-8")
                Path(tmp, "src", "nested", "deep.ts").write_text("deep", encoding="utf-8")
                Path(tmp, "other.ts").write_text("other", encoding="utf-8")

                result = find_files_by_pattern("src/**/*.ts")

                self.assertIn(os.path.join("src", "top.ts"), result)
                self.assertIn(os.path.join("src", "nested", "deep.ts"), result)
                self.assertNotIn("other.ts", result)
            finally:
                terminal_session.cwd = previous_cwd

    def test_load_skill_cannot_escape_configured_repository(self):
        from skills import SkillStore

        with tempfile.TemporaryDirectory() as tmp:
            repository = Path(tmp, "skills")
            repository.mkdir()
            outside = Path(tmp, "outside.md")
            outside.write_text("secret outside instructions", encoding="utf-8")
            store = SkillStore(str(repository))

            result = store.load_skill(str(outside))

            self.assertIn("not found", result.lower())
            self.assertNotIn("secret outside instructions", result)

    def test_terminal_tool_description_claims_only_cwd_persistence(self):
        from tools import registry

        schema = next(
            item for item in registry.schemas
            if item["function"]["name"] == "run_terminal_command"
        )
        description = schema["function"]["description"]
        self.assertIn("Only the working directory", description)
        self.assertNotIn("persistent shell session", description.lower())

    def test_disabled_trajectory_export_reports_skipped(self):
        from agent import export_current_trajectory
        from storage import TrajectoryLogger

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp, "trajectory.jsonl")
            agent = MagicMock()
            agent.session_id = "session"
            agent.logger = TrajectoryLogger(
                str(Path(tmp, "history.db")), write_enabled=False
            )

            message = export_current_trajectory(agent, str(output_path))

        self.assertIn("skipped", message.lower())
        self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()
