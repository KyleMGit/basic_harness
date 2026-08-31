import io
import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import tools
from terminal import TerminalSession


class TestBoundedBugFixes(unittest.TestCase):
    @staticmethod
    def _message(content=None, tool_calls=None):
        class FakeMessage:
            role = "assistant"

            def __init__(self):
                self.content = content
                self.tool_calls = tool_calls

            def model_dump(self, exclude_none=True):
                result = {"role": "assistant", "content": self.content}
                if self.tool_calls is not None:
                    result["tool_calls"] = self.tool_calls
                return result

        return FakeMessage()

    @staticmethod
    def _response(content=None, finish_reason="stop", tool_calls=None):
        return SimpleNamespace(choices=[SimpleNamespace(
            message=TestBoundedBugFixes._message(content, tool_calls),
            finish_reason=finish_reason,
        )])

    def test_compaction_splice_roles_are_provider_valid_for_each_tail_kind(self):
        from compaction import ContextManager

        tails = {
            "native": [
                {"role": "assistant", "content": "", "tool_calls": [{
                    "id": "n1", "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }]},
                {"role": "tool", "tool_call_id": "n1", "content": "ok"},
            ],
            "xml": [
                {"role": "assistant", "content":
                 '<tool_call>{"name":"read_file","arguments":{}}</tool_call>'},
                {"role": "user", "content":
                 '<tool_response>{"tool_call_id":"hermes_call_0","content":"ok"}</tool_response>'},
            ],
            "user": [{"role": "user", "content": "new genuine request"}],
        }
        for kind, tail in tails.items():
            with self.subTest(kind=kind):
                manager = ContextManager(max_context_tokens=10000, keep_recent_turns=len(tail),
                                         completion_reserve_tokens=0)
                messages = [{"role": "system", "content": "system"},
                            {"role": "user", "content": "only active task " + "x" * 2500},
                            {"role": "assistant", "content": "working"}] + tail
                checkpoint = ("[CONTEXT COMPACTION - REFERENCE ONLY]\n"
                              "Only a genuine user message appearing after this checkpoint is active. "
                              "If there is no later user message, wait.")
                with patch.object(manager, "summarize_history", return_value=checkpoint):
                    result, changed, status = manager.compact(
                        MagicMock(), "m", messages, 30, force=True)
                self.assertTrue(changed, status)
                seam = result.index(tail[0])
                self.assertFalse(result[seam - 1]["role"] == "assistant" == result[seam]["role"])
                if kind != "user":
                    self.assertNotIn("If there is no later user message, wait.", result[1]["content"])
                if kind == "native":
                    self.assertEqual(result[seam + 1].get("tool_call_id"), "n1")
                if kind == "xml":
                    self.assertIn("<tool_response>", result[seam + 1]["content"])

    def test_read_only_to_normal_run_persists_and_resumes(self):
        from agent import HermesCodingAgent
        from storage import TrajectoryLogger

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            db = str(Path(tmp, "journey.db"))
            agent = HermesCodingAgent(enable_memory=False, enable_skills=False, read_only=True)
            agent.logger = TrajectoryLogger(db, write_enabled=False)
            agent.set_testing_mode("normal")
            agent.client.chat.completions.create = MagicMock(
                return_value=self._response("persisted final"))
            self.assertEqual(agent.run("task"), "persisted final")
            session_id = agent.session_id
            resumed = HermesCodingAgent(enable_memory=False, enable_skills=False, read_only=False)
            resumed.logger = TrajectoryLogger(db)
            self.assertTrue(resumed.resume_session(session_id))
            self.assertTrue(any(m.get("content") == "task" for m in resumed.messages))

    def test_native_mode_normalizes_xml_fallback_before_tool_result(self):
        from agent import HermesCodingAgent

        agent = HermesCodingAgent(enable_memory=False, enable_skills=False, read_only=True,
                                  max_iterations=2)
        first = self._response(
            'thinking <tool_call>{"name":"read_file","arguments":{"file_path":"x"}}</tool_call>')
        second = self._response("valid final")

        def strict_create(**kwargs):
            if strict_create.calls:
                messages = kwargs["messages"]
                tool_index = next(i for i, m in enumerate(messages) if m.get("role") == "tool")
                assistant = messages[tool_index - 1]
                self.assertTrue(assistant.get("tool_calls"))
                self.assertNotIn("<tool_call>", assistant.get("content") or "")
                self.assertEqual(assistant["tool_calls"][0]["id"], messages[tool_index]["tool_call_id"])
            response = (first, second)[strict_create.calls]
            strict_create.calls += 1
            return response
        strict_create.calls = 0
        agent.client.chat.completions.create = strict_create
        with patch("agent.registry.execute", return_value="contents"):
            self.assertEqual(agent.run("read x"), "valid final")

    def test_empty_response_continues_to_valid_completion(self):
        from agent import HermesCodingAgent

        agent = HermesCodingAgent(enable_memory=False, enable_skills=False, read_only=True,
                                  max_iterations=2)
        agent.client.chat.completions.create = MagicMock(side_effect=[
            self._response(""), self._response("complete answer")])
        self.assertEqual(agent.run("task"), "complete answer")

    def test_length_truncation_preserves_partial_and_combines_continuation(self):
        from agent import HermesCodingAgent

        agent = HermesCodingAgent(enable_memory=False, enable_skills=False, read_only=True,
                                  max_iterations=2)
        agent.client.chat.completions.create = MagicMock(side_effect=[
            self._response("partial ", "length"), self._response("answer", "stop")])
        self.assertEqual(agent.run("task"), "partial answer")
        self.assertTrue(any("Continue the truncated response" in str(m.get("content"))
                            for m in agent.messages))

    def test_repeated_incomplete_responses_exhaust_without_completion(self):
        from agent import HermesCodingAgent

        for responses in ([self._response(""), self._response("")],
                          [self._response("part", "length"), self._response("ial", "length")]):
            with self.subTest(first_reason=responses[0].choices[0].finish_reason):
                agent = HermesCodingAgent(enable_memory=False, enable_skills=False, read_only=True,
                                          max_iterations=2)
                agent.client.chat.completions.create = MagicMock(side_effect=responses)
                agent.logger.end_session = MagicMock()
                result = agent.run("task")
                self.assertIn("incomplete", result.lower())
                completed = [call for call in agent.logger.end_session.call_args_list
                             if call.kwargs.get("status") == "COMPLETED"]
                self.assertEqual(completed, [])

    def test_iteration_limit_output_tells_user_to_type_continue(self):
        from agent import HermesCodingAgent

        agent = HermesCodingAgent(enable_memory=False, enable_skills=False, read_only=True,
                                  max_iterations=1)
        agent.client.chat.completions.create = MagicMock(return_value=self._response(""))

        output = io.StringIO()
        with redirect_stdout(output):
            result = agent.run("task")

        instruction = 'Type "Continue" to continue the analysis.'
        self.assertIn(instruction, output.getvalue())
        self.assertIn(instruction, result)

    def test_compaction_with_one_real_user_and_long_native_tool_loop(self):
        from compaction import ContextManager

        manager = ContextManager(max_context_tokens=10000, keep_recent_turns=6,
                                 completion_reserve_tokens=0)
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "Build the requested feature"},
        ]
        for number in range(8):
            call_id = f"call-{number}"
            messages.extend([
                {"role": "assistant", "content": f"running tool {number}", "tool_calls": [
                    {"id": call_id, "type": "function",
                     "function": {"name": "read_file", "arguments": "{}"}},
                ]},
                {"role": "tool", "tool_call_id": call_id,
                 "content": f"tool output {number} " + "x" * 400},
            ])
        messages.append({"role": "assistant", "content": "continuing implementation"})

        checkpoint = "[CONTEXT COMPACTION - REFERENCE ONLY]\nsummary"
        with patch.object(manager, "summarize_history", return_value=checkpoint):
            result, changed, status = manager.compact(MagicMock(), "m", messages, 20, force=True)

        self.assertTrue(changed, status)
        self.assertEqual(result[1]["content"], checkpoint)
        for index, message in enumerate(result):
            if message.get("tool_calls"):
                call_ids = {call["id"] for call in message["tool_calls"]}
                following = {
                    candidate.get("tool_call_id")
                    for candidate in result[index + 1:]
                    if candidate.get("role") == "tool"
                }
                self.assertTrue(call_ids <= following)

    def test_repeated_compaction_continues_single_user_native_tool_loop(self):
        from compaction import ContextManager

        manager = ContextManager(max_context_tokens=10000, keep_recent_turns=4,
                                 completion_reserve_tokens=0)
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "Build the requested feature"},
        ]
        wait_instruction = (
            "Only a genuine user message appearing after this checkpoint is active. "
            "If there is no later user message, wait."
        )
        continuation_instruction = (
            "The historical task snapshot is the active in-flight task; continue it "
            "without waiting for another user message."
        )

        checkpoint_round = 0

        def checkpoint(*_args, **_kwargs):
            nonlocal checkpoint_round
            checkpoint_round += 1
            if checkpoint_round == 1:
                wording = wait_instruction
            else:
                wording = ("Only a genuine user message\n"
                           " appearing after this checkpoint is active.  If there is no later user\n"
                           " message, wait.")
            return ("[CONTEXT COMPACTION — REFERENCE ONLY]\n" + wording +
                    "\n\n## Historical Task Snapshot\nBuild the requested feature")

        for round_number in range(2):
            for number in range(5):
                call_id = f"round-{round_number}-call-{number}"
                messages.extend([
                    {"role": "assistant", "content": f"running {call_id}", "tool_calls": [{
                        "id": call_id, "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }]},
                    {"role": "tool", "tool_call_id": call_id,
                     "content": "tool output " + "x" * 400},
                ])
            messages.append({"role": "assistant", "content": "continuing implementation"})
            with patch.object(manager, "summarize_history", side_effect=checkpoint):
                messages, changed, status = manager.compact(
                    MagicMock(), "m", messages, 30 + round_number, force=True)

            self.assertTrue(changed, status)
            checkpoint_text = messages[1]["content"]
            normalized_checkpoint = " ".join(checkpoint_text.split())
            self.assertNotIn(wait_instruction, normalized_checkpoint,
                             f"round {round_number + 1}")
            self.assertIn(continuation_instruction, normalized_checkpoint,
                          f"round {round_number + 1}")
            for index, message in enumerate(messages):
                if message.get("role") == "tool":
                    self.assertGreater(index, 0)
                    previous = messages[index - 1]
                    self.assertEqual(previous.get("role"), "assistant")
                    self.assertIn(
                        message.get("tool_call_id"),
                        {call["id"] for call in previous.get("tool_calls", [])},
                    )

    def test_compaction_keeps_active_user_and_xml_response_with_originating_call(self):
        from compaction import ContextManager

        manager = ContextManager(max_context_tokens=10000, keep_recent_turns=2,
                                 completion_reserve_tokens=0)
        active_user = {"role": "user", "content": "CURRENT ACTION: inspect both files"}
        xml_call = {"role": "assistant", "content": (
            '<tool_call>{"name":"read_file","arguments":{"file_path":"a.py"}}</tool_call>'
        )}
        xml_response = {"role": "user", "content": (
            '<tool_response>{"tool_call_id":"xml-1","content":"contents"}</tool_response>'
        )}
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "old task " + "x" * 3000},
            {"role": "assistant", "content": "old answer"},
            active_user,
            xml_call,
            xml_response,
            {"role": "assistant", "content": "parallel reads", "tool_calls": [
                {"id": "native-a", "type": "function",
                 "function": {"name": "read_file", "arguments": "{}"}},
                {"id": "native-b", "type": "function",
                 "function": {"name": "read_file", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "native-a", "content": "one"},
            {"role": "tool", "tool_call_id": "native-b", "content": "two"},
            {"role": "assistant", "content": "analyzing results"},
        ]

        checkpoint = "[CONTEXT COMPACTION - REFERENCE ONLY]\nsummary"
        with patch.object(manager, "summarize_history", return_value=checkpoint):
            result, changed, status = manager.compact(MagicMock(), "m", messages, 21, force=True)

        self.assertTrue(changed, status)
        tail = result[3:]
        self.assertIn(active_user, tail)
        if xml_response in tail:
            self.assertIn(xml_call, tail[:tail.index(xml_response)])

    def test_compaction_preserves_active_user_and_valid_tool_groups(self):
        from compaction import ContextManager

        manager = ContextManager(max_context_tokens=10000, keep_recent_turns=3,
                                 completion_reserve_tokens=0)
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "old task " + "x" * 3000},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "CURRENT ACTION: run both reads"},
            {"role": "assistant", "content": "calling", "tool_calls": [
                {"id": "a", "function": {"name": "read_file", "arguments": "{}"}},
                {"id": "b", "function": {"name": "read_file", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "a", "content": "one"},
            {"role": "tool", "tool_call_id": "b", "content": "two"},
            {"role": "assistant", "content": "tool analysis"},
        ]
        checkpoint = "[CONTEXT COMPACTION — REFERENCE ONLY]\nheader"
        with patch.object(manager, "summarize_history", return_value=checkpoint):
            result, changed, status = manager.compact(MagicMock(), "m", messages, 9, force=True)
        self.assertTrue(changed, status)
        self.assertEqual(result[3:], messages[3:])
        self.assertTrue(any(m.get("content") == "CURRENT ACTION: run both reads" for m in result[3:]))
        self.assertNotEqual(result[2]["role"], result[3]["role"])
        for index, message in enumerate(result):
            if message.get("tool_calls"):
                ids = {call["id"] for call in message["tool_calls"]}
                following = {m.get("tool_call_id") for m in result[index + 1:] if m.get("role") == "tool"}
                self.assertTrue(ids.isdisjoint(following) or ids <= following)

    def test_disabled_memory_is_not_consulted_by_compaction(self):
        from agent import HermesCodingAgent

        agent = HermesCodingAgent(enable_memory=False, enable_skills=False, read_only=True)
        agent.messages += [{"role": "user", "content": "old " + "x" * 3000},
                           {"role": "assistant", "content": "answer"},
                           {"role": "user", "content": "active"}]
        response = MagicMock(choices=[MagicMock(message=MagicMock(
            content="[CONTEXT COMPACTION — REFERENCE ONLY]\nsummary"))])
        agent.client.chat.completions.create = MagicMock(return_value=response)
        with patch("memory.project_memory_manager.load_memory", return_value="PERSISTENT_PROJECT_SENTINEL") as project_read, \
             patch("memory.user_profile_manager.load_profile", return_value="PERSISTENT_USER_SENTINEL") as user_read:
            agent.context_manager.keep_recent_turns = 1
            agent.context_manager.max_context_tokens = 10000
            agent.manage_context(force=True)
        project_read.assert_not_called()
        user_read.assert_not_called()
        sent = agent.client.chat.completions.create.call_args.kwargs["messages"]
        self.assertNotIn("MEMORY.md", str(sent))
        self.assertNotIn("USER.md", str(sent))
        self.assertNotIn("PERSISTENT_", str(sent))

    def test_enabled_memory_read_failure_falls_back_during_forced_compaction(self):
        from agent import HermesCodingAgent

        agent = HermesCodingAgent(enable_memory=True, enable_skills=False, read_only=True)
        agent.messages += [{"role": "user", "content": "old " + "x" * 3000},
                           {"role": "assistant", "content": "answer"},
                           {"role": "user", "content": "active"}]
        response = MagicMock(choices=[MagicMock(message=MagicMock(
            content="[CONTEXT COMPACTION - REFERENCE ONLY]\nsummary"))])
        agent.client.chat.completions.create = MagicMock(return_value=response)
        agent.context_manager.keep_recent_turns = 1
        agent.context_manager.max_context_tokens = 10000
        with patch("agent.project_memory_manager.load_memory", side_effect=OSError("read failed")), \
             patch("agent.user_profile_manager.load_profile", side_effect=OSError("read failed")):
            agent.manage_context(force=True)
        sent = agent.client.chat.completions.create.call_args.kwargs["messages"]
        self.assertIn("<MEMORY_PROVIDER_CONTEXT>\nNone.\n</MEMORY_PROVIDER_CONTEXT>", sent[1]["content"])

    def test_resume_repairs_each_unanswered_native_tool_call_once(self):
        from agent import HermesCodingAgent

        calls = [{"id": "a", "type": "function", "function": {"name": "one", "arguments": "{}"}},
                 {"id": "b", "type": "function", "function": {"name": "two", "arguments": "{}"}}]
        restored = [{"role": "user", "content": "task"},
                    {"role": "assistant", "content": "", "tool_calls": calls},
                    {"role": "tool", "tool_call_id": "a", "content": "done"}]
        agent = HermesCodingAgent(enable_memory=False, enable_skills=False, read_only=True)
        agent.logger.load_session_messages = MagicMock(return_value=("task", restored))
        agent.logger.load_session_state = MagicMock(return_value=None)
        self.assertTrue(agent.resume_session("s"))
        self.assertEqual([m.get("tool_call_id") for m in agent.messages if m.get("role") == "tool"], ["a", "b"])
        agent.logger.load_session_messages.return_value = ("task", agent.messages[1:])
        self.assertTrue(agent.resume_session("s"))
        self.assertEqual([m.get("tool_call_id") for m in agent.messages if m.get("role") == "tool"], ["a", "b"])

    def test_xml_missing_name_and_non_object_arguments_are_protocol_repairs(self):
        from protocol import ToolProtocol

        payloads = [
            '<tool_call>{"arguments":{"file_path":"x"}}</tool_call>',
            '<tool_call>{"name":"read_file","arguments":[]}</tool_call>',
            '<tool_call>{"name":"read_file","arguments":"7"}</tool_call>',
        ]
        for payload in payloads:
            with self.subTest(payload=payload):
                _, calls = ToolProtocol.extract_tool_calls(payload)
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0]["name"], "__protocol_error__")
                self.assertEqual(calls[0]["id"], "hermes_call_0")
                self.assertIsInstance(calls[0]["arguments"], dict)

    def test_agent_loop_repairs_malformed_xml_then_reaches_next_model_step(self):
        from agent import HermesCodingAgent

        malformed_payloads = [
            '<tool_call>{"arguments":{"file_path":"x"}}</tool_call>',
            '<tool_call>{"name":"read_file","arguments":[]}</tool_call>',
        ]
        for malformed in malformed_payloads:
            with self.subTest(malformed=malformed):
                agent = HermesCodingAgent(enable_memory=False, enable_skills=False, read_only=True,
                                          use_hermes_xml_protocol=True, max_iterations=2)
                agent.step = MagicMock(side_effect=[SimpleNamespace(content=malformed),
                                                    SimpleNamespace(content="repaired final")])
                result = agent.run("test malformed XML repair")
                self.assertEqual(result, "repaired final")
                self.assertNotEqual(result, "[Task Finished]")
                self.assertEqual(agent.step.call_count, 2)
                repairs = [m for m in agent.messages if m.get("role") == "user"
                           and "<tool_response>" in str(m.get("content"))]
                self.assertEqual(len(repairs), 1)
                self.assertIn('"tool_call_id": "hermes_call_0"', repairs[0]["content"])
                self.assertIn("Protocol repair required", repairs[0]["content"])

    def test_summary_host_canonicalizes_bogus_deterministic_sections(self):
        from compaction import ContextManager

        manager = ContextManager()
        client = MagicMock()
        bogus_checkpoint = ("[CONTEXT COMPACTION - REFERENCE ONLY]\n## Goal\nkeep going\n\n"
                            "## Exact Recovery Anchors\n- Paths: ./hallucinated.py\n\n"
                            "## Verbatim Historical User Messages\n- User Turn 999: bogus model body\n\n"
                            "## Next Steps\ncontinue")
        client.chat.completions.create.return_value = MagicMock(choices=[MagicMock(message=MagicMock(
            content="[CONTEXT COMPACTION — REFERENCE ONLY]\n## Goal\nkeep going"))])
        client.chat.completions.create.return_value.choices[0].message.content = bogus_checkpoint
        checkpoint = manager.summarize_history(client, "m", [
            {"role": "user", "content": "inspect ./src/app.py at deadbeef; token=FAKE_SECRET_1"}
        ], memory_context="None.")
        self.assertEqual(checkpoint.count("## Exact Recovery Anchors"), 1)
        self.assertEqual(checkpoint.count("## Verbatim Historical User Messages"), 1)
        self.assertIn("./src/app.py", checkpoint)
        self.assertIn("inspect ./src/app.py", checkpoint)
        self.assertNotIn("./hallucinated.py", checkpoint)
        self.assertNotIn("bogus model body", checkpoint)
        self.assertNotIn("FAKE_SECRET_1", checkpoint)

    def test_search_skips_discovered_link_escape(self):
        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as outside:
            old = tools.terminal_session
            tools.terminal_session = TerminalSession(workspace)
            try:
                Path(workspace, "inside.txt").write_text("needle", encoding="utf-8")
                Path(outside, "outside.txt").write_text("needle", encoding="utf-8")
                link = Path(workspace, "escape")
                try:
                    link.symlink_to(outside, target_is_directory=True)
                except OSError:
                    import subprocess
                    made = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), outside],
                                          capture_output=True, text=True)
                    if made.returncode:
                        self.skipTest("directory link/junction creation genuinely unavailable")
                grep = tools.grep_search("needle", ".")
                found = tools.find_files_by_pattern("**/*.txt", ".")
                self.assertIn("inside.txt", grep + found)
                self.assertNotIn("outside.txt", grep + found)
                self.assertNotIn("escape", grep + found)
            finally:
                tools.terminal_session = old

    def test_all_persistence_boundaries_redact_without_mutating_live_values(self):
        from storage import TrajectoryLogger

        sentinel = "FAKE_TOKEN_PERSIST_92"
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            db = Path(tmp, "history.db")
            export = Path(tmp, "trajectory.jsonl")
            logger = TrajectoryLogger(str(db))
            task = f"token={sentinel}"
            nested_arguments = {"outer": {"Authorization": sentinel,
                                            "client_secret": sentinel,
                                            "tokens": [{"refresh_token": sentinel}]},
                                "password": sentinel}
            calls = [{"id": "c", "function": {"name": "x", "arguments": json.dumps(nested_arguments)}}]
            messages = [{"role": "assistant", "content": f"secret={sentinel}", "tool_calls": calls},
                        {"role": "tool", "tool_call_id": "c", "content": f"token={sentinel}"}]
            state = {"previous_checkpoint": f"api_key={sentinel}",
                     "provider": {"Authorization": sentinel,
                                  "client_secret": sentinel,
                                  "refresh_token": sentinel}}
            original_calls = json.loads(json.dumps(calls))
            original_messages = json.loads(json.dumps(messages))
            original_state = json.loads(json.dumps(state))
            logger.start_session("s", task, f"authorization: bearer {sentinel}")
            logger.log_step("s", 1, "assistant", messages[0]["content"], calls)
            logger.log_step("s", 2, "tool", messages[1]["content"], tool_call_id="c")
            logger.save_session_state("s", messages, 2, state)
            logger.export_jsonl("s", str(export))
            self.assertNotIn(sentinel, export.read_text(encoding="utf-8"))
            self.assertIn(sentinel, task + json.dumps(calls) + json.dumps(messages) + json.dumps(state))
            self.assertEqual(calls, original_calls)
            self.assertEqual(messages, original_messages)
            self.assertEqual(state, original_state)
            with sqlite3.connect(db) as conn:
                persisted = " ".join(str(row) for table in ("sessions", "steps", "session_state")
                                     for row in conn.execute(f"SELECT * FROM {table}"))
            self.assertNotIn(sentinel, persisted)


if __name__ == "__main__":
    unittest.main()
