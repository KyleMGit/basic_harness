"""Regression tests for bounded, separately configurable compaction calls."""

import json
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent import HermesCodingAgent, parse_args
from compaction import ContextManager


def _request_tokens(messages, output_tokens):
    """Mirror the harness's documented rough four-characters-per-token estimate."""
    chars = 0
    for message in messages:
        chars += len(str(message.get("content") or ""))
        if message.get("tool_calls"):
            chars += len(json.dumps(message["tool_calls"]))
    return max(1, chars // 4) + output_tokens


class BudgetEnforcingCompletions:
    def __init__(self, context_limit):
        self.context_limit = context_limit
        self.calls = []

    def create(self, **kwargs):
        output_tokens = kwargs.get("max_tokens")
        if output_tokens is None:
            raise AssertionError("compactor request omitted max_tokens")
        estimated = _request_tokens(kwargs["messages"], output_tokens)
        if estimated > self.context_limit:
            raise RuntimeError(
                f"provider context overflow: {estimated}>{self.context_limit}"
            )
        self.calls.append(kwargs)
        checkpoint = (
            "[CONTEXT COMPACTION — REFERENCE ONLY]\n"
            "The checkpoint below is historical background.\n\n"
            "## Historical Task Snapshot\nContinue the active task.\n\n"
            f"## Completed Actions\nBounded chunk {len(self.calls)} summarized."
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=checkpoint))]
        )


class CapturingClient:
    def __init__(self, context_limit):
        self.completions = BudgetEnforcingCompletions(context_limit)
        self.chat = SimpleNamespace(completions=self.completions)


def _native_history(group_count=10, payload_chars=520):
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Implement the requested feature."},
    ]
    for number in range(group_count):
        call_id = f"budget-call-{number}"
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": f"working on {call_id}",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": json.dumps({"path": f"file-{number}.py"}),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": f"result-for-{call_id}:" + ("x" * payload_chars),
                },
            ]
        )
    messages.extend(
        [
            {"role": "user", "content": "Keep going with the same task."},
            {"role": "assistant", "content": "Continuing now."},
        ]
    )
    return messages


class TestCompactorRequestBudget(unittest.TestCase):
    def _manager(self, **overrides):
        kwargs = {
            "max_context_tokens": 20_000,
            "trigger_threshold": 0.70,
            "keep_recent_turns": 2,
            "completion_reserve_tokens": 0,
            "compaction_max_context_tokens": 3_000,
            "compaction_output_tokens": 256,
        }
        kwargs.update(overrides)
        try:
            return ContextManager(**kwargs)
        except TypeError as exc:
            self.fail(f"ContextManager is missing compaction budget configuration: {exc}")

    def test_long_history_is_chunked_into_bounded_compactor_requests(self):
        manager = self._manager()
        client = CapturingClient(context_limit=3_000)
        original = _native_history()

        compacted, changed, status = manager.compact(
            client, "primary-model", original, current_step=20, force=True
        )

        self.assertTrue(changed, status)
        self.assertGreater(len(client.completions.calls), 1)
        self.assertLess(len(compacted), len(original))
        for call in client.completions.calls:
            self.assertEqual(call["max_tokens"], 256)
            self.assertLessEqual(
                _request_tokens(call["messages"], call["max_tokens"]), 3_000
            )

    def test_native_tool_groups_are_not_split_between_chunks(self):
        manager = self._manager()
        client = CapturingClient(context_limit=3_000)

        _, changed, status = manager.compact(
            client,
            "primary-model",
            _native_history(group_count=12),
            current_step=20,
            force=True,
        )

        self.assertTrue(changed, status)
        self.assertGreater(len(client.completions.calls), 1)
        prompts = [
            "\n".join(str(message.get("content") or "") for message in call["messages"])
            for call in client.completions.calls
        ]
        for number in range(12):
            call_id = f"budget-call-{number}"
            call_marker = f'"id": "{call_id}"'
            containing = [prompt for prompt in prompts if call_marker in prompt]
            self.assertEqual(
                len(containing),
                1,
                f"{call_id} should occur in exactly one compacted chunk",
            )
            self.assertIn(f"result-for-{call_id}:", containing[0])

    def test_xml_tool_groups_are_not_split_between_chunks(self):
        manager = self._manager()
        client = CapturingClient(context_limit=3_000)
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "Perform the XML task."},
        ]
        for number in range(10):
            marker = f"xml-call-{number}"
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": (
                            '<tool_call>{"name":"read_file","arguments":'
                            f'{{"marker":"{marker}"}}}}</tool_call>'
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            '<tool_response>{"tool_call_id":"hermes_call_0",'
                            f'"content":"result-for-{marker}:'
                            + ("x" * 520)
                            + '"}</tool_response>'
                        ),
                    },
                ]
            )
        messages.extend(
            [
                {"role": "user", "content": "Continue the XML task."},
                {"role": "assistant", "content": "Continuing."},
            ]
        )

        _, changed, status = manager.compact(
            client, "primary-model", messages, current_step=20, force=True
        )

        self.assertTrue(changed, status)
        self.assertGreater(len(client.completions.calls), 1)
        prompts = [
            "\n".join(str(message.get("content") or "") for message in call["messages"])
            for call in client.completions.calls
        ]
        for number in range(10):
            marker = f"xml-call-{number}"
            containing = [prompt for prompt in prompts if f'"marker":"{marker}"' in prompt]
            self.assertEqual(len(containing), 1)
            self.assertIn(f"result-for-{marker}:", containing[0])

    def test_chunked_checkpoint_host_preserves_all_historical_user_messages(self):
        manager = self._manager()
        client = CapturingClient(context_limit=3_000)
        messages = _native_history(group_count=12)
        messages[4:4] = [
            {"role": "user", "content": "Historical instruction alpha."}
        ]
        messages[17:17] = [
            {"role": "user", "content": "Historical instruction beta."}
        ]

        compacted, changed, status = manager.compact(
            client, "primary-model", messages, current_step=20, force=True
        )

        self.assertTrue(changed, status)
        self.assertGreater(len(client.completions.calls), 1)
        checkpoint = compacted[1]["content"]
        self.assertIn("Historical instruction alpha.", checkpoint)
        self.assertIn("Historical instruction beta.", checkpoint)

    def test_unfit_fixed_overhead_sends_no_provider_request_and_preserves_state(self):
        manager = self._manager(
            compaction_max_context_tokens=1_000,
            compaction_output_tokens=256,
        )
        client = CapturingClient(context_limit=1_000)
        original = _native_history(group_count=3)
        before_state = manager.snapshot_state()

        result, changed, status = manager.compact(
            client, "primary-model", original, current_step=20, force=True
        )

        self.assertFalse(changed)
        self.assertIs(result, original)
        self.assertEqual(client.completions.calls, [])
        self.assertEqual(manager.snapshot_state(), before_state)
        self.assertIn("budget", status.lower())

    def test_unfit_indivisible_native_group_sends_no_oversized_request(self):
        manager = self._manager()
        client = CapturingClient(context_limit=3_000)
        huge_args = json.dumps({"payload": "z" * 12_000})
        original = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "Do the task."},
            {
                "role": "assistant",
                "content": "calling tool",
                "tool_calls": [
                    {
                        "id": "huge-call",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": huge_args},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "huge-call",
                "content": "small result",
            },
            {"role": "user", "content": "Continue."},
            {"role": "assistant", "content": "Continuing."},
        ]
        before_state = manager.snapshot_state()

        result, changed, status = manager.compact(
            client, "primary-model", original, current_step=20, force=True
        )

        self.assertFalse(changed)
        self.assertIs(result, original)
        self.assertEqual(client.completions.calls, [])
        self.assertEqual(manager.snapshot_state(), before_state)
        self.assertIn("budget", status.lower())


class TestCompactorCallerConfiguration(unittest.TestCase):
    def test_invalid_compaction_budgets_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "compaction_max_context_tokens"):
            ContextManager(compaction_max_context_tokens=0)
        with self.assertRaisesRegex(ValueError, "compaction_output_tokens"):
            ContextManager(
                compaction_max_context_tokens=1_000,
                compaction_output_tokens=0,
            )
        with self.assertRaisesRegex(ValueError, "smaller"):
            ContextManager(
                compaction_max_context_tokens=1_000,
                compaction_output_tokens=1_000,
            )

    def test_agent_uses_compactor_model_only_for_compaction(self):
        try:
            agent = HermesCodingAgent(
                model="primary-model",
                compaction_model="summary-model",
                max_context_tokens=20_000,
                compaction_max_context_tokens=3_000,
                compaction_output_tokens=256,
                enable_memory=False,
                enable_skills=False,
                read_only=True,
            )
        except TypeError as exc:
            self.fail(f"HermesCodingAgent is missing compactor configuration: {exc}")

        client = CapturingClient(context_limit=3_000)
        agent.client = client
        agent.messages = _native_history()
        agent.step_counter = 20
        agent.manage_context(force=True)

        self.assertGreater(len(client.completions.calls), 1)
        self.assertTrue(
            all(call["model"] == "summary-model" for call in client.completions.calls)
        )

        normal_calls = []

        def normal_create(**kwargs):
            normal_calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="done", tool_calls=None),
                        finish_reason="stop",
                    )
                ]
            )

        agent.client.chat.completions.create = normal_create
        agent.step()
        self.assertEqual(normal_calls[-1]["model"], "primary-model")

    def test_cli_exposes_environment_backed_compactor_options(self):
        environment = {
            "AGENT_COMPACTION_MODEL": "summary-from-env",
            "AGENT_COMPACTION_MAX_TOKENS": "8192",
            "AGENT_COMPACTION_OUTPUT_TOKENS": "768",
        }
        with patch.dict(os.environ, environment, clear=False), patch.object(
            sys, "argv", ["agent.py"]
        ):
            args = parse_args()
        for attribute in (
            "compaction_model",
            "compaction_max_tokens",
            "compaction_output_tokens",
        ):
            if not hasattr(args, attribute):
                self.fail(f"CLI is missing compactor option {attribute}")
        self.assertEqual(args.compaction_model, "summary-from-env")
        self.assertEqual(args.compaction_max_tokens, 8192)
        self.assertEqual(args.compaction_output_tokens, 768)

        with patch.object(
            sys,
            "argv",
            [
                "agent.py",
                "--compaction-model",
                "summary-from-flag",
                "--compaction-max-tokens",
                "4096",
                "--compaction-output-tokens",
                "512",
            ],
        ):
            args = parse_args()
        self.assertEqual(args.compaction_model, "summary-from-flag")
        self.assertEqual(args.compaction_max_tokens, 4096)
        self.assertEqual(args.compaction_output_tokens, 512)


if __name__ == "__main__":
    unittest.main(verbosity=2)
