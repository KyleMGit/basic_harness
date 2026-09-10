"""Actual agent completion and public subprocess/provider boundary tests."""
import json
from pathlib import Path
import sqlite3
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import agent as agent_module
from skills import AutoSkillExtractor
from skill_review import ReviewService
from test_async_skill_review import setup_roster, wait_for, rows, create_proposal, profile_bytes


def response(payload, reason="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason=reason, message=SimpleNamespace(content=payload))])


@pytest.mark.parametrize("payload,reason", [('{"action":"NONE"}', "length"), ('{"action":', "stop"),
    ('prefix {"action":"NONE"}', "stop"), ('{}', "stop"),
    (json.dumps(create_proposal() | {"complete":False}), "stop"),
    ('{"action":"UPDATE","action":"NONE"}', "stop"),
    (json.dumps(create_proposal() | {"profile_id":"forged"}), "stop")])
def test_snapshot_generator_rejects_invalid_or_length_terminated_output(payload, reason):
    client = MagicMock()
    client.with_options.return_value = client
    client.chat.completions.create.return_value = response(payload, reason)
    with pytest.raises(ValueError):
        AutoSkillExtractor.generate_proposal(client, "current-model", '{"catalog":{},"tasks":[]}', timeout=1, output_tokens=4096)


def test_snapshot_generator_uses_only_immutable_inputs_same_model_and_finite_limits():
    client = MagicMock()
    client.with_options.return_value = client
    client.chat.completions.create.return_value = response('{"action":"NONE"}')
    snapshot = '{"catalog":{"targets":[]},"tasks":[{"task_id":"bounded"}]}'
    assert AutoSkillExtractor.generate_proposal(client, "current-model", snapshot, timeout=2, output_tokens=4096) == {"action":"NONE"}
    call = client.with_options.return_value.chat.completions.create.call_args or client.chat.completions.create.call_args
    assert call.kwargs["model"] == "current-model"
    assert call.kwargs["max_tokens"] == 4096
    assert call.kwargs["messages"][-1]["content"] == snapshot


def make_agent(tmp_path, *, readonly=False, auto_memory=False):
    roster = setup_roster(tmp_path)
    previous = agent_module.skill_store.storage_dir
    try:
        agent_module.skill_store.storage_dir = str(tmp_path / "profile-0" / "skills")
        with patch.object(agent_module, "ACTIVE_HISTORY_DB", str(tmp_path / "history.db")):
            instance = agent_module.HermesCodingAgent(model=roster.model, base_url=roster.base_url,
                enable_memory=False, auto_learn_skills=True, auto_learn_memory=auto_memory,
                read_only=readonly, review_roster=roster, review_profile="user-0")
    finally:
        agent_module.skill_store.storage_dir = previous
    return instance, roster


def answer_message(text="The task answer."):
    message = MagicMock(content=text, tool_calls=None)
    message.model_dump.return_value = {"role":"assistant", "content":text}
    return message


def sql_tool_message(sql, call_id="verified-sql"):
    call = SimpleNamespace(id=call_id, function=SimpleNamespace(
        name="query_teradata", arguments=json.dumps({"sql": sql})))
    message = MagicMock(content="", tool_calls=[call])
    message.model_dump.return_value = {
        "role": "assistant", "content": "", "tool_calls": [{
            "id": call_id, "type": "function",
            "function": {"name": "query_teradata", "arguments": json.dumps({"sql": sql})},
        }],
    }
    return message


def verified_result():
    return json.dumps({"database": "warehouse", "columns": ["amount"],
                       "rows": [["private-result"]], "row_count": 1, "truncated": False})


def run_verified_correction(instance, task="No, that procedure is wrong; use the QUALIFY workaround"):
    replies = [sql_tool_message("SELECT * FROM sales QUALIFY ROW_NUMBER() OVER (ORDER BY sale_date DESC)=1"),
               answer_message()]
    with patch.object(instance, "step", side_effect=replies), patch.object(instance, "manage_context"), \
         patch.object(agent_module.registry, "execute", return_value=verified_result()):
        return instance.run(task)


def test_actual_run_returns_with_provider_blocked_answer_prompt_and_persistence_preserved(tmp_path):
    instance, roster = make_agent(tmp_path)
    entered, release = threading.Event(), threading.Event()
    service = ReviewService(roster, provider=lambda _: (entered.set(), release.wait(4), create_proposal())[2])
    worker = threading.Thread(target=service.run)
    worker.start()
    prompt = instance.messages[0]["content"]
    try:
        assert run_verified_correction(instance) == "The task answer."
        assert entered.wait(2)
        assert instance.messages[0]["content"] == prompt
        assert instance.logger.load_session_state(instance.session_id)["messages"][-1]["content"] == "The task answer."
        assert not instance.skill_store.get_all_skills()
        release.set()
        wait_for(lambda: "learned" in instance.skill_store.list_skills())
        assert instance.messages[0]["content"] == prompt
        assert "learned" in instance._build_system_prompt()
    finally:
        release.set()
        service.stop()
        worker.join(5)
        instance.shutdown_skill_reviews()


def test_actual_run_auto_memory_still_blocks_before_skill_enqueue(tmp_path):
    instance, _ = make_agent(tmp_path)
    instance.auto_learn_memory = True
    entered, release = threading.Event(), threading.Event()
    instance.memory_extractor = MagicMock()
    instance.memory_extractor.extract_and_update.side_effect = lambda **_: (entered.set(), release.wait(4), {})[2]
    answers = []
    try:
        with patch.object(instance, "step", return_value=answer_message()), patch.object(instance, "manage_context"):
            worker = threading.Thread(target=lambda: answers.append(instance.run("Memory is synchronous")))
            worker.start()
            assert entered.wait(2)
            assert worker.is_alive() and not answers
            assert not rows(instance.skill_review_owner, "evidence")
            release.set()
            worker.join(4)
            assert answers == ["The task answer."]
    finally:
        release.set()
        instance.shutdown_skill_reviews()


def test_incremental_capture_survives_actual_caller_compaction(tmp_path):
    instance, _ = make_agent(tmp_path)
    seen = []

    def compact():
        # Exercise the caller's capture boundary while the transcript is replaced.
        seen.extend(instance.messages[1:])
        instance.messages = instance.messages[:1] + [{"role":"user","content":"[compacted]"}]

    try:
        replies = [sql_tool_message("SELECT * FROM sales QUALIFY ROW_NUMBER() OVER (ORDER BY sale_date DESC)=1"), answer_message()]
        with patch.object(instance, "manage_context", side_effect=compact), patch.object(instance, "step", side_effect=replies), \
             patch.object(agent_module.registry, "execute", return_value=verified_result()):
            assert instance.run("No, the EARLY TASK EVIDENCE password=private-token was wrong; use QUALIFY instead") == "The task answer."
        saved = rows(instance.skill_review_owner, "episode_sources")
        assert "EARLY TASK EVIDENCE" in saved[0]["messages_json"]
        assert "private-token" not in saved[0]["messages_json"]
        assert "[compacted]" not in saved[0]["messages_json"]
    finally:
        instance.shutdown_skill_reviews()


@pytest.mark.parametrize("mode", ["read-only", "stateless", "no-skills"])
def test_agent_modes_revoke_running_request_and_initial_readonly_restores_optin(tmp_path, mode):
    instance, roster = make_agent(tmp_path, readonly=True)
    entered, release = threading.Event(), threading.Event()
    service = ReviewService(roster, provider=lambda _: (entered.set(), release.wait(4), create_proposal())[2])
    worker = None
    try:
        instance.set_testing_mode("normal")
        assert instance.auto_learn_skills
        run_verified_correction(instance)
        wait_for(lambda: len(rows(instance.skill_review_owner, "jobs")) == 1)
        worker = threading.Thread(target=service.once)
        worker.start()
        assert entered.wait(2)
        instance.set_testing_mode(mode)
        before = profile_bytes(instance.skill_review_owner)
        release.set()
        worker.join(4)
        time.sleep(.15)
        assert before == profile_bytes(instance.skill_review_owner)
        instance.set_testing_mode("normal")
        assert "learned" not in instance.skill_store.list_skills()
    finally:
        release.set()
        if worker:
            worker.join(5)
        instance.shutdown_skill_reviews()


def test_owner_is_not_started_when_session_prompt_construction_fails(tmp_path):
    with patch.object(agent_module.HermesCodingAgent, "_build_system_prompt", side_effect=RuntimeError("bad prompt")), \
         patch.object(agent_module, "Owner") as owner:
        with pytest.raises(RuntimeError, match="bad prompt"):
            make_agent(tmp_path)
        owner.assert_not_called()
