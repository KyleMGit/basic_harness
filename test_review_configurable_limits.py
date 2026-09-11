"""Focused contracts for configurable review-service byte guards."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from openai import OpenAI
import pytest

import skill_review as review
from review_diagnostics import ReviewDiagnosticError
from skill_review import BudgetRefusal, ReviewService, packed
from skills import AutoSkillExtractor
from test_async_skill_review import owner_for, rows, setup_roster
from test_skill_evidence_budget import eligible_evidence


SELECTED_DEFAULT = 256 * 1024
PREPARED_DEFAULT = 512 * 1024
WIRE_DEFAULT = 1024 * 1024


def sdk_client(observed):
    def respond(request):
        body = json.loads(request.content)
        observed.append((len(request.content), body["messages"][-1]["content"]))
        return httpx.Response(200, request=request, json={
            "id": "fake", "object": "chat.completion", "created": 1,
            "model": "test-model", "choices": [{
                "index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": '{"action":"NONE"}'},
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    http = httpx.Client(transport=httpx.MockTransport(respond))
    return OpenAI(api_key="synthetic", base_url="http://127.0.0.1:1/v1",
                  http_client=http, max_retries=0)


def test_service_defaults_are_actual_guards_and_output_default_is_unchanged():
    service = ReviewService(object())
    assert service.selected_view_bytes == SELECTED_DEFAULT
    assert service.prepared_input_bytes == PREPARED_DEFAULT
    assert service.wire_body_bytes == WIRE_DEFAULT
    assert service.output_tokens == 4096


@pytest.mark.parametrize("name,value", [
    ("selected_view_bytes", 0), ("selected_view_bytes", -1),
    ("selected_view_bytes", 1.5), ("selected_view_bytes", "1"),
    ("selected_view_bytes", True),
    ("prepared_input_bytes", 0), ("prepared_input_bytes", -1),
    ("prepared_input_bytes", 1.5), ("prepared_input_bytes", "1"),
    ("prepared_input_bytes", False),
    ("wire_body_bytes", 0), ("wire_body_bytes", -1),
    ("wire_body_bytes", 1.5), ("wire_body_bytes", "1"),
    ("wire_body_bytes", True),
])
def test_service_rejects_nonpositive_noninteger_and_bool_byte_guards(name, value):
    with pytest.raises(ValueError, match="byte"):
        ReviewService(object(), **{name: value})


def test_byte_guards_are_independent_and_output_ceiling_is_32768():
    service = ReviewService(
        object(), selected_view_bytes=3, prepared_input_bytes=2,
        wire_body_bytes=1, output_tokens=32768,
    )
    assert (service.selected_view_bytes, service.prepared_input_bytes,
            service.wire_body_bytes, service.output_tokens) == (3, 2, 1, 32768)
    with pytest.raises(ValueError, match="service limits"):
        ReviewService(object(), output_tokens=32769)


def test_sdk_configured_boundaries_measure_unicode_and_nested_escaping_exactly():
    prepared = json.dumps(
        {"catalog": {}, "tasks": [{"text": "é" + ("\\" * 70000)}]},
        ensure_ascii=False, separators=(",", ":"),
    )
    prepared_bytes = len(prepared.encode("utf-8"))
    wire_bytes = AutoSkillExtractor.sdk_wire_bytes("test-model", prepared)
    assert wire_bytes > prepared_bytes
    observed = []
    client = sdk_client(observed)
    try:
        assert AutoSkillExtractor.generate_proposal(
            client, "test-model", prepared, prepared_input_bytes=prepared_bytes,
            wire_body_bytes=wire_bytes,
        ) == {"action": "NONE"}
        assert observed == [(wire_bytes, prepared)]

        with pytest.raises(ReviewDiagnosticError) as prepared_error:
            AutoSkillExtractor.generate_proposal(
                client, "test-model", prepared,
                prepared_input_bytes=prepared_bytes - 1, wire_body_bytes=wire_bytes,
            )
        assert prepared_error.value.reason == "prepared_input_oversized"
        assert prepared_error.value.metadata["observed_bytes"] == prepared_bytes
        assert prepared_error.value.metadata["limit_bytes"] == prepared_bytes - 1

        with pytest.raises(ReviewDiagnosticError) as wire_error:
            AutoSkillExtractor.generate_proposal(
                client, "test-model", prepared,
                prepared_input_bytes=prepared_bytes, wire_body_bytes=wire_bytes - 1,
            )
        assert wire_error.value.reason == "wire_body_oversized"
        assert wire_error.value.metadata["observed_bytes"] == wire_bytes
        assert wire_error.value.metadata["limit_bytes"] == wire_bytes - 1
        assert observed == [(wire_bytes, prepared)]
    finally:
        client.close()


def test_nondefault_service_limits_reach_episode_selection_and_actual_sdk_transport(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    observed = []
    client = sdk_client(observed)
    try:
        item = eligible_evidence("large-configured-review", padding="\\" * 100000)
        assert owner.capture_turn(item).status == "ELIGIBLE"
        owner.flush_session(item.session_id)
        owner.pump()
        service = ReviewService(
            roster, selected_view_bytes=300 * 1024,
            prepared_input_bytes=600 * 1024, wire_body_bytes=1200 * 1024,
        )
        with patch("openai.OpenAI", return_value=client):
            assert service.once() == 1
        assert len(observed) == 1
        wire_bytes, prepared = observed[0]
        prepared_value = json.loads(prepared)
        selected_bytes = len(packed({"tasks": prepared_value["tasks"]}).encode())
        assert selected_bytes > 64 * 1024
        assert len(prepared.encode()) > 128 * 1024
        assert wire_bytes > 256 * 1024
        owner.pump()
        assert rows(owner, "jobs")[-1]["status"] == "NONE"
    finally:
        owner.close()


def test_selected_view_configured_boundary_is_exact_for_serialized_unicode(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    try:
        item = eligible_evidence("selected-exact", padding=("é\\" * 1000))
        assert owner.capture_turn(item).status == "ELIGIBLE"
        owner.flush_session(item.session_id)
        owner.pump()
        job = rows(owner, "jobs")[-1]
        prepared = ReviewService(roster)._episode_request(owner.entry, job)
        selected_bytes = len(packed({"tasks": json.loads(prepared)["tasks"]}).encode())

        exact = ReviewService(roster, selected_view_bytes=selected_bytes)
        assert exact._episode_request(owner.entry, job) == prepared
        with pytest.raises(BudgetRefusal) as error:
            ReviewService(
                roster, selected_view_bytes=selected_bytes - 1,
            )._episode_request(owner.entry, job)
        assert error.value.reason == "selected_view_bytes"
        assert error.value.observed == selected_bytes
        assert error.value.limit == selected_bytes - 1
    finally:
        owner.close()


@pytest.mark.parametrize("command,once", [("run", False), ("once", True)])
def test_run_and_once_cli_propagate_all_limit_flags(tmp_path, command, once):
    roster = setup_roster(tmp_path)
    captured = {}

    class Service:
        errors = {}
        _phase = "service.run"
        _phase_timing = {}

        def __init__(self, actual_roster, **kwargs):
            assert actual_roster is roster
            captured.update(kwargs)

        def stop(self):
            pass

        def run(self, *, once=False):
            captured["once"] = once
            return 0

    with patch.object(review, "load_roster", return_value=roster), \
         patch.object(review, "ReviewService", Service), \
         patch.object(review.signal, "signal"):
        assert review.main([
            command, "--roster", "synthetic.json",
            "--selected-view-bytes", "300001",
            "--prepared-input-bytes", "600002",
            "--wire-body-bytes", "1200003",
        ]) == 0
    assert captured["selected_view_bytes"] == 300001
    assert captured["prepared_input_bytes"] == 600002
    assert captured["wire_body_bytes"] == 1200003
    assert captured["once"] is once


@pytest.mark.parametrize("command", ["run", "once"])
def test_run_and_once_help_show_explicit_byte_defaults(command, capsys):
    with pytest.raises(SystemExit) as result:
        review.main([command, "--help"])
    assert result.value.code == 0
    output = capsys.readouterr().out
    normalized = " ".join(output.split())
    for flag, default in (
        ("--selected-view-bytes", SELECTED_DEFAULT),
        ("--prepared-input-bytes", PREPARED_DEFAULT),
        ("--wire-body-bytes", WIRE_DEFAULT),
    ):
        assert flag in output and f"default: {default}" in normalized
