"""Caller-visible diagnostics using temporary mailboxes and local fake providers."""
from contextlib import contextmanager
from dataclasses import replace
import json
import sqlite3
import threading
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import httpx
from openai import APITimeoutError, OpenAI
import pytest

import agent
import skill_review as review
from review_diagnostics import ReviewDiagnosticError
from skills import AutoSkillExtractor
from test_async_skill_review import evidence, owner_for, profile_bytes, rows, setup_roster


PRIVATE = "PRIVATE_BODY password=secret C:/private/profile SELECT * FROM evidence\n\x1b[31mFORGED"


@pytest.fixture
def mailbox(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    try:
        yield owner, roster
    finally:
        owner.close()


def capture(**kwargs):
    item = review.Evidence("session", **kwargs)
    item.add({"role": "user", "content": "Do the task"})
    item.add({"role": "assistant", "content": "Done"})
    return item


def chat(owner, item):
    instance = object.__new__(agent.HermesCodingAgent)
    instance.auto_learn_skills = True
    instance.read_only = instance.stateless = False
    instance.skill_review_owner = owner
    instance._task_evidence = item
    if owner is None:
        instance.run_auto_skill_synthesis("unused")
    else:
        # These diagnostics intentionally exercise the preserved legacy direct
        # enqueue contract; selective caller behavior has dedicated integration tests.
        with patch.object(owner, "capture_turn", side_effect=owner.enqueue):
            instance.run_auto_skill_synthesis("unused")
    assert instance._task_evidence is None
    return instance


def assert_private(output):
    for forbidden in ("PRIVATE_BODY", "password=secret", "C:/private", "SELECT *", "FORGED", "\x1b", "Traceback"):
        assert forbidden not in output


def assert_refusal(output):
    assert "NOT queued" in output
    assert "no automatic retry" in output
    assert "earlier queue work retained" in output
    assert_private(output)


@pytest.mark.parametrize("kind,expected", [
    ("raw", ("capture.traversal", "raw_bytes", "observed_bytes=", "limit_bytes=256")),
    ("serialized", ("capture.serialized", "serialized_bytes", "observed_bytes=", "limit_bytes=256")),
    ("messages", ("capture.serialized", "message_count", "observed_messages=3", "limit_messages=2")),
    ("depth", ("capture.traversal", "depth", "observed_depth=25", "limit_depth=24")),
    ("nodes", ("capture.traversal", "nodes", "observed_nodes=8193", "limit_nodes=8192")),
    ("unsupported", ("capture.traversal", "unsupported_data")),
    ("incomplete", ("capture.completion", "incomplete_completion")),
])
def test_chat_reports_specific_capture_refusals(mailbox, capsys, kind, expected):
    owner, _ = mailbox
    assert owner.enqueue(evidence(task="earlier")).status == "ACCEPTED"
    item = review.Evidence("session", max_bytes=256 if kind in ("raw", "serialized") else review.TURN_BYTES,
                           max_messages=2 if kind == "messages" else review.TURN_MESSAGES)
    content = "x" * 1000 if kind == "raw" else "é" * 100 if kind == "serialized" else "task"
    if kind == "depth":
        for _ in range(25):
            content = [content]
    elif kind == "nodes":
        content = [None] * 9000
    elif kind == "unsupported":
        content = {"unsupported set"}
    item.add({"role": "user", "content": content})
    if kind != "incomplete":
        item.add({"role": "assistant", "content": "Done"})
    if kind == "messages":
        item.add({"role": "assistant", "content": "Third"})
    chat(owner, item)
    output = capsys.readouterr().out
    for token in expected:
        assert token in output
    assert_refusal(output)
    assert [row["task_id"] for row in rows(owner, "evidence")] == ["earlier"]


def test_capture_keeps_256_kib_limit_and_stops_traversing_after_refusal(mailbox, capsys):
    owner, _ = mailbox
    item = review.Evidence("session")
    item.add({"role": "user", "content": "x" * 300000})
    class Untouchable(dict):
        def items(self):
            pytest.fail("Traversal continued after the first refusal")
    item.add({"role": "assistant", "content": Untouchable()})
    chat(owner, item)
    output = capsys.readouterr().out
    assert f"limit_bytes={review.TURN_BYTES}" in output and "observed_bytes=" in output
    assert_refusal(output)


@pytest.mark.parametrize("kind", ["records", "bytes", "task_bytes"])
def test_chat_reports_queue_and_admission_byte_bounds(mailbox, capsys, kind):
    owner, _ = mailbox
    if kind == "records":
        owner.max_records = 1
        owner.enqueue(evidence(task="old"))
    elif kind == "bytes":
        for index in range(115):
            assert owner.enqueue(evidence(task=str(index), text="x" * 68000)).status == "ACCEPTED"
    before = rows(owner, "evidence")
    item = capture()
    if kind == "task_bytes":
        with patch.object(item, "finish", return_value=review.EvidenceResult("READY", "s", "t", "x" * (review.TURN_BYTES + 1))):
            chat(owner, item)
    else:
        if kind == "bytes":
            item = review.Evidence("s")
            item.add({"role": "user", "content": "x" * 68000})
            item.add({"role": "assistant", "content": "Done"})
        chat(owner, item)
    output = capsys.readouterr().out
    assert f"reason={'queue_' + kind if kind != 'task_bytes' else 'serialized_bytes'}" in output
    assert "stage=admission" in output
    assert ("limit_records=1" if kind == "records" else f"limit_bytes={review.MAX_QUEUE_BYTES - review.QUEUE_HEADROOM_BYTES}" if kind == "bytes" else f"limit_bytes={review.TURN_BYTES}") in output
    assert "observed_" in output
    assert_refusal(output)
    assert rows(owner, "evidence") == before


@pytest.mark.parametrize("failure", ["lock", "sqlite", "other"])
def test_chat_admission_failure_class_and_stage_are_safe(mailbox, capsys, failure):
    owner, _ = mailbox
    point, error = {
        "lock": ("gate", TimeoutError(PRIVATE)),
        "sqlite": ("connect", sqlite3.OperationalError(PRIVATE)),
        "other": ("read_auth", ValueError(PRIVATE)),
    }[failure]
    with patch.object(review, point, side_effect=error):
        chat(owner, capture())
    output = capsys.readouterr().out
    assert f"stage=admission.{ {'lock': 'lock', 'sqlite': 'sqlite', 'other': 'authorization'}[failure]}" in output
    assert f"error={type(error).__name__}" in output
    if failure == "lock":
        assert "coordination_timeout_s=0.5" in output
    if failure == "sqlite":
        assert "sqlite_busy_timeout_s=0.5" in output
    assert_refusal(output)


def test_chat_actual_sqlite_contention_retains_queue(mailbox, capsys):
    owner, _ = mailbox
    owner.enqueue(evidence(task="old"))
    before = rows(owner, "evidence")
    with sqlite3.connect(owner.entry.mailbox) as blocker:
        blocker.execute("BEGIN EXCLUSIVE")
        chat(owner, capture())
    output = capsys.readouterr().out
    assert "stage=admission.sqlite" in output and "error=OperationalError" in output
    assert_refusal(output)
    assert rows(owner, "evidence") == before


def test_chat_accepted_and_auto_discovery_guidance(mailbox, capsys):
    owner, _ = mailbox
    chat(owner, capture())
    accepted = capsys.readouterr().out
    assert "ACCEPTED" in accepted and "admission only" in accepted and "publication" in accepted
    chat(None, capture())
    unavailable = capsys.readouterr().out
    assert "--auto-skills" in unavailable and "--profiles-dir" in unavailable
    assert "configure a host-authorized review roster" not in unavailable
    assert_refusal(unavailable)


def test_chat_unexpected_exception_name_cannot_inject_logs(mailbox, capsys):
    owner, _ = mailbox
    evil = type(PRIVATE, (RuntimeError,), {})
    with patch.object(owner, "enqueue", side_effect=evil(PRIVATE)):
        chat(owner, capture())
    output = capsys.readouterr().out
    assert "stage=episode.admission" in output and "error=RuntimeError" in output
    assert_refusal(output)


def prepared(owner):
    assert owner.enqueue(evidence()).status == "ACCEPTED"
    owner.pump()
    return rows(owner, "jobs")[0]


def run_scans(service, count=4):
    refresh = service._refresh
    scans = 0
    def bounded_run():
        nonlocal scans
        refresh()
        scans += 1
        if scans >= count:
            service.stop()
    with patch.object(service, "_refresh", side_effect=bounded_run):
        return service.run()


@pytest.mark.parametrize("once", [True, False])
def test_service_pending_actual_contention_console_and_retry_guidance(mailbox, capsys, once):
    owner, roster = mailbox
    prepared(owner)
    service = review.ReviewService(roster, provider=lambda _: pytest.fail("Unexpected inference"))
    entered, release = threading.Event(), threading.Event()
    def hold_gate():
        with review.gate(owner.entry):
            entered.set()
            release.wait(4)
    thread = threading.Thread(target=hold_gate)
    thread.start()
    try:
        assert entered.wait(2)
        assert (service.once() if once else run_scans(service)) == 0
    finally:
        release.set()
        thread.join(3)
    output = capsys.readouterr().err
    assert output.count("stage=pending.scan") == 1
    assert "error=TimeoutError" in output and "profile=user-0" in output
    assert "coordination_timeout_s=0.5" in output
    assert ("no automatic future run" if once else "automatic future run-loop retry") in output
    assert "provider_timeout" not in output
    assert rows(owner, "jobs")[0]["status"] == "PREPARED"


@pytest.mark.parametrize("point,stage,state", [
    ("_claim", "claim", "PREPARED"),
    ("_infer", "worker.authorization", "RUNNING"),
    ("_finish", "result.persistence", "RUNNING"),
])
@pytest.mark.parametrize("once", [True, False])
def test_service_console_identifies_interrupted_stage(mailbox, capsys, point, stage, state, once):
    owner, roster = mailbox
    job = prepared(owner)
    service = review.ReviewService(roster, provider=lambda _: {"action": "NONE"}, timeout=1.25)
    with patch.object(service, point, side_effect=TimeoutError(PRIVATE)):
        service.once() if once else run_scans(service, 8)
    output = capsys.readouterr().err
    assert output.count(f"stage={stage}") == 1
    assert "error=TimeoutError" in output and "profile=user-0" in output
    assert f"job={job['job_id']}" in output
    assert "provider_timeout" not in output
    if state == "RUNNING":
        assert "restart the service" in output and "no active retry" in output
    else:
        assert ("no automatic future run" if once else "automatic future run-loop retry") in output
    assert_private(output + json.dumps(service.errors))
    assert rows(owner, "jobs")[0]["status"] == state
    assert rows(owner, "evidence")


def test_service_recovery_failure_is_separate_from_pending_scan(mailbox, capsys):
    owner, roster = mailbox
    prepared(owner)
    with sqlite3.connect(owner.entry.mailbox) as conn:
        conn.execute("UPDATE jobs SET status='RUNNING',service_generation='old'")
    original = review.connect
    def fail_recovery(entry, readonly=False, **kwargs):
        if not readonly:
            raise sqlite3.OperationalError(PRIVATE)
        return original(entry, readonly=readonly, **kwargs)
    with patch.object(review, "connect", side_effect=fail_recovery):
        assert review.ReviewService(roster, provider=lambda _: {}).once() == 0
    output = capsys.readouterr().err
    assert "stage=pending.recovery" in output and "error=OperationalError" in output
    assert "sqlite_busy_timeout_s=0.5" in output
    assert_private(output)
    assert rows(owner, "jobs")[0]["status"] == "RUNNING"


@pytest.mark.parametrize("kind,status,stage", [
    ("sdk_timeout", "FAILED", "provider.inference"),
    ("provider_value", "INVALID", "provider.inference_validation"),
    ("invalid_proposal", "INVALID", "provider.validation"),
])
def test_provider_failures_are_visible_and_safe_in_console_and_metadata(mailbox, capsys, kind, status, stage):
    owner, roster = mailbox
    job = prepared(owner)
    def provider(_):
        if kind == "sdk_timeout":
            raise APITimeoutError(request=httpx.Request("POST", "http://127.0.0.1:1/private", headers={"Authorization": PRIVATE}))
        if kind == "provider_value":
            raise ValueError(PRIVATE)
        return {"action": PRIVATE}
    service = review.ReviewService(roster, provider=provider, timeout=1.25)
    assert service.once() == 1
    output = capsys.readouterr().err
    assert f"stage={stage}" in output and f"outcome={status}" in output
    assert "profile=user-0" in output and f"job={job['job_id']}" in output
    if kind == "sdk_timeout":
        assert "error=APITimeoutError" in output and "provider_timeout_s=1.25" in output
    assert "no automatic provider retry" in output and "NONE" not in output
    owner.pump()
    saved = rows(owner, "jobs")[0]
    assert saved["status"] == status
    assert_private(output + json.dumps(saved) + json.dumps(service.errors))
    assert not rows(owner, "evidence")


def fake_sdk_response(payload, reason="stop", usage=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(finish_reason=reason, message=SimpleNamespace(content=payload))],
        usage=usage,
    )


@pytest.mark.parametrize("payload,finish,reason", [
    ('{"action":"NONE"}', "length", "non_stop_finish"),
    ('{"action":', "stop", "malformed_json"),
    ('{"action":"UPDATE","action":"NONE"}', "stop", "duplicate_fields"),
    ('{"action":"CREATE","unexpected":true}', "stop", "invalid_action_or_fields"),
])
def test_builtin_adapter_reason_reaches_stderr_and_owner_persisted_detail(
        mailbox, capsys, payload, finish, reason):
    owner, roster = mailbox
    job = prepared(owner)
    client = MagicMock()
    client.with_options.return_value = client
    client.chat.completions.create.return_value = fake_sdk_response(
        payload, finish,
        SimpleNamespace(prompt_tokens=123, completion_tokens=17, total_tokens=140),
    )
    with patch("openai.OpenAI", return_value=client):
        assert review.ReviewService(roster, timeout=1.25, output_tokens=512).once() == 1
    assert client.chat.completions.create.call_count == 1
    output = capsys.readouterr().err
    for token in (
        "stage=provider.inference_validation", "error=ValueError",
        f"reason={reason}", "request_attempted=true",
        "response_received=true", "output_tokens=512", "usage_prompt_tokens=123",
        "usage_completion_tokens=17", "usage_total_tokens=140", "outcome=INVALID",
    ):
        assert token in output
    owner.pump()
    saved = rows(owner, "jobs")[0]
    assert saved["status"] == "INVALID"
    assert f"reason={reason}" in saved["detail"]
    if finish == "length":
        assert "finish_reason=length" in output
    assert f"job={job['job_id']}" in saved["detail"]
    assert not rows(owner, "evidence")
    assert_private(output + json.dumps(saved))


def test_builtin_adapter_api_timeout_stays_failed_with_attempt_facts(mailbox, capsys):
    owner, roster = mailbox
    prepared(owner)
    client = MagicMock()
    client.with_options.return_value = client
    client.chat.completions.create.side_effect = APITimeoutError(
        request=httpx.Request("POST", "http://127.0.0.1:1/private", headers={"Authorization": PRIVATE}))
    with patch("openai.OpenAI", return_value=client):
        assert review.ReviewService(roster, timeout=1.25).once() == 1
    output = capsys.readouterr().err
    assert client.chat.completions.create.call_count == 1
    assert "stage=provider.inference" in output and "error=APITimeoutError" in output
    assert "request_attempted=true" in output and "response_received=" not in output
    assert "provider_timeout_s=1.25" in output and "outcome=FAILED" in output
    owner.pump()
    saved = rows(owner, "jobs")[0]
    assert saved["status"] == "FAILED"
    assert "request_attempted=true" in saved["detail"]
    assert "response_received=" not in saved["detail"]
    assert_private(output + json.dumps(saved))


@pytest.mark.parametrize("phase", ["with_options", "create_lookup"])
@pytest.mark.parametrize("error_type,status", [
    (ValueError, "INVALID"),
    (RuntimeError, "FAILED"),
])
def test_builtin_adapter_setup_failure_has_zero_attempts_and_safe_persisted_detail(
        mailbox, capsys, phase, error_type, status):
    owner, roster = mailbox
    prepared(owner)
    error = error_type(PRIVATE)
    create_calls = 0

    class Completions:
        @property
        def create(self):
            if phase == "create_lookup":
                raise error
            def invoke(**_kwargs):
                nonlocal create_calls
                create_calls += 1
            return invoke

    class Client:
        def with_options(self, **_kwargs):
            if phase == "with_options":
                raise error
            return SimpleNamespace(chat=SimpleNamespace(completions=Completions()))

        def close(self):
            pass

    with patch("openai.OpenAI", return_value=Client()):
        assert review.ReviewService(roster, timeout=1.25).once() == 1
    output = capsys.readouterr().err
    assert create_calls == 0
    assert f"error={error_type.__name__}" in output and f"outcome={status}" in output
    assert "request_attempted=false" in output and "response_received=false" in output
    owner.pump()
    saved = rows(owner, "jobs")[0]
    assert saved["status"] == status
    assert "request_attempted=false" in saved["detail"]
    assert "response_received=false" in saved["detail"]
    assert_private(output + json.dumps(saved))


@pytest.mark.parametrize("status,error_name", [
    (401, "AuthenticationError"),
    (429, "RateLimitError"),
    (500, "InternalServerError"),
])
def test_builtin_adapter_real_sdk_status_response_is_known_and_not_retried(
        mailbox, capsys, status, error_name):
    owner, roster = mailbox
    prepared(owner)
    requests = 0

    def respond(request):
        nonlocal requests
        requests += 1
        return httpx.Response(
            status, request=request,
            json={"error": {"message": PRIVATE, "type": "poison", "code": "poison"}},
        )

    http_client = httpx.Client(transport=httpx.MockTransport(respond))
    client = OpenAI(api_key="test-key", base_url=roster.base_url,
                    http_client=http_client, max_retries=0)
    try:
        with patch("openai.OpenAI", return_value=client):
            assert review.ReviewService(roster, timeout=1.25).once() == 1
    finally:
        client.close()
    output = capsys.readouterr().err
    assert requests == 1
    assert f"error={error_name}" in output and "outcome=FAILED" in output
    assert "request_attempted=true" in output and "response_received=true" in output
    owner.pump()
    saved = rows(owner, "jobs")[0]
    assert saved["status"] == "FAILED"
    assert f"error={error_name}" in saved["detail"]
    assert "request_attempted=true" in saved["detail"]
    assert "response_received=true" in saved["detail"]
    assert_private(output + json.dumps(saved))


def test_builtin_adapter_create_value_error_has_unknown_response_and_safe_detail(mailbox, capsys):
    owner, roster = mailbox
    prepared(owner)
    client = MagicMock()
    client.with_options.return_value = client
    client.chat.completions.create.side_effect = ValueError(PRIVATE)
    with patch("openai.OpenAI", return_value=client):
        assert review.ReviewService(roster, timeout=1.25).once() == 1
    output = capsys.readouterr().err
    assert client.chat.completions.create.call_count == 1
    assert "error=ValueError" in output and "outcome=INVALID" in output
    assert "request_attempted=true" in output and "response_received=" not in output
    owner.pump()
    saved = rows(owner, "jobs")[0]
    assert saved["status"] == "INVALID"
    assert "request_attempted=true" in saved["detail"]
    assert "response_received=" not in saved["detail"]
    assert_private(output + json.dumps(saved))


def test_custom_adapter_cannot_forge_host_reason_metadata(mailbox, capsys):
    owner, roster = mailbox
    prepared(owner)
    client = MagicMock()
    client.with_options.return_value = client
    client.chat.completions.create.return_value = fake_sdk_response('{"action":')
    with pytest.raises(ValueError) as caught:
        AutoSkillExtractor.generate_proposal(
            client, "current-model", '{"catalog":{},"tasks":[]}', timeout=1)
    forged = caught.value
    forged.reason = PRIVATE
    forged.metadata = {
        "finish_reason": PRIVATE, "request_attempted": PRIVATE,
        "response_received": True, "observed_bytes": PRIVATE,
    }
    service = review.ReviewService(
        roster, provider=lambda _: (_ for _ in ()).throw(forged))
    assert service.once() == 1
    output = capsys.readouterr().err
    assert "stage=provider.inference_validation" in output and "error=ValueError" in output
    assert "reason=" not in output and "request_attempted=" not in output
    assert "response_received=" not in output and "finish_reason=" not in output
    assert_private(output + json.dumps(rows(owner, "jobs")))


def test_review_metadata_is_revalidated_at_serialization():
    error = ReviewDiagnosticError("malformed_json", response_received=True)
    error.reason = PRIVATE
    error.metadata = {
        "request_attempted": PRIVATE,
        "response_received": True,
        "finish_reason": PRIVATE,
        "output_tokens": -1,
        "observed_bytes": 10 ** 20,
        PRIVATE: PRIVATE,
    }
    detail = review.error_detail(
        "provider.inference_validation", error, use_context=False,
        use_review_context=True)
    assert "reason=<invalid>" in detail
    assert "response_received=true" in detail
    assert "finish_reason=<invalid>" in detail
    assert "request_attempted=" not in detail and "output_tokens=" not in detail
    assert "observed_bytes=" not in detail
    assert_private(detail)


def test_service_validation_reason_is_specific_for_custom_adapter_result(mailbox, capsys):
    owner, roster = mailbox
    prepared(owner)
    assert review.ReviewService(roster, provider=lambda _: {"action": PRIVATE}).once() == 1
    output = capsys.readouterr().err
    assert "stage=provider.validation" in output
    assert "reason=invalid_action_or_fields" in output and "outcome=INVALID" in output
    assert "request_attempted=" not in output and "response_received=" not in output
    owner.pump()
    assert rows(owner, "jobs")[0]["status"] == "INVALID"
    assert_private(output + json.dumps(rows(owner, "jobs")))


def test_distinct_stages_are_deduped_and_persistence_history_is_not_false_recovery(mailbox, capsys):
    owner, roster = mailbox
    job = prepared(owner)
    service = review.ReviewService(roster, provider=lambda _: {"action": "NONE"})
    original = service._claim
    attempts = 0
    def claim(entry, job):
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            raise TimeoutError(PRIVATE)
        return original(entry, job)
    with patch.object(service, "_claim", side_effect=claim), patch.object(service, "_finish", side_effect=TimeoutError(PRIVATE)):
        run_scans(service, 10)
    output = capsys.readouterr().err
    assert output.count("stage=claim") == output.count("stage=result.persistence") == 1
    history = service.errors[owner.entry.profile_id]
    assert "historical" in history and "stage=result.persistence" in history
    assert "recovered" not in history and "current" not in history
    assert rows(owner, "jobs")[0]["status"] == "RUNNING"
    assert job["job_id"] in history


def test_service_diagnostics_sanitize_ids_and_bound_retained_state(mailbox, capsys):
    owner, roster = mailbox
    job = prepared(owner)
    service = review.ReviewService(roster, provider=lambda _: {})
    candidates = []
    for index in range(180):
        entry = replace(owner.entry, profile_id=f"profile-{index}")
        candidates.append((index, entry.profile_id, entry, job | {"job_id": f"{index:032x}"}))
    entry = replace(owner.entry, profile_id=PRIVATE)
    candidates.append((181, "malicious", entry, job | {"job_id": PRIVATE}))
    with patch.object(service, "_pending", return_value=candidates), patch.object(service, "_claim", side_effect=TimeoutError(PRIVATE)):
        service.once()
    output = capsys.readouterr().err
    assert "profile=<invalid>" in output and "job=<invalid>" in output
    assert_private(output + json.dumps(service.errors) + repr(service._reported))
    assert len(service.errors) <= 128 and len(service._reported) <= 128


@pytest.mark.parametrize("command", ["once", "status"])
def test_top_level_errors_are_safe_actionable_and_keep_exit_code(tmp_path, capsys, command):
    roster = setup_roster(tmp_path)
    with patch.object(review, "load_roster", side_effect=ValueError(PRIVATE)):
        assert review.main([command, "--roster", str(tmp_path / "roster.json")]) == 2
    output = capsys.readouterr().out
    assert f"stage={'service.startup' if command == 'once' else 'service.status'}" in output
    assert "error=ValueError" in output and "check" in output.lower()
    assert_private(output)


def test_once_final_json_includes_historical_provider_error(mailbox, tmp_path, capsys):
    owner, roster = mailbox
    prepared(owner)
    service = review.ReviewService(roster, provider=lambda _: (_ for _ in ()).throw(RuntimeError(PRIVATE)))
    with patch.object(review, "ReviewService", return_value=service):
        assert review.main(["once", "--roster", str(tmp_path / "roster.json")]) == 0
    output = capsys.readouterr()
    summary = json.loads(output.out)
    assert summary["processed"] == 1
    assert "historical" in summary["errors"]["user-0"]
    assert "stage=provider.inference" in output.err
    assert_private(output.out + output.err)


@pytest.mark.parametrize("point,stage", [("prepare_review", "owner.preparation"), ("apply_review", "owner.publication"), ("_ack", "owner.acknowledgement")])
def test_owner_background_error_has_stage_and_host_context_without_chat(mailbox, capsys, point, stage):
    owner, roster = mailbox
    if point == "prepare_review":
        owner.enqueue(evidence())
    else:
        prepared(owner)
        review.ReviewService(roster, provider=lambda _: {"action": "NONE"}).once()
    target = owner if point == "_ack" else owner.store
    with patch.object(target, point, side_effect=OSError(PRIVATE)):
        owner.pump()
    assert f"stage={stage}" in owner.last_error and "profile=user-0" in owner.last_error
    assert "historical" in owner.last_error and "error=OSError" in owner.last_error
    if point != "prepare_review":
        assert rows(owner, "jobs")[0]["job_id"] in owner.last_error
    assert_private(owner.last_error)
    assert not capsys.readouterr().out
    before = owner.last_error
    owner.set_enabled(False)
    snapshot = profile_bytes(owner)
    with patch.object(owner, "_pump", side_effect=OSError(PRIVATE)):
        owner.pump()
    assert owner.last_error == before
    assert snapshot == profile_bytes(owner)


def test_revoked_provider_failure_does_not_write_diagnostics_to_profile(mailbox, capsys):
    owner, roster = mailbox
    prepared(owner)
    snapshot = []
    def provider(_):
        owner.set_enabled(False)
        snapshot.append(profile_bytes(owner))
        raise RuntimeError(PRIVATE)
    assert review.ReviewService(roster, provider=provider).once() == 1
    owner.pump()
    assert snapshot[0] == profile_bytes(owner)
    assert not capsys.readouterr().err


@pytest.mark.parametrize("point,stage,state", [("_claim", "claim", "PREPARED"), ("_infer", "worker.authorization", "RUNNING"), ("_finish", "result.persistence", "RUNNING")])
def test_real_service_stage_lock_reports_its_configured_timeout(mailbox, capsys, point, stage, state):
    owner, roster = mailbox
    prepared(owner)
    service = review.ReviewService(roster, provider=lambda _: {"action": "NONE"}, timeout=1.25)
    original = getattr(service, point)
    def blocked(*args):
        with patch.object(review, "gate", side_effect=TimeoutError(PRIVATE)):
            return original(*args)
    with patch.object(service, point, side_effect=blocked):
        service.once()
    output = capsys.readouterr().err
    assert f"stage={stage}" in output and "coordination_timeout_s=5" in output
    assert "provider_timeout" not in output
    assert rows(owner, "jobs")[0]["status"] == state
    assert_private(output)


def test_result_sqlite_fault_is_visible_without_being_a_provider_timeout(mailbox, capsys):
    owner, roster = mailbox
    prepared(owner)
    service = review.ReviewService(roster, provider=lambda _: {"action": "NONE"}, timeout=1.25)
    original = service._finish
    def blocked(*args):
        with patch.object(review, "connect", side_effect=sqlite3.OperationalError(PRIVATE)):
            return original(*args)
    with patch.object(service, "_finish", side_effect=blocked):
        service.once()
    output = capsys.readouterr().err
    assert "stage=result.persistence" in output and "sqlite_busy_timeout_s=0.5" in output
    assert "provider_timeout" not in output and "restart the service" in output
    assert rows(owner, "jobs")[0]["status"] == "RUNNING"
    assert_private(output)


def test_discovery_and_status_diagnostics_bound_and_sanitize_bad_child_names(tmp_path, capsys):
    root = tmp_path / "profiles"
    root.mkdir()
    @contextmanager
    def children(_):
        yield [SimpleNamespace(name=f"bad.name-{i}") for i in range(180)] + [SimpleNamespace(name=PRIVATE)]
    with patch.object(review.os, "scandir", children):
        assert review.main(["status", "--profiles-dir", str(root), "--model", "test-model", "--base-url", "http://127.0.0.1:1/v1"]) == 0
    output = capsys.readouterr().out
    parsed = json.loads(output)
    assert not parsed["profiles"] and len(parsed["discovery_errors"]) <= 128
    assert "stage=discovery.child" in output and "<invalid>" in output
    assert_private(output)


def test_top_level_busy_singleton_names_stage_and_known_zero_wait(mailbox, tmp_path, capsys):
    with review.ProcessLock("hermes-skill-review-service", timeout=0):
        assert review.main(["once", "--roster", str(tmp_path / "roster.json")]) == 2
    output = capsys.readouterr().out
    assert "stage=service.startup.singleton" in output and "coordination_timeout_s=0" in output
    assert "busy" in output and "prior service" in output
    assert_private(output)


def test_top_level_provider_initialization_failure_is_safe(mailbox, tmp_path, capsys):
    with patch("openai.OpenAI", side_effect=RuntimeError(PRIVATE)):
        assert review.main(["once", "--roster", str(tmp_path / "roster.json")]) == 2
    output = capsys.readouterr().out
    assert "stage=service.startup.provider" in output and "error=RuntimeError" in output
    assert_private(output)


def test_status_mailbox_failure_has_profile_and_sqlite_context(mailbox, tmp_path, capsys):
    with patch.object(review, "connect", side_effect=sqlite3.OperationalError(PRIVATE)):
        assert review.main(["status", "--roster", str(tmp_path / "roster.json")]) == 2
    output = capsys.readouterr().out
    assert "stage=service.status" in output and "profile=user-0" in output
    assert "sqlite_busy_timeout_s=0.5" in output
    assert_private(output)


def test_owner_disconnect_after_revocation_does_not_update_diagnostic(mailbox):
    owner, _ = mailbox
    owner.set_enabled(False)
    before = owner.last_error
    with patch.object(review, "read_auth", side_effect=ValueError(PRIVATE)):
        owner.close()
    assert owner.last_error == before


def test_untrusted_exception_cannot_spoof_internal_diagnostic_fields(mailbox, capsys):
    owner, roster = mailbox
    prepared(owner)
    forged = review.DiagnosticFailure(PRIVATE, RuntimeError(PRIVATE))
    forged.error = PRIVATE
    service = review.ReviewService(roster, provider=lambda _: (_ for _ in ()).throw(forged))
    assert service.once() == 1
    output = capsys.readouterr().err
    assert "stage=provider.inference" in output
    assert_private(output + json.dumps(rows(owner, "jobs")))


def test_owner_returned_publication_failure_does_not_persist_untrusted_detail(mailbox):
    from skill_catalog import Publication
    owner, roster = mailbox
    prepared(owner)
    review.ReviewService(roster, provider=lambda _: {"action": "NONE"}).once()
    with patch.object(owner.store, "apply_review", return_value=Publication("INVALID", PRIVATE)):
        owner.pump()
    job = rows(owner, "jobs")[0]
    assert job["status"] == "INVALID" and "stage=owner.publication" in job["detail"]
    assert_private(json.dumps(job))


@pytest.mark.parametrize("consumer,stage", [("service", "pending.scan"), ("owner", "owner.delivery")])
def test_delivery_and_pending_validation_errors_include_available_job_id(mailbox, capsys, consumer, stage):
    owner, roster = mailbox
    job = prepared(owner)
    if consumer == "owner":
        review.ReviewService(roster, provider=lambda _: {"action": "NONE"}).once()
    with patch.object(review, "valid_job", side_effect=ValueError(PRIVATE)):
        if consumer == "owner":
            owner.pump()
            output = owner.last_error
        else:
            review.ReviewService(roster, provider=lambda _: {}).once()
            output = capsys.readouterr().err
    assert f"stage={stage}" in output and f"job={job['job_id']}" in output
    assert_private(output)


def test_incomplete_chat_includes_available_message_count(mailbox, capsys):
    owner, _ = mailbox
    item = review.Evidence("s")
    item.add({"role": "user", "content": "unfinished"})
    chat(owner, item)
    output = capsys.readouterr().out
    assert "observed_messages=1" in output and "minimum_messages=2" in output
    assert_refusal(output)
