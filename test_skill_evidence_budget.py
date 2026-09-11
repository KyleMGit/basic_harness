"""SQL projection and coordinated review-budget boundaries."""
from contextlib import nullcontext
import json
import hashlib
from pathlib import Path
import sqlite3
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from openai import OpenAI
import httpx
import pytest

from protocol import ToolProtocol
from skill_review import (BATCH_BYTES, CARRY_BYTES, ELIGIBLE_AGE_SECONDS, EPISODE_BYTES, EPISODE_MESSAGES,
                          IDLE_SECONDS, MAILBOX_BYTES, MAX_QUEUE_BYTES, QUEUE_HEADROOM_BYTES,
                          RAW_MESSAGE_BYTES, Evidence, EvidenceResult, ReviewService,
                          MailboxCapacityError, Owner, connect, packed)
from skills import AutoSkillExtractor
from test_async_skill_review import create_proposal, owner_for, rows, setup_roster
from test_skill_review_process import fake_openai


def native_sql_evidence(result_content, *, sql="SELECT customer_id, amount FROM sales"):
    item = Evidence("session", "task")
    item.add({"role": "user", "content": "Inspect the warehouse procedure"})
    item.add({
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "call-1",
            "type": "function",
            "function": {"name": "query_teradata", "arguments": json.dumps({"sql": sql})},
        }],
    })
    item.add({"role": "tool", "tool_call_id": "call-1", "content": result_content})
    item.add({"role": "assistant", "content": "The procedure executed; figures are not asserted."})
    return item.finish()


def test_native_business_rows_are_omitted_before_capture_even_when_large():
    private = "PRIVATE-CUSTOMER-ROW-" * 8000
    result = json.dumps({
        "database": "warehouse",
        "columns": ["customer", "amount"],
        "rows": [[private, 123.45]],
        "row_count": 1,
        "truncated": False,
    })
    evidence = native_sql_evidence(result)
    assert evidence.status == "READY"
    assert private not in evidence.messages_json
    projected = json.loads(evidence.messages_json)[2]
    receipt = json.loads(projected["content"])
    assert receipt == {
        "business_result_omitted": True,
        "call_id": "call-1",
        "columns": ["customer", "amount"],
        "database": "warehouse",
        "row_count": 1,
        "truncated": False,
    }


def test_xml_business_rows_are_omitted_without_retaining_wrapped_copy():
    private = "XML-PRIVATE-ROW"
    raw_result = json.dumps({
        "database": "warehouse", "columns": ["value"], "rows": [[private]],
        "row_count": 1, "truncated": False,
    })
    call = '<tool_call>{"name":"query_impala","arguments":{"sql":"SELECT value FROM facts"}}</tool_call>'
    wrapped = ToolProtocol.format_hermes_tool_response("query_impala", raw_result, "hermes_call_0")
    item = Evidence("session", "xml-task")
    item.add({"role": "user", "content": "Run the related Impala procedure"})
    item.add({"role": "assistant", "content": call})
    item.add({"role": "user", "content": wrapped})
    item.add({"role": "assistant", "content": "Completed with a bounded receipt."})
    evidence = item.finish()
    assert evidence.status == "READY"
    assert private not in evidence.messages_json
    assert evidence.messages_json.count("business_result_omitted") == 1
    assert "hermes_call_0" in evidence.messages_json


def test_metadata_is_retained_but_unknown_mixed_sql_output_and_copied_rows_are_omitted():
    metadata = json.dumps({
        "database": "warehouse", "columns": ["column_name", "data_type"],
        "rows": [["customer_id", "BIGINT"], ["amount", "DECIMAL"]],
        "row_count": 2, "truncated": False,
    })
    kept = native_sql_evidence(metadata, sql="SELECT column_name, data_type FROM information_schema.columns")
    assert "customer_id" in kept.messages_json and "business_result_omitted" not in kept.messages_json

    mixed = json.dumps({
        "database": "warehouse", "columns": ["column_name"], "rows": [["secret-mixed"]],
        "row_count": 1, "truncated": False, "provider_note": "model-labelled metadata",
    })
    omitted = native_sql_evidence(mixed, sql="SELECT column_name FROM information_schema.columns")
    assert "secret-mixed" not in omitted.messages_json
    assert "unclassified_sql_result_omitted" in omitted.messages_json

    raw = json.dumps({"database": "warehouse", "columns": ["value"], "rows": [["copied-secret"]],
                      "row_count": 1, "truncated": False})
    item = Evidence("session", "copied")
    item.add({"role": "user", "content": "Run and explain the procedure"})
    item.add({"role": "assistant", "content": "", "tool_calls": [{
        "id": "copy", "type": "function",
        "function": {"name": "query_impala", "arguments": json.dumps({"sql": "SELECT value FROM facts"})},
    }]})
    item.add({"role": "tool", "tool_call_id": "copy", "content": raw})
    item.add({"role": "assistant", "content": raw})
    result = item.finish()
    assert "copied-secret" not in result.messages_json
    assert "assistant_business_result_copy_omitted" in result.messages_json


def test_metadata_mention_in_literal_cannot_release_business_rows():
    raw = json.dumps({"database": "warehouse", "columns": ["customer"], "rows": [["PRIVATE-BUSINESS-SENTINEL"]], "row_count": 1, "truncated": False})
    evidence = native_sql_evidence(raw, sql="SELECT customer FROM sales WHERE note = 'information_schema'")
    assert "PRIVATE-BUSINESS-SENTINEL" not in evidence.messages_json


def test_mixed_catalog_business_query_cannot_release_business_rows():
    raw = json.dumps({"database": "warehouse", "columns": ["customer"], "rows": [["PRIVATE-BUSINESS-SENTINEL"]], "row_count": 1, "truncated": False})
    evidence = native_sql_evidence(raw, sql="SELECT s.customer FROM sales s JOIN information_schema.tables t ON s.type = t.table_name")
    assert "PRIVATE-BUSINESS-SENTINEL" not in evidence.messages_json


@pytest.mark.parametrize("sql", [
    "SELECT customer FROM sales -- information_schema.columns",
    "SELECT customer FROM sales /* JOIN information_schema.tables */",
    "WITH catalog AS (SELECT * FROM information_schema.columns) SELECT customer FROM sales",
    "SELECT c.column_name, s.customer FROM information_schema.columns c, sales s",
])
def test_metadata_comments_and_mixed_ctes_cannot_release_business_rows(sql):
    raw = json.dumps({"database": "warehouse", "columns": ["customer"],
                      "rows": [["PRIVATE-BUSINESS-SENTINEL"]], "row_count": 1, "truncated": False})
    assert "PRIVATE-BUSINESS-SENTINEL" not in native_sql_evidence(raw, sql=sql).messages_json


@pytest.mark.parametrize("sql", [
    "SELECT columnname FROM DBC.ColumnsV",
    "SELECT table_name FROM information_schema.tables",
    "DESCRIBE sales",
])
def test_supported_single_catalog_sources_retain_bounded_metadata(sql):
    raw = json.dumps({"database": "warehouse", "columns": ["name"],
                      "rows": [["SUPPORTED-METADATA"]], "row_count": 1, "truncated": False})
    evidence = native_sql_evidence(raw, sql=sql)
    assert "SUPPORTED-METADATA" in evidence.messages_json
    assert "business_result_omitted" not in evidence.messages_json


def test_256_kib_turn_and_256_message_limits_replace_old_caps():
    item = Evidence("session", "large-retained")
    item.add({"role": "user", "content": "x" * (64 * 1024)})
    for index in range(127):
        item.add({"role": "assistant", "content": f"checkpoint {index}"})
        item.add({"role": "user", "content": f"continue {index}"})
    item.add({"role": "assistant", "content": "done"})
    evidence = item.finish()
    assert evidence.status == "READY"
    assert len(json.loads(evidence.messages_json)) == 256


def test_decoded_json_obeys_depth_and_node_limits():
    nested = "leaf"
    for _ in range(25):
        nested = [nested]
    deep = Evidence("session", "deep")
    deep.add({"role": "user", "content": json.dumps(nested)})
    deep.add({"role": "assistant", "content": "done"})
    result = deep.finish()
    assert result.status == "OVERFLOW"
    assert result.reason == "depth" and result.limit == 24

    nodes = Evidence("session", "nodes")
    nodes.add({"role": "user", "content": json.dumps([0] * 8192)})
    nodes.add({"role": "assistant", "content": "done"})
    result = nodes.finish()
    assert result.status == "OVERFLOW"
    assert result.reason == "nodes" and result.limit == 8192


def test_raw_message_limit_distinguishes_exact_inspection_from_plus_one():
    shell = packed({"content": "", "role": "user"})
    exact = Evidence("session", "raw-exact")
    exact.add({"role": "user", "content": "x" * (RAW_MESSAGE_BYTES - len(shell.encode()))})
    result = exact.finish()
    assert result.reason == "serialized_bytes"
    assert result.observed == RAW_MESSAGE_BYTES + 2

    refused = Evidence("session", "raw-plus-one")
    refused.add({"role": "user", "content": "x" * (RAW_MESSAGE_BYTES - len(shell.encode()) + 1)})
    result = refused.finish()
    assert result.reason == "raw_bytes"
    assert result.observed == RAW_MESSAGE_BYTES + 1 and result.limit == RAW_MESSAGE_BYTES


@pytest.mark.parametrize("kind", ["assistant_json", "native_arguments", "xml_arguments"])
def test_raw_limit_precedes_json_decoding_for_non_omittable_evidence(kind):
    content = json.dumps({"text": "x" * (RAW_MESSAGE_BYTES + 1)})
    if kind == "assistant_json":
        message = {"role": "assistant", "content": content}
    elif kind == "native_arguments":
        message = {"role": "assistant", "content": "", "tool_calls": [{
            "id": "oversize", "type": "function",
            "function": {"name": "query_teradata", "arguments": content},
        }]}
    else:
        message = {
            "role": "assistant",
            "content": "<tool_call>" + json.dumps({
                "name": "query_teradata",
                "arguments": {"sql": "SELECT " + "x" * (RAW_MESSAGE_BYTES + 1)},
            }) + "</tool_call>",
        }
    original = json.loads
    oversized_decodes = []

    def tracked(value, *args, **kwargs):
        if isinstance(value, str) and len(value.encode("utf-8")) > RAW_MESSAGE_BYTES:
            oversized_decodes.append(len(value.encode("utf-8")))
        return original(value, *args, **kwargs)

    item = Evidence("raw-before-decode", "raw-before-decode")
    item.add({"role": "user", "content": "Inspect only synthetic evidence"})
    with patch("skill_review.json.loads", side_effect=tracked):
        item.add(message)
    result = item.finish()
    assert result.status == "OVERFLOW" and result.reason == "raw_bytes"
    assert result.observed > RAW_MESSAGE_BYTES and result.limit == RAW_MESSAGE_BYTES
    assert oversized_decodes == []


def sized_evidence(session, task, size, message_count=2, *, required=False):
    messages = [{"role": "user", "content": ""}]
    for index in range(message_count - 2):
        messages.append({"role": "assistant" if index % 2 == 0 else "user", "content": str(index)})
    messages.append({"role": "assistant", "content": "done"})
    base = packed(messages)
    padding = size - len(base.encode())
    assert padding >= 0
    messages[0]["content"] = "x" * padding
    messages_json = packed(messages)
    assert len(messages_json.encode()) == size
    events = ([{"tool": "query_teradata", "backend": "query_teradata",
                "signature": "required-" + task, "resources": ["shared"],
                "nonroutine": False, "outcome": "failure"}] if required else [])
    return EvidenceResult("READY", session, task, messages_json=messages_json,
                          message_count=message_count,
                          event_json=packed({"version": 1, "correction": False, "events": events}))


def test_episode_byte_and_message_boundaries_are_enforced_at_owner_admission(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    try:
        sizes = [200 * 1024, 200 * 1024, EPISODE_BYTES - 400 * 1024]
        for index, size in enumerate(sizes):
            assert owner.capture_turn(sized_evidence("bytes", f"exact-{index}", size, required=True)).status == "SKIPPED"
        stored = rows(owner, "episode_sources")
        assert sum(source["bytes"] for source in stored) == EPISODE_BYTES
    finally:
        owner.close()

    over_root = tmp_path / "over"
    over_root.mkdir()
    roster = setup_roster(over_root)
    owner = owner_for(roster, background=False)
    try:
        for index, size in enumerate([200 * 1024, 200 * 1024]):
            assert owner.capture_turn(sized_evidence("bytes", f"prefix-{index}", size, required=True)).status == "SKIPPED"
        result = owner.capture_turn(sized_evidence(
            "bytes", "plus-one", EPISODE_BYTES - 400 * 1024 + 1, required=True))
        assert result.status == "OVERFLOW"
        assert f"observed_bytes={EPISODE_BYTES + 1}" in result.detail
        assert f"limit_bytes={EPISODE_BYTES}" in result.detail
    finally:
        owner.close()

    count_root = tmp_path / "messages"
    count_root.mkdir()
    roster = setup_roster(count_root)
    owner = owner_for(roster, background=False)
    try:
        assert owner.capture_turn(sized_evidence("messages", "first", 32 * 1024, 255, required=True)).status == "SKIPPED"
        assert owner.capture_turn(sized_evidence("messages", "second", 32 * 1024, 256, required=True)).status == "SKIPPED"
        result = owner.capture_turn(sized_evidence("messages", "plus-two", 1024, 2, required=True))
        assert result.status == "OVERFLOW"
        assert f"observed_messages={EPISODE_MESSAGES + 1}" in result.detail
        assert f"limit_messages={EPISODE_MESSAGES}" in result.detail
    finally:
        owner.close()


def eligible_evidence(task, *, padding="", correction=False):
    item = Evidence("session-" + task, task)
    prefix = "No, that procedure is wrong; use the verified workaround. " if correction else ""
    item.add({"role": "user", "content": prefix + padding + " reusable totals"})
    item.add({"role": "assistant", "content": "", "tool_calls": [{
        "id": "bad", "type": "function",
        "function": {"name": "query_teradata", "arguments": json.dumps({"sql": "SELECT bad_fn(amount) FROM sales"})},
    }]})
    item.add({"role": "tool", "tool_call_id": "bad", "content": "Teradata query failed: syntax error"})
    item.add({"role": "assistant", "content": "", "tool_calls": [{
        "id": "good", "type": "function",
        "function": {"name": "query_teradata", "arguments": json.dumps({"sql": "SELECT SUM(amount) OVER () FROM sales"})},
    }]})
    item.add({"role": "tool", "tool_call_id": "good", "content": json.dumps({
        "database": "warehouse", "columns": ["amount"], "rows": [["private"]],
        "row_count": 1, "truncated": False,
    })})
    item.add({"role": "assistant", "content": "Verified execution completed."})
    return item.finish()


def test_zero_request_budget_refusal_does_not_suppress_new_fitting_episode(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    requests = []
    service = ReviewService(
        roster, provider=lambda request: requests.append(request) or {"action": "NONE"},
        selected_view_bytes=BATCH_BYTES,
    )
    try:
        oversized = eligible_evidence("refused-before-review", padding="x" * (200 * 1024))
        assert owner.capture_turn(oversized).status == "ELIGIBLE"
        owner.flush_session(oversized.session_id)
        owner.pump()
        assert service.once() == 1
        owner.pump()
        assert requests == []
        assert rows(owner, "jobs")[-1]["status"] == "BUDGET_REFUSED"
        assert rows(owner, "review_fingerprints") == []

        fitting = eligible_evidence("new-fitting-complete-work")
        assert owner.capture_turn(fitting).status == "ELIGIBLE"
        owner.flush_session(fitting.session_id)
        owner.pump()
        assert service.once() == 1
        owner.pump()
        assert len(requests) == 1
    finally:
        owner.close()


@pytest.mark.parametrize("publication_outcome", ["FAILED", "INVALID", "APPLIED", "NONE"])
def test_review_fingerprints_follow_final_publication_outcome(tmp_path, publication_outcome):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    requests = []
    proposal = create_proposal("publication_outcome")
    if publication_outcome == "INVALID":
        proposal["action"] = "UPDATE"
        proposal["target_id"] = "unknown-host-target"
        del proposal["name"]
    elif publication_outcome == "NONE":
        proposal = {"action": "NONE"}
    service = ReviewService(
        roster, provider=lambda request: requests.append(request) or proposal)
    try:
        initial = eligible_evidence("publication-first-work")
        assert owner.capture_turn(initial).status == "ELIGIBLE"
        owner.flush_session(initial.session_id)
        owner.pump()
        assert service.once() == 1
        failure = patch.object(
            owner.store, "_publish", side_effect=PermissionError("synthetic publication failure"))
        with failure if publication_outcome == "FAILED" else nullcontext():
            owner.pump()
        assert len(requests) == 1
        assert rows(owner, "jobs")[-1]["status"] == publication_outcome
        assert rows(owner, "episode_sources") == []

        fingerprints = rows(owner, "review_fingerprints")
        if publication_outcome in ("FAILED", "INVALID"):
            assert fingerprints == []
        else:
            assert fingerprints

        newer = eligible_evidence("publication-new-complete-work")
        admission = owner.capture_turn(newer)
        if publication_outcome in ("FAILED", "INVALID"):
            assert admission.status == "ELIGIBLE"
            owner.flush_session(newer.session_id)
            owner.pump()
            assert service.once() == 1
            owner.pump()
            assert len(requests) == 2
        else:
            assert admission.status == "SKIPPED"
            owner.flush_session(newer.session_id)
            owner.pump()
            assert service.once() == 0
            assert len(requests) == 1
    finally:
        owner.close()


def correction_evidence(session, task="correction"):
    item = Evidence(session, task)
    item.add({"role": "user", "content": "No, that procedure is wrong; use the verified QUALIFY correction."})
    item.add({"role": "assistant", "content": "", "tool_calls": [{
        "id": "corrected", "type": "function",
        "function": {"name": "query_teradata", "arguments": json.dumps({
            "sql": "SELECT * FROM sales QUALIFY ROW_NUMBER() OVER (ORDER BY sale_date DESC)=1"})},
    }]})
    item.add({"role": "tool", "tool_call_id": "corrected", "content": json.dumps({
        "database": "warehouse", "columns": ["amount"], "rows": [["private"]],
        "row_count": 1, "truncated": False,
    })})
    item.add({"role": "assistant", "content": "Verified correction completed."})
    return item.finish()


def test_large_complete_first_context_uses_selected_view_not_carry_limit(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    calls = []
    session = "large-first-context"
    try:
        first = sized_evidence(session, "large-original", CARRY_BYTES + 4096)
        learned = eligible_evidence("resolution")
        learned = EvidenceResult(learned.status, session, learned.task_id, learned.messages_json,
                                 learned.reason, learned.observed, learned.limit, learned.partial,
                                 learned.message_count, learned.event_json)
        assert owner.capture_turn(first).status == "SKIPPED"
        assert owner.capture_turn(learned).status == "ELIGIBLE"
        owner.flush_session(session)
        owner.pump()
        service = ReviewService(roster, provider=lambda request: calls.append(request) or {"action": "NONE"})
        assert service.once() == 1
        owner.pump()
        assert len(calls) == 1
        assert "x" * 1024 in calls[0]
        assert rows(owner, "jobs")[-1]["status"] == "NONE"
        anchor = json.loads(rows(owner, "episodes")[-1]["anchor_json"])
        assert anchor["missing_context"] is True
        assert anchor["reason"] == "carry_anchor_bytes"
        assert len(packed(anchor).encode()) <= CARRY_BYTES
    finally:
        owner.close()


def test_missing_oversized_anchor_truthfully_refuses_later_dependent_correction(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    calls = []
    session = "missing-anchor"
    try:
        first = sized_evidence(session, "oversized-original", CARRY_BYTES + 2048)
        learned = eligible_evidence("initial-resolution")
        learned = EvidenceResult(learned.status, session, learned.task_id, learned.messages_json,
                                 learned.reason, learned.observed, learned.limit, learned.partial,
                                 learned.message_count, learned.event_json)
        owner.capture_turn(first)
        owner.capture_turn(learned)
        owner.flush_session(session)
        owner.pump()
        service = ReviewService(roster, provider=lambda request: calls.append(request) or {"action": "NONE"})
        assert service.once() == 1
        owner.pump()
        assert len(calls) == 1

        assert owner.capture_turn(correction_evidence(session)).status == "ELIGIBLE"
        owner.pump()
        assert service.once() == 1
        owner.pump()
        assert len(calls) == 1
        assert rows(owner, "jobs")[-1]["status"] == "BUDGET_REFUSED"
        assert "missing_context" in rows(owner, "jobs")[-1]["detail"]
    finally:
        owner.close()


def test_long_routine_draft_retires_optional_whole_turns_then_correction_progresses(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    requests = []
    session = "long-routine-session"
    try:
        for index in range(140):
            result = owner.capture_turn(sized_evidence(session, f"routine-{index}", 4096, 4))
            assert result.status == "SKIPPED"
        sources = rows(owner, "episode_sources")
        assert sources[0]["task_id"] == "routine-0"
        assert len(sources) < 140
        assert not rows(owner, "jobs")

        assert owner.capture_turn(correction_evidence(session)).status == "ELIGIBLE"
        owner.pump()
        service = ReviewService(roster, provider=lambda request: requests.append(request) or {"action": "NONE"})
        assert service.once() == 1
        owner.pump()
        assert len(requests) == 1 and "QUALIFY" in requests[0]
    finally:
        owner.close()


def test_explicit_session_retirement_bounds_inactive_state_but_preserves_pinned_work(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    try:
        pending = eligible_evidence("pinned")
        owner.capture_turn(pending)
        owner.flush_session(pending.session_id)
        owner.pump()
        before = rows(owner, "episode_sources")
        owner.flush_session(pending.session_id, retire=True)
        assert rows(owner, "episode_sources") == before
        assert rows(owner, "jobs")[-1]["status"] == "PREPARED"
    finally:
        owner.close()

    retired_root = tmp_path / "retired"
    retired_root.mkdir()
    roster = setup_roster(retired_root)
    owner = owner_for(roster, background=False)
    try:
        for index in range(6):
            item = eligible_evidence(f"retired-{index}", padding="x" * (CARRY_BYTES + 1024),
                                     correction=True)
            owner.capture_turn(item)
            owner.flush_session(item.session_id)
            owner.pump()
            assert ReviewService(roster, provider=lambda _: {"action": "NONE"}).once() == 1
            owner.pump()
            owner.flush_session(item.session_id, retire=True)
        assert rows(owner, "episodes") == []
        assert rows(owner, "episode_sources") == []
        with connect(owner.entry, readonly=True) as conn:
            assert Owner._payload_usage(conn) < 16 * 1024
    finally:
        owner.close()


def test_review_fingerprints_suppress_cross_session_duplicate_but_allow_new_shape(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    calls = []
    service = ReviewService(roster, provider=lambda request: calls.append(request) or {"action": "NONE"})
    try:
        first = eligible_evidence("fingerprint-first")
        owner.capture_turn(first)
        owner.flush_session(first.session_id)
        owner.pump()
        assert service.once() == 1
        owner.pump()
        assert rows(owner, "review_fingerprints")

        duplicate = eligible_evidence("fingerprint-duplicate")
        assert owner.capture_turn(duplicate).status == "SKIPPED"
        owner.flush_session(duplicate.session_id)
        owner.pump()
        assert service.once() == 0
        assert len(calls) == 1

        changed = eligible_evidence("fingerprint-new-shape")
        changed_events = json.loads(changed.event_json)
        changed_events["events"][-1]["signature"] = "new-functional-signature"
        changed = EvidenceResult(changed.status, changed.session_id, changed.task_id,
                                 changed.messages_json, changed.reason, changed.observed,
                                 changed.limit, changed.partial, changed.message_count,
                                 packed(changed_events))
        assert owner.capture_turn(changed).status == "ELIGIBLE"
        owner.flush_session(changed.session_id)
        owner.pump()
        assert service.once() == 1
        owner.pump()
        assert len(calls) == 2
        fingerprints = rows(owner, "review_fingerprints")
        assert len(fingerprints) <= 1024
        assert sum(row["bytes"] for row in fingerprints) <= 128 * 1024
    finally:
        owner.close()


def test_episode_queue_record_and_byte_refusals_preserve_accepted_work(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False, max_records=2)
    try:
        assert owner.capture_turn(sized_evidence("one", "one", 1024)).status == "SKIPPED"
        assert owner.capture_turn(sized_evidence("two", "two", 1024)).status == "SKIPPED"
        refusal = owner.capture_turn(sized_evidence("three", "three", 1024))
        assert refusal.status == "OVERFLOW" and "queue_records" in refusal.detail
        assert [row["task_id"] for row in rows(owner, "episode_sources")] == ["one", "two"]
    finally:
        owner.close()

    byte_root = tmp_path / "queue-bytes"
    byte_root.mkdir()
    roster = setup_roster(byte_root)
    owner = owner_for(roster, background=False)
    try:
        with patch("skill_review.MAX_QUEUE_BYTES", 96 * 1024), \
             patch("skill_review.QUEUE_HEADROOM_BYTES", 8 * 1024):
            assert owner.capture_turn(sized_evidence("one", "one", 60 * 1024)).status == "SKIPPED"
            refusal = owner.capture_turn(sized_evidence("two", "two", 40 * 1024))
        assert refusal.status == "OVERFLOW" and "queue_bytes" in refusal.detail
        assert [row["task_id"] for row in rows(owner, "episode_sources")] == ["one"]
    finally:
        owner.close()


def test_oversized_existing_mailbox_is_refused_without_modification(tmp_path):
    roster = setup_roster(tmp_path)
    entry = roster.profiles[0]
    Path(entry.mailbox).parent.mkdir(parents=True)
    with sqlite3.connect(entry.mailbox) as conn:
        conn.execute("CREATE TABLE oversized(payload BLOB)")
        conn.execute("INSERT INTO oversized VALUES(zeroblob(?))", (MAILBOX_BYTES + 4096,))
    before = hashlib.sha256(Path(entry.mailbox).read_bytes()).hexdigest()
    with pytest.raises(MailboxCapacityError):
        owner_for(roster, background=False)
    assert hashlib.sha256(Path(entry.mailbox).read_bytes()).hexdigest() == before


def test_legacy_prepared_job_is_consumed_after_owner_reconnect(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    assert owner.enqueue(eligible_evidence("legacy-prepared")).status == "ACCEPTED"
    owner.pump()
    assert rows(owner, "jobs")[-1]["status"] == "PREPARED"
    owner.close()

    owner = owner_for(roster, background=False)
    try:
        assert ReviewService(roster, provider=lambda _: {"action": "NONE"}).once() == 1
        owner.pump()
        assert rows(owner, "jobs")[-1]["status"] == "NONE"
        assert rows(owner, "evidence") == []
    finally:
        owner.close()


def test_unfit_oldest_episode_is_terminally_refused_without_provider_and_later_work_progresses(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    calls = []
    try:
        first = eligible_evidence("oversized", padding="x" * BATCH_BYTES)
        second = eligible_evidence("fitting")
        assert owner.capture_turn(first).status == "ELIGIBLE"
        owner.flush_session(first.session_id)
        owner.pump()
        assert owner.capture_turn(second).status == "ELIGIBLE"
        owner.flush_session(second.session_id)

        service = ReviewService(
            roster, provider=lambda request: calls.append(request) or {"action": "NONE"},
            selected_view_bytes=BATCH_BYTES,
        )
        assert service.once() == 1
        owner.pump()
        assert rows(owner, "jobs")[0]["status"] == "BUDGET_REFUSED"
        assert calls == []

        owner.pump()
        assert service.once() == 1
        owner.pump()
        assert len(calls) == 1
        assert [job["status"] for job in rows(owner, "jobs")] == ["BUDGET_REFUSED", "NONE"]
    finally:
        owner.close()


def test_idle_dispatch_occurs_at_120_seconds_not_before(tmp_path):
    roster = setup_roster(tmp_path)
    with patch("skill_review.time.time", return_value=1000.0):
        owner = owner_for(roster, background=False)
        try:
            assert owner.capture_turn(eligible_evidence("idle")).status == "ELIGIBLE"
        finally:
            pass
    try:
        with patch("skill_review.time.time", return_value=1000.0 + IDLE_SECONDS - .001):
            owner.pump()
            assert rows(owner, "jobs") == []
        with patch("skill_review.time.time", return_value=1000.0 + IDLE_SECONDS):
            owner.pump()
            assert len(rows(owner, "jobs")) == 1
    finally:
        owner.close()


def test_session_change_and_900_second_completed_turn_boundary_dispatch(tmp_path):
    roster = setup_roster(tmp_path)
    with patch("skill_review.time.time", return_value=2000.0):
        owner = owner_for(roster, background=False)
        first = eligible_evidence("session-boundary")
        assert owner.capture_turn(first).status == "ELIGIBLE"
    try:
        trivial = Evidence("different-session", "trivial")
        trivial.add({"role": "user", "content": "thanks"})
        trivial.add({"role": "assistant", "content": "acknowledged"})
        with patch("skill_review.time.time", return_value=2001.0):
            assert owner.capture_turn(trivial.finish()).status == "SKIPPED"
        owner.pump()
        assert len(rows(owner, "jobs")) == 1
    finally:
        owner.close()

    age_root = tmp_path / "age"
    age_root.mkdir()
    roster = setup_roster(age_root)
    with patch("skill_review.time.time", return_value=3000.0):
        owner = owner_for(roster, background=False)
        aged = eligible_evidence("aged")
        assert owner.capture_turn(aged).status == "ELIGIBLE"
    try:
        followup = Evidence(aged.session_id, "age-followup")
        followup.add({"role": "user", "content": "continue with the same formatting"})
        followup.add({"role": "assistant", "content": "done"})
        with patch("skill_review.time.time", return_value=3000.0 + ELIGIBLE_AGE_SECONDS - .001):
            owner.capture_turn(followup.finish())
            assert rows(owner, "jobs") == []
        followup2 = Evidence(aged.session_id, "age-followup-2")
        followup2.add({"role": "user", "content": "continue with the same limit"})
        followup2.add({"role": "assistant", "content": "done"})
        with patch("skill_review.time.time", return_value=3000.0 + ELIGIBLE_AGE_SECONDS):
            owner.capture_turn(followup2.finish())
            owner.pump()
        assert len(rows(owner, "jobs")) == 1
    finally:
        owner.close()


def test_complete_sdk_wire_over_256_kib_is_refused_before_http_request():
    with fake_openai() as (endpoint, _, _, requests):
        client = OpenAI(api_key="synthetic", base_url=endpoint, max_retries=0)
        prepared = json.dumps({"catalog": {}, "tasks": [{"text": "\\" * 65200}]}, separators=(",", ":"))
        assert len(prepared.encode()) < 128 * 1024
        try:
            with pytest.raises(ValueError, match="wire"):
                AutoSkillExtractor.generate_proposal(
                    client, "test-model", prepared, timeout=2,
                    wire_body_bytes=256 * 1024,
                )
            assert requests == []
        finally:
            client.close()


def test_mailbox_and_payload_headroom_limits_are_exact(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    try:
        import sqlite3
        with connect(owner.entry) as conn:
            page_size = conn.execute("PRAGMA page_size").fetchone()[0]
            assert conn.execute("PRAGMA max_page_count").fetchone()[0] * page_size == MAILBOX_BYTES
        assert MAX_QUEUE_BYTES == 8 * 1024 * 1024
        assert QUEUE_HEADROOM_BYTES == 512 * 1024
    finally:
        owner.close()


def test_actual_sdk_serialization_matches_wire_accounting_at_limit_plus_minus_one():
    target = 256 * 1024

    def prepared(slashes, suffix):
        return json.dumps({"catalog": {}, "tasks": [{"text": "\\" * slashes + "a" * suffix}]},
                          separators=(",", ":"))

    base = prepared(65000, 0)
    suffix = target - AutoSkillExtractor.sdk_wire_bytes("test-model", base)
    exact = prepared(65000, suffix)
    assert len(exact.encode()) <= 128 * 1024
    assert AutoSkillExtractor.sdk_wire_bytes("test-model", exact) == target
    observed = []

    def respond(request):
        observed.append(len(request.content))
        return httpx.Response(200, request=request, json={
            "id": "fake", "object": "chat.completion", "created": 1, "model": "test-model",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": '{"action":"NONE"}'}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    http = httpx.Client(transport=httpx.MockTransport(respond))
    client = OpenAI(api_key="synthetic", base_url="http://127.0.0.1:1/v1", http_client=http, max_retries=0)
    try:
        assert AutoSkillExtractor.generate_proposal(
            client, "test-model", exact, wire_body_bytes=target,
        ) == {"action": "NONE"}
        assert observed == [target]
        with pytest.raises(ValueError, match="wire"):
            AutoSkillExtractor.generate_proposal(
                client, "test-model", prepared(65000, suffix + 1),
                wire_body_bytes=target,
            )
        assert observed == [target]
    finally:
        client.close()


def test_serialized_turn_limit_is_exact_and_message_257_is_refused():
    limit = 1024
    empty = json.dumps([{"content": "", "role": "user"},
                        {"content": "done", "role": "assistant"}],
                       ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    padding = limit - len(empty.encode())
    passing = Evidence("session", "exact", max_bytes=limit)
    passing.add({"role": "user", "content": "x" * padding})
    passing.add({"role": "assistant", "content": "done"})
    assert len(passing.finish().messages_json.encode()) == limit

    refused = Evidence("session", "plus-one", max_bytes=limit)
    refused.add({"role": "user", "content": "x" * (padding + 1)})
    refused.add({"role": "assistant", "content": "done"})
    result = refused.finish()
    assert result.status == "OVERFLOW" and result.reason == "serialized_bytes"
    assert result.observed == limit + 1 and result.limit == limit

    count = Evidence("session", "message-count")
    count.add({"role": "user", "content": "start"})
    for index in range(255):
        count.add({"role": "assistant" if index % 2 == 0 else "user", "content": str(index)})
    count.add({"role": "assistant", "content": "overflow"})
    result = count.finish()
    assert result.status == "OVERFLOW" and result.reason == "message_count"
    assert result.observed == 257 and result.limit == 256


def test_prepared_json_limit_is_exact_at_128_kib():
    shell = '{"text":""}'
    exact = '{"text":"' + ("x" * (128 * 1024 - len(shell))) + '"}'
    assert len(exact.encode()) == 128 * 1024
    client = MagicMock()
    client.with_options.return_value = client
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content='{\"action\":\"NONE\"}'))])
    assert AutoSkillExtractor.generate_proposal(
        client, "test-model", exact, prepared_input_bytes=128 * 1024,
    ) == {"action": "NONE"}
    with pytest.raises(ValueError, match="Prepared"):
        AutoSkillExtractor.generate_proposal(
            client, "test-model", exact + " ", prepared_input_bytes=128 * 1024,
        )
    assert client.chat.completions.create.call_count == 1
