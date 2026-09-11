"""CSV export evidence capture at protocol, producer, and scheduling boundaries."""
import csv
import hashlib
import json
import re
import sqlite3
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import agent as agent_module
import db_tools
from protocol import ToolProtocol
from skill_review import Evidence, Owner, ReviewService
from test_async_skill_review import rows
from test_skill_review_integration import answer_message, make_agent


EXPORTS = (
    ("export_impala_csv", "Impala"),
    ("export_teradata_csv", "Teradata"),
)


def export_manifest(tool, sql, **changes):
    backend = dict(EXPORTS)[tool]
    manifest = {
        "backend": backend,
        "batch_size": 2,
        "byte_size": 97,
        "columns": ["customer_id", "amount"],
        "completed": True,
        "database": "warehouse",
        "file_path": "exports/complete.csv",
        "row_count": 4,
        "sql_sha256": hashlib.sha256(sql.encode("utf-8")).hexdigest(),
    }
    manifest.update(changes)
    return json.dumps(manifest, sort_keys=True, separators=(",", ":"))


def capture_export(tool, sql, result, protocol="native"):
    item = Evidence("export-session", f"{protocol}-{tool}")
    item.add({"role": "user", "content": "Export the complete reusable report"})
    if protocol == "native":
        item.add({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "export-call",
                "type": "function",
                "function": {"name": tool, "arguments": json.dumps({"sql": sql})},
            }],
        })
        item.add({"role": "tool", "tool_call_id": "export-call", "content": result})
    else:
        item.add({
            "role": "assistant",
            "content": "<tool_call>" + json.dumps({
                "name": tool,
                "arguments": {"sql": sql},
                "tool_call_id": "export-call",
            }) + "</tool_call>",
        })
        item.add({
            "role": "user",
            "content": ToolProtocol.format_hermes_tool_response(
                tool, result, tool_call_id="export-call"
            ),
        })
    item.add({"role": "assistant", "content": "The complete export finished."})
    return item.finish()


def event_data(evidence):
    return json.loads(evidence.event_json)["events"]


def projected_receipt(evidence, protocol):
    messages = json.loads(evidence.messages_json)
    if protocol == "native":
        return json.loads(messages[2]["content"])
    match = re.search(r"<tool_response>\s*(.*?)\s*</tool_response>", messages[2]["content"], re.S)
    assert match
    envelope = json.loads(match.group(1))
    return json.loads(envelope["content"])


@pytest.mark.parametrize("tool,backend", EXPORTS)
@pytest.mark.parametrize("protocol", ["native", "xml"])
def test_successful_nonroutine_export_is_captured_with_bounded_complete_receipt(
        tool, backend, protocol):
    sql = "SELECT c.id, SUM(s.amount) OVER (PARTITION BY c.id) FROM customers c JOIN sales s ON s.id=c.id"
    evidence = capture_export(tool, sql, export_manifest(tool, sql), protocol)

    assert evidence.status == "READY"
    events = event_data(evidence)
    assert len(events) == 1
    assert events[0]["backend"] == tool
    assert events[0]["call_id"] == "export-call"
    assert events[0]["export"] is True
    assert events[0]["metadata"] is False
    assert events[0]["nonroutine"] is True
    assert events[0]["outcome"] == "business_success"
    assert events[0]["tool"] == tool
    assert events[0]["resources"] and len(events[0]["signature"]) == 64
    assert not any(key.startswith("_") for key in events[0])
    receipt = projected_receipt(evidence, protocol)
    assert receipt == {
        "backend": backend,
        "batch_size": 2,
        "byte_size": 97,
        "call_id": "export-call",
        "columns": ["customer_id", "amount"],
        "completed": True,
        "database": "warehouse",
        "exported_result_omitted": True,
        "file_path": "exports/complete.csv",
        "row_count": 4,
        "sql_sha256": hashlib.sha256(sql.encode("utf-8")).hexdigest(),
    }
    assert "truncated" not in receipt and "rows" not in receipt


@pytest.mark.parametrize("tool,_backend", EXPORTS)
@pytest.mark.parametrize("protocol", ["native", "xml"])
def test_successful_simple_export_remains_noneligible(tool, _backend, protocol):
    sql = "SELECT COUNT(*) FROM sales"
    evidence = capture_export(tool, sql, export_manifest(tool, sql), protocol)

    assert evidence.status == "READY"
    assert event_data(evidence)[0]["outcome"] == "business_success"
    assert event_data(evidence)[0]["nonroutine"] is False
    assert Owner._episode_signal([evidence.event_json]) == (False, False, False)


@pytest.mark.parametrize("tool,_backend", EXPORTS)
def test_export_of_discovery_sql_preserves_metadata_classification(tool, _backend):
    sql = "SELECT column_name FROM information_schema.columns"
    evidence = capture_export(tool, sql, export_manifest(tool, sql))

    assert event_data(evidence)[0]["outcome"] == "metadata_success"
    assert event_data(evidence)[0]["metadata"] is True
    assert Owner._episode_signal([evidence.event_json]) == (False, False, False)


def invalid_export_result(case, tool, backend, sql):
    if case == "failure":
        return f"{backend} CSV export failed during fetch"
    if case == "malformed":
        return "completed export at exports/complete.csv"
    if case == "incomplete":
        return export_manifest(tool, sql, completed=False)
    if case == "missing_key":
        payload = json.loads(export_manifest(tool, sql))
        del payload["byte_size"]
        return json.dumps(payload)
    if case == "wrong_backend":
        return export_manifest(tool, sql, backend="Teradata" if backend == "Impala" else "Impala")
    if case == "wrong_digest":
        return export_manifest(tool, sql, sql_sha256=hashlib.sha256(b"different sql").hexdigest())
    if case == "wrong_type":
        return export_manifest(tool, sql, row_count=True)
    if case == "column_over_producer_bound":
        return export_manifest(tool, sql, columns=["c" * 513])
    if case == "database_over_producer_bound":
        return export_manifest(tool, sql, database="d" * 513)
    return export_manifest(tool, sql, unexpected="not produced")


@pytest.mark.parametrize("tool,backend", EXPORTS)
@pytest.mark.parametrize("protocol", ["native", "xml"])
@pytest.mark.parametrize("case", [
    "failure", "malformed", "incomplete", "missing_key", "wrong_backend",
    "wrong_digest", "wrong_type", "column_over_producer_bound",
    "database_over_producer_bound", "extra_key",
])
def test_failed_or_untrusted_export_results_never_create_positive_evidence(
        tool, backend, protocol, case):
    sql = "SELECT * FROM customers JOIN sales USING (customer_id)"
    result = invalid_export_result(case, tool, backend, sql)
    evidence = capture_export(tool, sql, result, protocol)

    assert evidence.status == "READY"
    events = event_data(evidence)
    assert len(events) == 1
    assert events[0]["outcome"] == ("failure" if case == "failure" else "unknown")
    assert not any(event["outcome"] == "business_success" for event in events)
    assert Owner._episode_signal([evidence.event_json]) == (False, False, False)
    if case != "failure":
        assert "unclassified_sql_result_omitted" in evidence.messages_json
        assert "exports/complete.csv" not in evidence.messages_json


@pytest.mark.parametrize("tool,_backend", EXPORTS)
@pytest.mark.parametrize("protocol", ["native", "xml"])
def test_unmatched_export_response_cannot_invent_sql_success(tool, _backend, protocol):
    sql = "SELECT * FROM customers JOIN sales USING (customer_id)"
    manifest = export_manifest(tool, sql)
    item = Evidence("orphan-session", f"orphan-{protocol}-{tool}")
    item.add({"role": "user", "content": "Do not infer an originating call"})
    if protocol == "native":
        item.add({"role": "tool", "tool_call_id": "orphan", "content": manifest})
    else:
        item.add({
            "role": "user",
            "content": ToolProtocol.format_hermes_tool_response(
                tool, manifest, tool_call_id="orphan"
            ),
        })
    item.add({"role": "assistant", "content": "No originating export was observed."})
    evidence = item.finish()

    assert evidence.status == "READY"
    assert event_data(evidence) == []
    assert Owner._episode_signal([evidence.event_json]) == (False, False, False)


def test_actual_streaming_manifest_captures_metadata_not_complete_csv_rows(tmp_path):
    sql = (
        "SELECT a.customer_id, a.amount FROM synthetic a "
        "JOIN synthetic b ON b.customer_id = a.customer_id ORDER BY a.customer_id"
    )
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute("CREATE TABLE synthetic(customer_id INTEGER, amount TEXT)")
        connection.executemany("INSERT INTO synthetic VALUES (?, ?)", [
            (1, "PRIVATE-ALPHA"), (2, "PRIVATE-BETA"), (3, "PRIVATE-GAMMA"),
        ])
        cursor = connection.execute(sql)
        output = tmp_path / "complete.csv"
        raw_manifest = db_tools._stream_csv(
            cursor, str(output), 2, "warehouse", "Impala", sql, str(tmp_path), False
        )
    finally:
        connection.close()

    with output.open(newline="", encoding="utf-8") as handle:
        complete_rows = list(csv.reader(handle))
    assert complete_rows == [
        ["customer_id", "amount"],
        ["1", "PRIVATE-ALPHA"],
        ["2", "PRIVATE-BETA"],
        ["3", "PRIVATE-GAMMA"],
    ]

    evidence = capture_export("export_impala_csv", sql, raw_manifest)
    receipt = projected_receipt(evidence, "native")
    assert receipt["completed"] is True and receipt["row_count"] == 3
    assert receipt["byte_size"] == output.stat().st_size
    assert receipt["file_path"] == "complete.csv"
    assert "PRIVATE-ALPHA" not in evidence.messages_json
    assert "PRIVATE-BETA" not in evidence.messages_json
    assert "PRIVATE-GAMMA" not in evidence.messages_json


@pytest.mark.parametrize("tool,backend", EXPORTS)
@pytest.mark.parametrize("protocol", ["native", "xml"])
def test_actual_streaming_manifest_accepts_producer_boundary_labels_with_bounded_receipt(
        tmp_path, tool, backend, protocol):
    expression_prefix = "a.value || '"
    expression = expression_prefix + ("x" * (512 - len(expression_prefix) - 1)) + "'"
    database = "d" * 512
    assert len(expression) == len(database) == db_tools._MAX_COLUMN_CHARS == 512
    sql = (
        f"SELECT {expression} FROM synthetic a "
        "JOIN synthetic b ON b.id = a.id ORDER BY a.id"
    )
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute("CREATE TABLE synthetic(id INTEGER, value TEXT)")
        connection.execute("INSERT INTO synthetic VALUES (1, 'PRIVATE-BOUNDARY')")
        cursor = connection.execute(sql)
        output = tmp_path / f"{tool}-{protocol}.csv"
        raw_manifest = db_tools._stream_csv(
            cursor, str(output), 2, database, backend, sql, str(tmp_path), False
        )
    finally:
        connection.close()

    manifest = json.loads(raw_manifest)
    assert manifest["columns"] == [expression]
    assert manifest["database"] == database
    with output.open(newline="", encoding="utf-8") as handle:
        complete_rows = list(csv.reader(handle))
    assert complete_rows == [[expression], ["PRIVATE-BOUNDARY" + ("x" * 499)]]

    evidence = capture_export(tool, sql, raw_manifest, protocol)
    assert event_data(evidence)[0]["outcome"] == "business_success"
    receipt = projected_receipt(evidence, protocol)
    assert receipt["columns"] == [expression[:256]]
    assert receipt["columns_truncated"] is True
    assert receipt["database"] == database[:256]
    assert receipt["database_truncated"] is True
    assert receipt["exported_result_omitted"] is True
    assert receipt["completed"] is True and receipt["row_count"] == 1
    assert "rows" not in receipt and "truncated" not in receipt
    assert "PRIVATE-BOUNDARY" not in evidence.messages_json


def native_export_message(tool, sql, call_id="scheduled-export"):
    call = SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=tool, arguments=json.dumps({"sql": sql})),
    )
    message = MagicMock(content="", tool_calls=[call])
    message.model_dump.return_value = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": tool, "arguments": json.dumps({"sql": sql})},
        }],
    }
    return message


def xml_export_message(tool, sql):
    return SimpleNamespace(
        content="<tool_call>" + json.dumps({
            "name": tool, "arguments": {"sql": sql},
        }) + "</tool_call>",
        tool_calls=None,
    )


@pytest.mark.parametrize("protocol,tool", [
    ("native", "export_impala_csv"),
    ("xml", "export_teradata_csv"),
])
def test_actual_agent_owner_mailbox_and_service_schedule_successful_export(
        tmp_path, protocol, tool):
    instance, roster = make_agent(tmp_path)
    sql = "SELECT * FROM customers JOIN sales USING (customer_id)"
    requests = []
    instance.use_hermes_xml_protocol = protocol == "xml"
    tool_message = (
        xml_export_message(tool, sql) if protocol == "xml"
        else native_export_message(tool, sql)
    )
    try:
        with patch.object(instance, "step", side_effect=[
                tool_message, answer_message("Complete CSV exported.")]), \
                patch.object(instance, "manage_context"), \
                patch.object(agent_module.registry, "execute",
                             return_value=export_manifest(tool, sql)):
            assert instance.run("Export the joined customer sales report") == "Complete CSV exported."

        assert instance.last_skill_admission.status == "ELIGIBLE"
        sources = rows(instance.skill_review_owner, "episode_sources")
        assert len(sources) == 1
        assert event_data(SimpleNamespace(event_json=sources[0]["event_json"]))[0]["outcome"] == \
            "business_success"
        assert "exported_result_omitted" in sources[0]["messages_json"]

        instance.skill_review_owner.flush_session(instance.session_id)
        instance.skill_review_owner.pump()
        service = ReviewService(
            roster, provider=lambda request: requests.append(request) or {"action": "NONE"}
        )
        assert service.once() == 1
        assert len(requests) == 1
        assert "exported_result_omitted" in requests[0]
        assert "rows" not in projected_receipt(
            SimpleNamespace(messages_json=sources[0]["messages_json"]), protocol
        )
    finally:
        instance.shutdown_skill_reviews()
