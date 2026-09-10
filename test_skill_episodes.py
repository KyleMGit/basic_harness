"""Selective episode admission at the real agent caller and scheduler boundaries."""
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import agent as agent_module
import pytest
from skill_review import Owner, ReviewService, connect
from test_async_skill_review import create_proposal, rows
from test_skill_review_integration import answer_message, make_agent


def tool_message(name, arguments, call_id="sql-call"):
    call = SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )
    message = SimpleNamespace(content="", tool_calls=[call])
    message.model_dump = lambda exclude_none=True: {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }],
    }
    return message


def sql_result(rows_value=None):
    values = [["private-business-value"]] if rows_value is None else rows_value
    return json.dumps({
        "database": "warehouse",
        "columns": ["amount"],
        "rows": values,
        "row_count": len(values),
        "truncated": False,
    })


def run_answer(instance, task):
    with patch.object(instance, "step", return_value=answer_message()), \
         patch.object(instance, "manage_context"):
        return instance.run(task)


def run_sql(instance, task, sql, result):
    replies = [tool_message("query_teradata", {"sql": sql}), answer_message("Verified procedure completed.")]
    with patch.object(instance, "step", side_effect=replies), \
         patch.object(instance, "manage_context"), \
         patch.object(agent_module.registry, "execute", return_value=result):
        return instance.run(task)


def xml_sql_message(sql):
    return SimpleNamespace(
        content=("<tool_call>\n" + json.dumps({
            "name": "query_teradata", "arguments": {"sql": sql},
        }) + "\n</tool_call>"),
        tool_calls=None,
    )


@pytest.mark.parametrize("first_sql,first_result", [
    ("SELECT column_name FROM information_schema.columns", json.dumps({
        "database": "warehouse", "columns": ["column_name"], "rows": [["amount"]],
        "row_count": 1, "truncated": False,
    })),
    ("SELECT bad_fn(amount) FROM sales", "Teradata query failed: syntax error"),
])
def test_actual_agent_xml_multiple_rounds_match_runtime_call_ids(first_sql, first_result, tmp_path):
    instance, _ = make_agent(tmp_path)
    instance.use_hermes_xml_protocol = True
    replies = [
        xml_sql_message(first_sql),
        xml_sql_message("SELECT SUM(amount) OVER () FROM sales"),
        answer_message("Verified multi-round procedure completed."),
    ]
    try:
        with patch.object(instance, "step", side_effect=replies), \
             patch.object(instance, "manage_context"), \
             patch.object(agent_module.registry, "execute", side_effect=[first_result, sql_result()]):
            assert instance.run("Investigate and verify the reusable SQL procedure")
        saved = rows(instance.skill_review_owner, "episode_sources")
        assert len(saved) == 1
        event_data = json.loads(saved[0]["event_json"])
        assert [event["outcome"] for event in event_data["events"]] == [
            "metadata_success" if "information_schema" in first_sql else "failure",
            "business_success",
        ]
        assert json.loads(saved[0]["messages_json"])[-1]["content"] == "Verified multi-round procedure completed."
    finally:
        instance.shutdown_skill_reviews()


def prepare_active_episode(instance):
    run_sql(instance, "Find customer totals", "SELECT bad_fn(amount) FROM sales",
            "Teradata query failed: syntax error")
    run_sql(instance, "Use the supported window workaround",
            "SELECT SUM(amount) OVER () FROM sales", sql_result())
    instance.skill_review_owner.flush_session(instance.session_id)
    instance.skill_review_owner.pump()
    assert rows(instance.skill_review_owner, "jobs")[-1]["status"] == "PREPARED"


@pytest.mark.parametrize("stage", ["PREPARED", "RUNNING", "RESULT"])
def test_routine_new_sql_shape_during_active_job_keeps_frozen_review_valid(tmp_path, stage):
    instance, roster = make_agent(tmp_path)
    requests = []
    entered, release = threading.Event(), threading.Event()
    worker = None

    def provider(request):
        requests.append(request)
        if stage == "RUNNING":
            entered.set()
            release.wait(4)
        return {"action": "NONE"}

    try:
        prepare_active_episode(instance)
        service = ReviewService(roster, provider=provider)
        if stage == "RUNNING":
            worker = threading.Thread(target=service.once)
            worker.start()
            assert entered.wait(2)
        elif stage == "RESULT":
            assert service.once() == 1
            assert rows(instance.skill_review_owner, "jobs")[-1]["status"] == "RESULT"

        run_sql(instance, "Show the count for the same report",
                "SELECT COUNT(amount) FROM sales", sql_result())
        if stage == "PREPARED":
            assert service.once() == 1
        elif stage == "RUNNING":
            release.set()
            worker.join(4)

        instance.skill_review_owner.pump()
        assert len(requests) == 1
        assert rows(instance.skill_review_owner, "jobs")[-1]["status"] == "NONE"
        assert not [source for source in rows(instance.skill_review_owner, "episode_sources")
                    if source["job_id"]]
    finally:
        release.set()
        if worker:
            worker.join(5)
        instance.shutdown_skill_reviews()


@pytest.mark.parametrize("stage", ["PREPARED", "RUNNING", "RESULT"])
def test_new_failure_during_active_job_is_retained_for_later_legitimate_review(tmp_path, stage):
    instance, roster = make_agent(tmp_path)
    requests = []
    entered, release = threading.Event(), threading.Event()
    worker = None

    def provider(request):
        requests.append(request)
        if stage == "RUNNING" and len(requests) == 1:
            entered.set()
            release.wait(4)
        return {"action": "NONE"}

    try:
        prepare_active_episode(instance)
        service = ReviewService(roster, provider=provider)
        if stage == "RUNNING":
            worker = threading.Thread(target=service.once)
            worker.start()
            assert entered.wait(2)
        elif stage == "RESULT":
            assert service.once() == 1

        run_sql(instance, "Try the additional reusable expression",
                "SELECT another_bad_fn(amount) FROM sales",
                "Teradata query failed: unsupported function")
        if stage == "PREPARED":
            assert service.once() == 1
        elif stage == "RUNNING":
            release.set()
            worker.join(4)
        instance.skill_review_owner.pump()
        assert len(requests) == 1
        deferred = rows(instance.skill_review_owner, "episode_sources")
        assert len(deferred) == 1 and deferred[0]["job_id"] is None
        assert "another_bad_fn" in deferred[0]["messages_json"]

        run_sql(instance, "Use the supported QUALIFY resolution",
                "SELECT * FROM sales QUALIFY ROW_NUMBER() OVER (ORDER BY sale_date DESC)=1",
                sql_result())
        instance.skill_review_owner.flush_session(instance.session_id)
        instance.skill_review_owner.pump()
        assert ReviewService(roster, provider=provider).once() == 1
        instance.skill_review_owner.pump()
        assert len(requests) == 2
        assert "another_bad_fn" in requests[1] and "QUALIFY" in requests[1]
    finally:
        release.set()
        if worker:
            worker.join(5)
        instance.shutdown_skill_reviews()


def test_actual_agent_trivial_and_routine_turns_make_zero_review_requests(tmp_path):
    instance, roster = make_agent(tmp_path)
    requests = []
    try:
        assert run_answer(instance, "Thanks, please acknowledge this message") == "The task answer."
        assert run_answer(instance, "Format the answer as bullets") == "The task answer."
        instance.skill_review_owner.flush_session(instance.session_id)
        instance.skill_review_owner.pump()
        assert ReviewService(roster, provider=lambda request: requests.append(request) or {"action": "NONE"}).once() == 0
        assert requests == []
    finally:
        instance.shutdown_skill_reviews()


def test_actual_agent_related_failure_resolution_and_routine_variants_consolidate(tmp_path):
    instance, roster = make_agent(tmp_path)
    requests = []
    try:
        failed = "Teradata query failed (stage=execute): ProgrammingError [3706]: syntax error"
        assert run_sql(instance, "Find reusable customer totals", "SELECT bad_fn(amount) FROM sales GROUP BY customer_id", failed)
        assert run_sql(instance, "Continue: use the supported window workaround", "SELECT customer_id, SUM(amount) OVER (PARTITION BY customer_id) FROM sales", sql_result())
        assert run_sql(instance, "Use the same result for last month", "SELECT customer_id, SUM(amount) OVER (PARTITION BY customer_id) FROM sales WHERE sale_date >= DATE '2026-08-01'", sql_result())
        assert run_sql(instance, "Sort it descending and limit 10", "SELECT customer_id, SUM(amount) OVER (PARTITION BY customer_id) FROM sales ORDER BY 2 DESC LIMIT 10", sql_result())

        instance.skill_review_owner.flush_session(instance.session_id)
        instance.skill_review_owner.pump()
        service = ReviewService(roster, provider=lambda request: requests.append(request) or {"action": "NONE"})
        assert service.once() == 1
        instance.skill_review_owner.pump()
        instance.skill_review_owner.pump()
        assert service.once() == 0
        assert len(requests) == 1
        prepared = json.loads(requests[0])
        assert len(prepared["tasks"]) == 1
        rendered = json.dumps(prepared)
        assert "bad_fn" in rendered and "SUM(amount) OVER" in rendered
        assert "private-business-value" not in rendered
        assert "business_result_omitted" in rendered
    finally:
        instance.shutdown_skill_reviews()


def test_verified_correction_after_ack_is_ready_immediately_but_routine_followup_is_not(tmp_path):
    instance, roster = make_agent(tmp_path)
    requests = []
    service = ReviewService(roster, provider=lambda request: requests.append(request) or create_proposal("window_totals"))
    try:
        failed = "Teradata query failed (stage=execute): ProgrammingError [3706]: syntax error"
        run_sql(instance, "Find reusable totals", "SELECT bad_fn(amount) FROM sales", failed)
        run_sql(instance, "Continue with a supported workaround", "SELECT SUM(amount) OVER () FROM sales", sql_result())
        instance.skill_review_owner.flush_session(instance.session_id)
        instance.skill_review_owner.pump()
        assert service.once() == 1
        instance.skill_review_owner.pump()

        run_sql(instance, "No, that procedure is wrong; use QUALIFY after ROW_NUMBER instead", "SELECT * FROM sales QUALIFY ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY sale_date DESC)=1", sql_result())
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and not rows(instance.skill_review_owner, "jobs")[-1]["status"] == "PREPARED":
            instance.skill_review_owner.pump()
            time.sleep(.01)
        correction_service = ReviewService(roster, provider=lambda request: requests.append(request) or {"action": "NONE"})
        assert correction_service.once() == 1
        instance.skill_review_owner.pump()
        correction_request = json.loads(requests[1])
        assert correction_request["tasks"][0]["related_skills"] == [
            {"action": "CREATE", "name": "window_totals"}
        ]

        run_sql(instance, "Now limit that to 20 rows", "SELECT * FROM sales QUALIFY ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY sale_date DESC)=1 LIMIT 20", sql_result())
        instance.skill_review_owner.flush_session(instance.session_id)
        instance.skill_review_owner.pump()
        assert correction_service.once() == 0
        assert len(requests) == 2
    finally:
        instance.shutdown_skill_reviews()


def test_correction_during_running_revision_cannot_publish_obsolete_proposal(tmp_path):
    instance, roster = make_agent(tmp_path)
    entered, release = threading.Event(), threading.Event()
    old_service = ReviewService(roster, provider=lambda _: (entered.set(), release.wait(4), create_proposal("obsolete"))[2])
    worker = None
    try:
        failed = "Teradata query failed: syntax error"
        run_sql(instance, "Find reusable totals", "SELECT bad_fn(amount) FROM sales", failed)
        run_sql(instance, "Use the supported window workaround", "SELECT SUM(amount) OVER () FROM sales", sql_result())
        instance.skill_review_owner.flush_session(instance.session_id)
        instance.skill_review_owner.pump()
        worker = threading.Thread(target=old_service.once)
        worker.start()
        assert entered.wait(2)

        run_sql(instance, "No, that procedure is wrong; use QUALIFY instead", "SELECT * FROM sales QUALIFY ROW_NUMBER() OVER (ORDER BY sale_date DESC)=1", sql_result())
        release.set()
        worker.join(4)
        instance.skill_review_owner.pump()
        assert "obsolete" not in instance.skill_store.list_skills()

        instance.skill_review_owner.pump()
        assert ReviewService(roster, provider=lambda _: {"action": "NONE"}).once() == 1
        instance.skill_review_owner.pump()
        assert "obsolete" not in instance.skill_store.list_skills()
    finally:
        release.set()
        if worker:
            worker.join(5)
        instance.shutdown_skill_reviews()


def test_next_agent_question_completes_while_service_episode_preparation_is_blocked(tmp_path):
    instance, roster = make_agent(tmp_path)
    entered, release = threading.Event(), threading.Event()
    original = ReviewService._episode_request
    worker = None

    def blocked(service, entry, job):
        entered.set()
        release.wait(4)
        return original(service, entry, job)

    try:
        failed = "Teradata query failed: syntax error"
        run_sql(instance, "Find reusable totals", "SELECT bad_fn(amount) FROM sales", failed)
        run_sql(instance, "Use the supported window workaround", "SELECT SUM(amount) OVER () FROM sales", sql_result())
        instance.skill_review_owner.flush_session(instance.session_id)
        instance.skill_review_owner.pump()
        service = ReviewService(roster, provider=lambda _: {"action": "NONE"})
        with patch.object(ReviewService, "_episode_request", blocked):
            worker = threading.Thread(target=service.once)
            worker.start()
            assert entered.wait(2)
            started = time.monotonic()
            assert run_answer(instance, "What is the next foreground answer?") == "The task answer."
            assert time.monotonic() - started < .5
            assert worker.is_alive()
            release.set()
            worker.join(4)
    finally:
        release.set()
        if worker:
            worker.join(5)
        instance.shutdown_skill_reviews()


@pytest.mark.parametrize("stage", ["before_claim", "after_result"])
def test_correction_before_claim_or_after_result_supersedes_old_revision(tmp_path, stage):
    instance, roster = make_agent(tmp_path)
    old_calls, fresh_calls = [], []
    try:
        failed = "Teradata query failed: syntax error"
        run_sql(instance, "Find reusable totals", "SELECT bad_fn(amount) FROM sales", failed)
        run_sql(instance, "Use the supported window workaround", "SELECT SUM(amount) OVER () FROM sales", sql_result())
        instance.skill_review_owner.flush_session(instance.session_id)
        instance.skill_review_owner.pump()
        if stage == "after_result":
            assert ReviewService(roster, provider=lambda request: old_calls.append(request) or create_proposal("obsolete")).once() == 1
            assert rows(instance.skill_review_owner, "jobs")[0]["status"] == "RESULT"

        run_sql(instance, "No, that procedure is wrong; use QUALIFY instead", "SELECT * FROM sales QUALIFY ROW_NUMBER() OVER (ORDER BY sale_date DESC)=1", sql_result())
        instance.skill_review_owner.pump()
        assert rows(instance.skill_review_owner, "jobs")[0]["status"] == "SUPERSEDED"
        assert "obsolete" not in instance.skill_store.list_skills()

        instance.skill_review_owner.pump()
        service = ReviewService(roster, provider=lambda request: fresh_calls.append(request) or {"action": "NONE"})
        assert service.once() == 1
        instance.skill_review_owner.pump()
        assert len(fresh_calls) == 1 and "QUALIFY" in fresh_calls[0]
        assert "SUM(amount) OVER" in fresh_calls[0]
    finally:
        instance.shutdown_skill_reviews()


def test_failed_correction_keeps_episode_challenged_until_later_verification(tmp_path):
    instance, roster = make_agent(tmp_path)
    requests = []
    try:
        failed = "Teradata query failed: syntax error"
        run_sql(instance, "Find reusable totals", "SELECT bad_fn(amount) FROM sales", failed)
        run_sql(instance, "Use the supported window workaround", "SELECT SUM(amount) OVER () FROM sales", sql_result())
        instance.skill_review_owner.flush_session(instance.session_id)
        instance.skill_review_owner.pump()

        run_sql(instance, "No, that procedure is wrong; use QUALIFY instead", "SELECT * FROM sales QUALIFY bad_fn(amount)=1", failed)
        instance.skill_review_owner.flush_session(instance.session_id)
        instance.skill_review_owner.pump()
        assert ReviewService(roster, provider=lambda request: requests.append(request) or {"action": "NONE"}).once() == 0
        assert requests == []

        run_sql(instance, "Continue the correction with verified ROW_NUMBER", "SELECT * FROM sales QUALIFY ROW_NUMBER() OVER (ORDER BY sale_date DESC)=1", sql_result())
        instance.skill_review_owner.pump()
        service = ReviewService(roster, provider=lambda request: requests.append(request) or {"action": "NONE"})
        assert service.once() == 1
        instance.skill_review_owner.pump()
        assert len(requests) == 1 and "ROW_NUMBER" in requests[0]
    finally:
        instance.shutdown_skill_reviews()


def test_parameter_correction_wording_does_not_create_second_request(tmp_path):
    instance, roster = make_agent(tmp_path)
    requests = []
    service = ReviewService(roster, provider=lambda request: requests.append(request) or {"action": "NONE"})
    try:
        run_sql(instance, "Find customer totals", "SELECT bad_fn(amount) FROM sales", "Teradata query failed: syntax error")
        run_sql(instance, "Use the supported window workaround", "SELECT SUM(amount) OVER () FROM sales", sql_result())
        instance.skill_review_owner.flush_session(instance.session_id)
        instance.skill_review_owner.pump()
        assert service.once() == 1
        instance.skill_review_owner.pump()
        run_sql(instance, "No, give me two months instead.", "SELECT SUM(amount) OVER () FROM sales WHERE sale_date >= DATE '2026-07-01'", sql_result())
        instance.skill_review_owner.flush_session(instance.session_id)
        instance.skill_review_owner.pump()
        service.once()
        instance.skill_review_owner.pump()
        assert len(requests) == 1, "Routine parameter correction was independently reviewed"
    finally:
        instance.shutdown_skill_reviews()


def test_failed_corrective_turn_cannot_publish_old_result(tmp_path):
    instance, roster = make_agent(tmp_path)
    service = ReviewService(roster, provider=lambda request: create_proposal("obsolete_parent_probe"))
    try:
        run_sql(instance, "Find customer totals", "SELECT bad_fn(amount) FROM sales", "Teradata query failed: syntax error")
        run_sql(instance, "Use the supported window workaround", "SELECT SUM(amount) OVER () FROM sales", sql_result())
        instance.skill_review_owner.flush_session(instance.session_id)
        instance.skill_review_owner.pump()
        assert service.once() == 1
        with patch.object(instance, "step", side_effect=RuntimeError("synthetic interrupted correction")), patch.object(instance, "manage_context"):
            result = instance.run("No, that procedure is wrong; the join duplicates rows. Use a verified deduplicated join instead.")
        assert "LLM API Error" in result
        instance.skill_review_owner.pump()
        assert "obsolete_parent_probe" not in instance.skill_store.list_skills()
    finally:
        instance.shutdown_skill_reviews()


def test_failed_routine_parameter_wording_does_not_challenge_pending_result(tmp_path):
    instance, roster = make_agent(tmp_path)
    service = ReviewService(roster, provider=lambda request: create_proposal("routine_result_survives"))
    try:
        run_sql(instance, "Find customer totals", "SELECT bad_fn(amount) FROM sales", "Teradata query failed: syntax error")
        run_sql(instance, "Use the supported window workaround", "SELECT SUM(amount) OVER () FROM sales", sql_result())
        instance.skill_review_owner.flush_session(instance.session_id)
        instance.skill_review_owner.pump()
        assert service.once() == 1
        with patch.object(instance, "step", side_effect=RuntimeError("synthetic interrupted parameter edit")), patch.object(instance, "manage_context"):
            result = instance.run("No, that procedure uses the wrong month; use two months instead.")
        assert "LLM API Error" in result
        instance.skill_review_owner.pump()
        assert "routine_result_survives" in instance.skill_store.list_skills()
    finally:
        instance.shutdown_skill_reviews()


def test_correction_after_ack_retains_original_request_anchor(tmp_path):
    instance, roster = make_agent(tmp_path)
    requests = []
    service = ReviewService(roster, provider=lambda request: requests.append(request) or {"action": "NONE"})
    original = "Find customer totals using the ORIGINAL-ANCHOR customer ownership definition"
    try:
        run_sql(instance, original, "SELECT bad_fn(amount) FROM sales", "Teradata query failed: syntax error")
        run_sql(instance, "Use the supported window workaround", "SELECT SUM(amount) OVER () FROM sales", sql_result())
        instance.skill_review_owner.flush_session(instance.session_id)
        instance.skill_review_owner.pump()
        assert service.once() == 1
        instance.skill_review_owner.pump()
        episode = rows(instance.skill_review_owner, "episodes")[0]
        anchor = json.loads(episode["anchor_json"])
        assert anchor["context_only"] is True and anchor["kind"] == "original_request"
        assert episode["anchor_bytes"] == len(episode["anchor_json"].encode())
        assert episode["anchor_messages"] == 1
        with connect(instance.skill_review_owner.entry, readonly=True) as conn:
            assert Owner._payload_usage(conn) >= episode["anchor_bytes"]
        run_sql(instance, "No, that procedure is wrong; use QUALIFY instead.", "SELECT * FROM sales QUALIFY ROW_NUMBER() OVER (ORDER BY sale_date DESC)=1", sql_result())
        instance.skill_review_owner.pump()
        assert service.once() == 1
        assert len(requests) == 2
        assert "ORIGINAL-ANCHOR" in requests[1], "Related correction lost its original request context"
        prepared = json.loads(requests[1])
        assert prepared["tasks"][0]["context_only_anchor"]["context_only"] is True
    finally:
        instance.shutdown_skill_reviews()


def test_active_then_interrupted_challenge_stays_revoked_across_reconnect(tmp_path):
    instance, roster = make_agent(tmp_path)
    service = ReviewService(roster, provider=lambda _: create_proposal("obsolete_active_probe"))
    entered, release = threading.Event(), threading.Event()
    worker = None
    try:
        run_sql(instance, "Find customer totals", "SELECT bad_fn(amount) FROM sales", "Teradata query failed: syntax error")
        run_sql(instance, "Use the supported window workaround", "SELECT SUM(amount) OVER () FROM sales", sql_result())
        instance.skill_review_owner.flush_session(instance.session_id)
        instance.skill_review_owner.pump()
        assert service.once() == 1

        def interrupted_step():
            entered.set()
            release.wait(4)
            raise RuntimeError("synthetic interrupted correction")

        with patch.object(instance, "step", side_effect=interrupted_step), patch.object(instance, "manage_context"):
            worker = threading.Thread(target=lambda: instance.run(
                "No, that procedure is wrong; the join duplicates rows. Use a verified deduplicated join instead."))
            worker.start()
            assert entered.wait(2)
            assert rows(instance.skill_review_owner, "episodes")[0]["state"] == "CHALLENGED"
            instance.skill_review_owner.pump()
            assert "obsolete_active_probe" not in instance.skill_store.list_skills()
            release.set()
            worker.join(4)

        instance.skill_review_owner.close()
        instance.skill_review_owner = Owner(roster, "user-0", instance.skill_store, background=False)
        instance.skill_review_owner.pump()
        assert rows(instance.skill_review_owner, "episodes")[0]["state"] == "CHALLENGED"
        assert "obsolete_active_probe" not in instance.skill_store.list_skills()
    finally:
        release.set()
        if worker:
            worker.join(5)
        instance.shutdown_skill_reviews()


def test_explicit_inactive_challenge_retirement_never_revives_obsolete_result(tmp_path):
    instance, roster = make_agent(tmp_path)
    try:
        prepare_active_episode(instance)
        assert ReviewService(roster, provider=lambda _: create_proposal("must_not_revive")).once() == 1
        assert rows(instance.skill_review_owner, "jobs")[-1]["status"] == "RESULT"
        challenge = instance.skill_review_owner.begin_turn(
            instance.session_id,
            "No, that procedure is wrong; the join duplicates rows. Use a verified deduplicated join.",
        )
        assert challenge.status == "CHALLENGED"
        instance.skill_review_owner.flush_session(instance.session_id, retire=True)
        instance.skill_review_owner.pump()
        assert rows(instance.skill_review_owner, "episodes") == []
        assert rows(instance.skill_review_owner, "episode_sources") == []
        assert rows(instance.skill_review_owner, "jobs")[-1]["status"] == "SUPERSEDED"
        assert "must_not_revive" not in instance.skill_store.list_skills()
    finally:
        instance.shutdown_skill_reviews()
