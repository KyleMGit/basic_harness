"""Host-generated asynchronous skill-review outcome notice contracts."""
from io import StringIO
import json
from pathlib import Path
import sqlite3
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from skill_review import (BudgetRefusal, NOTICE_DETAIL_COUNT, NOTICE_SUMMARY_COUNT_LIMIT,
                          Owner, ReviewService, notice_seal, packed, read_auth,
                          review_start_seal, valid_budget_diagnostic)
from test_async_skill_review import (
    create_proposal,
    evidence,
    owner_for,
    rows,
    setup_roster,
)


def complete(owner, roster, provider, *, session="session", task="task"):
    assert owner.enqueue(evidence(session=session, task=task)).status == "ACCEPTED"
    owner.pump()
    assert ReviewService(roster, provider=provider).once() == 1
    owner.pump()


def rendered(owner):
    output = StringIO()
    owner.deliver_notices(output)
    return output.getvalue()


def test_authorized_worker_entry_records_one_start_before_blocked_review_result(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    dispatched = threading.Event()
    enter_worker = threading.Event()
    provider_entered = threading.Event()
    release_provider = threading.Event()
    calls = []

    def provider(request):
        calls.append(request)
        provider_entered.set()
        assert release_provider.wait(4)
        return {"action": "NONE"}

    service = ReviewService(roster, provider=provider)
    original_infer = service._infer

    def paused_before_worker_entry(*args, **kwargs):
        dispatched.set()
        assert enter_worker.wait(4)
        return original_infer(*args, **kwargs)

    service._infer = paused_before_worker_entry
    worker = None
    try:
        assert owner.enqueue(evidence(session="start-session", task="start-task")).status == "ACCEPTED"
        assert rows(owner, "notices") == []
        owner.pump()
        assert rows(owner, "jobs")[-1]["status"] == "PREPARED"
        assert rows(owner, "notices") == []

        worker = threading.Thread(target=service.once)
        worker.start()
        assert dispatched.wait(2)
        owner.pump()
        assert rows(owner, "jobs")[-1]["status"] == "RUNNING"
        assert rows(owner, "notices") == []

        enter_worker.set()
        assert provider_entered.wait(2)
        running = rows(owner, "jobs")[-1]
        assert running["review_start_seal"]
        assert rows(owner, "notices") == []  # The service cannot write the owner outbox.
        owner.pump()
        pending = rows(owner, "notices")
        assert len(pending) == 1
        assert pending[0]["kind"] == "START"
        start_payload = json.loads(pending[0]["payload_json"])
        assert set(start_payload) == {"version", "kind", "started_at", "source"}
        assert "Review started" in rendered(owner)
        assert len(calls) == 1

        release_provider.set()
        worker.join(4)
        assert not worker.is_alive()
        owner.pump()
        remaining = rows(owner, "notices")
        assert [row["kind"] for row in remaining] == ["START", "DETAIL"]
        assert remaining[0]["delivered"] is not None
        assert remaining[1]["delivered"] is None
        assert "Review completed with no skill change" in rendered(owner)
        assert len(calls) == 1
    finally:
        enter_worker.set()
        release_provider.set()
        if worker:
            worker.join(4)
        owner.close()


def test_fast_result_preserves_start_before_final_after_source_cleanup(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    calls = []
    try:
        assert owner.enqueue(evidence(session="fast-session", task="fast-task")).status == "ACCEPTED"
        owner.pump()
        assert ReviewService(
            roster, provider=lambda request: calls.append(request) or {"action": "NONE"}
        ).once() == 1
        job = rows(owner, "jobs")[-1]
        assert job["status"] == "RESULT" and job["review_start_seal"]
        assert rows(owner, "notices") == []

        owner.pump()
        assert not rows(owner, "evidence")
        notices = rows(owner, "notices")
        assert [row["kind"] for row in notices] == ["START", "DETAIL"]
        assert notices[0]["created"] <= notices[1]["created"]
        start = json.loads(notices[0]["payload_json"])
        assert start["source"]["session_ids"] == ["fast-session"]
        assert start["source"]["task_ids"] == ["fast-task"]
        output = rendered(owner)
        assert output.index("[Skill Review] Review started.") < output.index(
            "Review completed with no skill change")
        assert "still running" not in output and "result pending" not in output
        assert len(calls) == 1
    finally:
        owner.close()


def test_started_job_recovery_keeps_one_marker_and_suppresses_duplicate_start(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    first = ReviewService(roster, provider=lambda _: {"action": "NONE"})
    first.generation = "1" * 32
    calls = []
    try:
        assert owner.enqueue(evidence(task="recovered-start")).status == "ACCEPTED"
        owner.pump()
        _, _, entry, job = first._pending()[0]
        assert first._claim(entry, job)

        def interrupted(_):
            raise KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            first._infer(entry, job, interrupted)
        started = rows(owner, "jobs")[-1]
        marker = (started["review_started"], started["review_start_generation"],
                  started["review_start_seal"])
        owner.pump()
        assert [row["kind"] for row in rows(owner, "notices")] == ["START"]
        assert "Review started" in rendered(owner)

        assert ReviewService(
            roster, provider=lambda request: calls.append(request) or {"action": "NONE"}
        ).once() == 1
        recovered = rows(owner, "jobs")[-1]
        assert (recovered["review_started"], recovered["review_start_generation"],
                recovered["review_start_seal"]) == marker
        owner.pump()
        notices = rows(owner, "notices")
        assert [row["kind"] for row in notices] == ["START", "DETAIL"]
        assert len(calls) == 1
        assert "Review started" not in rendered(owner)
    finally:
        owner.close()


def test_cross_profile_signed_start_cannot_authorize_or_notify_worker(tmp_path):
    roster = setup_roster(tmp_path, count=2)
    alice = owner_for(roster, 0, background=False)
    bob = owner_for(roster, 1, background=False)
    service = ReviewService(roster, provider=lambda _: {"action": "NONE"})
    service.generation = "2" * 32
    calls = []
    try:
        assert alice.enqueue(evidence(task="alice-forged-start")).status == "ACCEPTED"
        alice.pump()
        _, _, entry, job = service._pending()[0]
        assert service._claim(entry, job)
        with sqlite3.connect(alice.entry.mailbox) as conn:
            conn.row_factory = sqlite3.Row
            current = dict(conn.execute("SELECT * FROM jobs WHERE job_id=?", (job["job_id"],)).fetchone())
            current["review_started"] = time.time()
            current["review_start_generation"] = service.generation
            current["review_start_seal"] = review_start_seal(read_auth(roster, bob.entry), current)
            conn.execute(
                "UPDATE jobs SET review_started=?,review_start_generation=?,review_start_seal=? WHERE job_id=?",
                (current["review_started"], current["review_start_generation"],
                 current["review_start_seal"], job["job_id"]),
            )
        assert service._infer(
            entry, job, lambda request: calls.append(request) or {"action": "NONE"}
        ) is None
        alice.pump()
        assert calls == []
        assert rows(alice, "notices") == []
        assert rows(bob, "notices") == []
    finally:
        alice.close()
        bob.close()


def test_revocation_after_claim_prevents_start_marker_and_provider_call(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    service = ReviewService(roster, provider=lambda _: {"action": "NONE"})
    service.generation = "3" * 32
    calls = []
    try:
        assert owner.enqueue(evidence(task="revoked-before-start")).status == "ACCEPTED"
        owner.pump()
        _, _, entry, job = service._pending()[0]
        assert service._claim(entry, job)
        owner.set_enabled(False)
        before = Path(entry.mailbox).read_bytes()
        assert service._infer(
            entry, job, lambda request: calls.append(request) or {"action": "NONE"}
        ) is None
        assert calls == []
        assert Path(entry.mailbox).read_bytes() == before
        with sqlite3.connect(entry.mailbox) as conn:
            assert conn.execute(
                "SELECT review_start_seal FROM jobs WHERE job_id=?", (job["job_id"],)
            ).fetchone()[0] is None
    finally:
        owner.close()


def test_service_without_owner_added_start_columns_keeps_legacy_final_protocol(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    try:
        assert owner.enqueue(evidence(task="legacy-capability")).status == "ACCEPTED"
        owner.pump()
    finally:
        owner.close()
    with sqlite3.connect(roster.profiles[0].mailbox) as conn:
        conn.execute("ALTER TABLE jobs DROP COLUMN review_start_seal")
        conn.execute("ALTER TABLE jobs DROP COLUMN review_start_generation")
        conn.execute("ALTER TABLE jobs DROP COLUMN review_started")

    assert ReviewService(roster, provider=lambda _: {"action": "NONE"}).once() == 1
    owner = owner_for(roster, background=False)
    try:
        owner.pump()
        notices = rows(owner, "notices")
        assert [row["kind"] for row in notices] == ["DETAIL"]
        assert "Review started" not in rendered(owner)
        assert rows(owner, "jobs")[-1]["status"] == "NONE"
    finally:
        owner.close()


def test_legacy_final_only_summary_row_remains_valid_and_deliverable(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    try:
        auth = read_auth(roster, owner.entry)
        created = time.time()
        payload = dict(
            version=1, kind="SUMMARY", count=2, outcomes={"FAILED": 1, "NONE": 1},
            first_created=created - 1, last_created=created,
            disclosure="older per-notice names and source identifiers were compacted",
            saturated=False,
        )
        notice = dict(
            notice_id="summary-" + "0" * 32, profile_id=owner.entry.profile_id,
            store_id=owner.entry.store_id, generation=auth["generation"], job_id="0" * 32,
            kind="SUMMARY", payload_json=packed(payload), created=created - 1,
        )
        with sqlite3.connect(owner.entry.mailbox) as conn:
            conn.execute(
                "INSERT INTO notices(notice_id,profile_id,store_id,generation,job_id,kind,payload_json,"
                "created,notice_seal) VALUES(?,?,?,?,?,?,?,?,?)",
                (notice["notice_id"], notice["profile_id"], notice["store_id"], notice["generation"],
                 notice["job_id"], notice["kind"], notice["payload_json"], notice["created"],
                 notice_seal(auth, notice)),
            )
        output = StringIO()
        assert owner.deliver_notices(output) == 1
        assert "2 older outcomes were compacted" in output.getvalue()
        assert "FAILED=1" in output.getvalue() and "NONE=1" in output.getvalue()
    finally:
        owner.close()


def test_committed_create_notice_is_durable_and_independently_acknowledged(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    try:
        assert owner.enqueue(evidence(session="older-session", task="older-task")).status == "ACCEPTED"
        owner.pump()
        assert ReviewService(roster, provider=lambda _: create_proposal("learned_workflow")).once() == 1
        owner.pump()

        assert rows(owner, "jobs")[0]["status"] == "APPLIED"
        pending = rows(owner, "notices")
        assert [row["kind"] for row in pending] == ["START", "DETAIL"]
        assert all(row["delivered"] is None for row in pending)
        assert "instructions" not in str(pending)

        output = StringIO()
        assert owner.deliver_notices(output) == 2
        rendered = output.getvalue()
        assert "Skill Review" in rendered
        assert "Created skill 'learned_workflow'" in rendered
        assert "selected legacy batch" in rendered
        assert "older-session" in rendered and "older-task" in rendered
        assert all(row["delivered"] is not None for row in rows(owner, "notices"))

        assert owner.deliver_notices(StringIO()) == 0
    finally:
        owner.close()


def test_update_none_failed_and_refused_publication_have_truthful_notices(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    try:
        owner.store.save_skill("existing", "Workflow", "Old steps.")

        def update(request):
            target_id = json.loads(request)["catalog"]["targets"][0]["target_id"]
            return dict(action="UPDATE", target_id=target_id, description="Workflow",
                        instructions="Replacement complete steps.", complete=True)

        complete(owner, roster, update, task="update-task")
        complete(owner, roster, lambda _: {"action": "NONE"}, task="none-task")
        complete(owner, roster, lambda _: create_proposal("existing"), task="collision-task")

        def failed(_):
            raise TimeoutError("provider prose must not persist")

        complete(owner, roster, failed, task="failed-task")
        notices = rows(owner, "notices")
        detail_payloads = [json.loads(row["payload_json"]) for row in notices if row["kind"] == "DETAIL"]
        assert all(payload["version"] == 1 and "budget" not in payload for payload in detail_payloads)
        assert [json.loads(row["payload_json"])["outcome"]
                for row in notices if row["kind"] == "DETAIL"] == [
                    "APPLIED", "NONE", "COLLISION", "FAILED"]
        stored = str(notices)
        assert "Replacement complete steps" not in stored
        assert "provider prose" not in stored

        text = rendered(owner)
        assert "Updated skill 'existing'" in text
        assert "Review completed with no skill change" in text
        assert "proposed skill name already exists" in text
        assert "review or publication attempt failed" in text
        assert "Created skill 'existing'" not in text
    finally:
        owner.close()


def test_cache_warning_is_success_and_receipt_recovery_is_not_fresh_publication(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    real_replace = __import__("os").replace

    def fail_cache(source, destination):
        if Path(destination).suffix == ".json":
            raise OSError("cache-only failure")
        return real_replace(source, destination)

    try:
        assert owner.enqueue(evidence(task="cache-warning")).status == "ACCEPTED"
        owner.pump()
        assert ReviewService(roster, provider=lambda _: create_proposal("cache_warning")).once() == 1
        with patch("skill_catalog.os.replace", side_effect=fail_cache):
            owner.pump()
        assert rows(owner, "jobs")[-1]["status"] == "APPLIED"
        assert "Created skill 'cache_warning'" in rendered(owner)

        complete(owner, roster, lambda _: create_proposal("recovered"), task="crash-window")
        first_job = rows(owner, "jobs")[-1]
        # Recreate the real commit-before-ack window with another publication.
        assert owner.enqueue(evidence(task="receipt-window")).status == "ACCEPTED"
        owner.pump()
        assert ReviewService(roster, provider=lambda _: create_proposal("receipt_recovered")).once() == 1
        with patch.object(owner, "_ack", side_effect=OSError("simulated owner crash")):
            owner.pump()
        assert (Path(owner.store.storage_dir) / "receipt_recovered.md").exists()
        assert owner.store.delete_skill("receipt_recovered")
        owner.close()
        owner = owner_for(roster, background=False)
        owner.pump()
        assert rows(owner, "jobs")[-1]["status"] == "DUPLICATE"
        text = rendered(owner)
        assert "Recovered the prior publication receipt for created skill 'receipt_recovered'" in text
        assert "no new publication was made" in text
        assert first_job["job_id"] != rows(owner, "jobs")[-1]["job_id"]
    finally:
        owner.close()


def test_cross_profile_routing_safe_identifiers_and_bad_integrity_fail_closed(tmp_path):
    roster = setup_roster(tmp_path, count=2)
    alice, bob = owner_for(roster, 0, background=False), owner_for(roster, 1, background=False)
    try:
        complete(alice, roster, lambda _: create_proposal("alice_skill"),
                 session="bad\n\x1b[31m\u202elabel", task="bad\rjob")
        complete(bob, roster, lambda _: create_proposal("bob_skill"), task="bob-task")
        alice_text = rendered(alice)
        assert "alice_skill" in alice_text and "bob_skill" not in alice_text
        assert "\x1b" not in alice_text and "\u202e" not in alice_text
        assert "<invalid>" in alice_text
        bob_text = rendered(bob)
        assert "bob_skill" in bob_text and "alice_skill" not in bob_text

        complete(alice, roster, lambda _: create_proposal("tamper_target"), task="tamper")
        with sqlite3.connect(alice.entry.mailbox) as conn:
            conn.execute("UPDATE notices SET payload_json=? WHERE delivered IS NULL",
                         ('{"version":1,"kind":"DETAIL","name":"\\u001b[31mFORGED"}',))
        output = StringIO()
        assert alice.deliver_notices(output) == 0
        assert output.getvalue() == ""
        assert rows(alice, "notices")[-1]["delivered"] == -1
    finally:
        alice.close()
        bob.close()


def test_notice_name_and_identifier_renderers_escape_terminal_controls():
    from skill_review import Owner

    rendered_name = Owner._notice_name("safe\x1b[31m\n\r\u202eend")
    assert rendered_name == "safe\\u001b[31m\\u000a\\u000d\\u202eend"
    assert not any(ord(character) < 32 for character in rendered_name)
    assert Owner._notice_identifier("session\nforged") == "<invalid>"
    assert Owner._notice_identifier("session-safe_1.2:3") == "session-safe_1.2:3"


class FailingStream(StringIO):
    def __init__(self, *, fail_flush=False):
        super().__init__()
        self.fail_flush = fail_flush

    def write(self, value):
        if not self.fail_flush:
            raise OSError("synthetic write failure")
        return super().write(value)

    def flush(self):
        if self.fail_flush:
            raise OSError("synthetic flush failure")
        return super().flush()


@pytest.mark.parametrize("fail_flush", [False, True])
def test_print_or_flush_failure_retains_pending(tmp_path, fail_flush):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    try:
        complete(owner, roster, lambda _: {"action": "NONE"})
        with pytest.raises(OSError):
            owner.deliver_notices(FailingStream(fail_flush=fail_flush))
        assert rows(owner, "notices")[0]["delivered"] is None
        assert "no skill change" in rendered(owner)
    finally:
        owner.close()


def test_print_before_ack_crash_can_repeat_and_success_then_suppresses(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    try:
        complete(owner, roster, lambda _: {"action": "NONE"})
        first = StringIO()
        with patch.object(owner, "_ack_delivered", return_value=False):
            assert owner.deliver_notices(first) == 0
        assert "no skill change" in first.getvalue()
        assert rows(owner, "notices")[0]["delivered"] is None
        second = StringIO()
        assert owner.deliver_notices(second) == 2
        assert second.getvalue() == first.getvalue()
        assert owner.deliver_notices(StringIO()) == 0
    finally:
        owner.close()


def test_disconnect_reconnect_preserves_pending_and_source_cleanup(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    complete(owner, roster, lambda _: create_proposal("after_reconnect"),
             session="source-session", task="source-task")
    assert not rows(owner, "evidence")
    job_id = rows(owner, "notices")[0]["job_id"]
    owner.close()
    owner = owner_for(roster, background=False)
    try:
        text = rendered(owner)
        assert "after_reconnect" in text and "source-session" in text and "source-task" in text
        assert rows(owner, "notices")[0]["job_id"] == job_id
    finally:
        owner.close()


def test_detailed_cap_compacts_explicit_outcome_summary_without_blocking_learning(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    total = NOTICE_DETAIL_COUNT + 6
    try:
        for index in range(total):
            if index == 0:
                def provider(_):
                    raise TimeoutError("bounded failure")
            elif index % 2:
                provider = lambda _: {"action": "NONE"}
            else:
                provider = lambda _, i=index: create_proposal(f"bounded_{i}")
            complete(owner, roster, provider, task=f"capacity-{index}")
        pending = [row for row in rows(owner, "notices") if row["delivered"] is None]
        assert len([row for row in pending if row["kind"] in ("START", "DETAIL")]) == NOTICE_DETAIL_COUNT
        summaries = [row for row in pending if row["kind"] == "SUMMARY"]
        assert len(summaries) == 1
        summary = json.loads(summaries[0]["payload_json"])
        assert summary["count"] == 36
        assert summary["starts"] == 18
        assert summary["outcomes"] == {"APPLIED": 8, "FAILED": 1, "NONE": 9}
        assert len(rows(owner, "jobs")) <= 32
        assert "bounded_28" in owner.store.list_skills()

        exact_summary = owner._render_notice(summaries[0])
        assert "START=18" in exact_summary and "APPLIED=8" in exact_summary
        assert "FAILED=1" in exact_summary and "NONE=9" in exact_summary
        assert "review events" in exact_summary and "not distinct jobs or completed outcomes" in exact_summary

        # Exercise the bounded counter's explicit saturation path without a
        # million-event test loop. The row is resealed with this temporary
        # profile's real authority, so this is valid stored state, not tampering.
        auth = read_auth(roster, owner.entry)
        with sqlite3.connect(owner.entry.mailbox) as conn:
            conn.row_factory = sqlite3.Row
            row = dict(conn.execute("SELECT * FROM notices WHERE kind='SUMMARY' AND delivered IS NULL").fetchone())
            saturated = json.loads(row["payload_json"])
            saturated["count"] = NOTICE_SUMMARY_COUNT_LIMIT
            saturated["outcomes"]["APPLIED"] = NOTICE_SUMMARY_COUNT_LIMIT
            row["payload_json"] = packed(saturated)
            row["notice_seal"] = notice_seal(auth, row)
            conn.execute("UPDATE notices SET payload_json=?,notice_seal=? WHERE seq=?",
                         (row["payload_json"], row["notice_seal"], row["seq"]))
        complete(owner, roster, lambda _: {"action": "NONE"}, task="capacity-saturation")

        text = rendered(owner)
        assert "at least 1000000 older review events were compacted" in text
        assert "APPLIED>=1000000" in text and "FAILED>=1" in text and "NONE>=9" in text
        assert "not a named success notice" in text
    finally:
        owner.close()


def test_disable_orders_after_inflight_delivery_and_prevents_later_write(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    entered, release = threading.Event(), threading.Event()

    class BlockingStream(StringIO):
        def write(self, value):
            entered.set()
            release.wait(4)
            return super().write(value)

    try:
        complete(owner, roster, lambda _: {"action": "NONE"})
        worker = threading.Thread(target=owner.deliver_notices, args=(BlockingStream(),))
        worker.start()
        assert entered.wait(2)
        disabled = threading.Thread(target=owner.set_enabled, args=(False,))
        disabled.start()
        time.sleep(.05)
        assert disabled.is_alive()
        release.set()
        worker.join(3)
        disabled.join(3)
        assert not worker.is_alive() and not disabled.is_alive()
        before = Path(owner.entry.mailbox).read_bytes()
        assert owner.deliver_notices(StringIO()) == 0
        assert Path(owner.entry.mailbox).read_bytes() == before
    finally:
        release.set()
        owner.close()


def test_episode_notice_keeps_revision_scope_after_source_cleanup_and_never_enters_model_input(tmp_path):
    from test_skill_episodes import prepare_active_episode
    from test_skill_review_integration import make_agent

    instance, roster = make_agent(tmp_path)
    try:
        prepare_active_episode(instance)
        messages_before = json.dumps(instance.messages, sort_keys=True)
        assert ReviewService(roster, provider=lambda _: create_proposal("episode_skill")).once() == 1
        instance.skill_review_owner.pump()
        assert not [row for row in rows(instance.skill_review_owner, "episode_sources") if row["job_id"]]
        notice = rows(instance.skill_review_owner, "notices")[-1]
        payload = json.loads(notice["payload_json"])
        assert payload["source"]["scope"] == "bound episode source set"
        assert payload["source"]["revision"] >= 2
        assert payload["source"]["source_count"] == 2
        assert len(payload["source"]["task_ids"]) == 2
        text = rendered(instance.skill_review_owner)
        assert "bound episode source set" in text and "bound source record(s)" in text
        assert "episode_skill" in text
        assert json.dumps(instance.messages, sort_keys=True) == messages_before
        assert "episode_skill" not in json.dumps(instance.messages)
    finally:
        instance.shutdown_skill_reviews()


def test_superseded_revision_is_quiet_and_fresh_revision_gets_its_own_notice(tmp_path):
    from test_skill_episodes import prepare_active_episode, run_sql, sql_result
    from test_skill_review_integration import make_agent

    instance, roster = make_agent(tmp_path)
    try:
        prepare_active_episode(instance)
        assert ReviewService(roster, provider=lambda _: create_proposal("obsolete_notice")).once() == 1
        run_sql(instance, "No, that procedure is wrong; use QUALIFY instead",
                "SELECT * FROM sales QUALIFY ROW_NUMBER() OVER (ORDER BY sale_date DESC)=1", sql_result())
        instance.skill_review_owner.pump()
        assert rows(instance.skill_review_owner, "jobs")[0]["status"] == "SUPERSEDED"
        superseded_notices = rows(instance.skill_review_owner, "notices")
        assert [row["kind"] for row in superseded_notices] == ["START"]
        assert "obsolete_notice" not in rendered(instance.skill_review_owner)

        instance.skill_review_owner.pump()
        assert ReviewService(roster, provider=lambda _: {"action": "NONE"}).once() == 1
        instance.skill_review_owner.pump()
        pending = rows(instance.skill_review_owner, "notices")
        assert [row["kind"] for row in pending] == ["START", "START", "DETAIL"]
        payload = json.loads(pending[-1]["payload_json"])
        assert payload["outcome"] == "NONE"
        assert payload["source"]["revision"] >= 3
        assert "obsolete_notice" not in rendered(instance.skill_review_owner)
    finally:
        instance.shutdown_skill_reviews()


def test_legacy_multi_session_batch_attribution_is_selected_sources_not_current_question(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False, batch_count=4)
    try:
        assert owner.enqueue(evidence(session="old-a", task="old-task-a")).status == "ACCEPTED"
        assert owner.enqueue(evidence(session="old-b", task="old-task-b")).status == "ACCEPTED"
        owner.pump()
        assert ReviewService(roster, provider=lambda _: {"action": "NONE"}).once() == 1
        owner.pump()
        payload = json.loads(rows(owner, "notices")[0]["payload_json"])
        assert payload["source"]["scope"] == "selected legacy batch"
        assert payload["source"]["session_count"] == 2
        assert payload["source"]["task_count"] == 2
        text = rendered(owner)
        assert "old-a" in text and "old-b" in text
        assert "current" not in text.lower()
    finally:
        owner.close()


def test_start_notice_accepts_only_legacy_version_one():
    payload = {
        "version": 1, "kind": "START", "started_at": 1.0,
        "source": {
            "scope": "bound episode source set", "source_count": 1, "session_count": 1,
            "task_count": 1, "session_ids": ["session"], "task_ids": ["task"],
            "source_ids": ["source"], "identifiers_truncated": False,
            "first_created": 1.0, "last_created": 1.0, "episode_id": "episode", "revision": 1,
        },
    }
    assert Owner._valid_notice_payload(payload, "START")
    assert not Owner._valid_notice_payload({**payload, "version": 2}, "START")


@pytest.mark.parametrize("reason,unit", [
    ("episode_sources", "sources"),
    ("carry_anchor_bytes", "bytes"),
    ("missing_context", "bytes"),
    ("selected_messages", "messages"),
    ("selected_view_bytes", "bytes"),
    ("prepared_bytes", "bytes"),
])
def test_budget_refusal_serializer_allowlists_reason_unit_and_exact_integers(reason, unit):
    refusal = BudgetRefusal(reason, 12, 10)
    diagnostic = {
        "reason": reason,
        "unit": unit,
        "observed": 12,
        "limit": 10,
        "provider_requests": 0,
    }
    assert refusal.diagnostic() == diagnostic
    detail = refusal.detail(profile_id="safe-profile", job_id="1" * 32)
    assert f"reason={reason}" in detail
    assert f"observed_{unit}=12 limit_{unit}=10" in detail
    assert "profile=safe-profile" in detail and f"job={'1' * 32}" in detail
    assert "zero provider requests" in detail
    payload = {
        "version": 2, "kind": "DETAIL", "outcome": "BUDGET_REFUSED", "action": "", "name": "",
        "reason": "the bounded review request was refused before a provider call", "budget": diagnostic,
        "source": {
            "scope": "bound episode source set", "source_count": 1, "session_count": 1,
            "task_count": 1, "session_ids": ["session"], "task_ids": ["task"],
            "source_ids": ["source"], "identifiers_truncated": False,
            "first_created": 1.0, "last_created": 1.0, "episode_id": "episode", "revision": 1,
        },
    }
    assert Owner._valid_notice_payload(payload, "DETAIL")
    rendered_detail = Owner._render_notice({
        "kind": "DETAIL", "payload_json": packed(payload), "job_id": "1" * 32,
    })
    assert f"reason={reason}; observed=12 {unit}; limit=10 {unit}; provider requests=0" in rendered_detail


@pytest.mark.parametrize("reason", [[], {}])
def test_unhashable_budget_reason_safely_degrades_to_generic(reason):
    diagnostic = {
        "reason": reason, "unit": "bytes", "observed": 12, "limit": 10,
        "provider_requests": 0,
    }
    assert not valid_budget_diagnostic(diagnostic)

    refusal = BudgetRefusal(reason, 12, 10)
    assert refusal.diagnostic() is None
    detail = refusal.detail(profile_id="safe-profile", job_id="1" * 32)
    assert "reason=budget_limit" in detail
    assert "observed_" not in detail and "limit_" not in detail
    assert "profile=safe-profile" in detail and f"job={'1' * 32}" in detail
    assert "zero provider requests" in detail


@pytest.mark.parametrize("poisoned", [
    {"reason": [], "unit": "bytes", "observed": 12, "limit": 10,
     "provider_requests": 0},
    {"reason": {}, "unit": "bytes", "observed": 12, "limit": 10,
     "provider_requests": 0},
    {"reason": "unknown\nPRIVATE", "unit": "bytes", "observed": 12, "limit": 10,
     "provider_requests": 0},
    {"reason": "selected_view_bytes", "unit": "bytes\x1bPRIVATE", "observed": 12,
     "limit": 10, "provider_requests": 0},
    {"reason": "selected_view_bytes", "unit": "bytes", "observed": True, "limit": 10,
     "provider_requests": 0},
    {"reason": "selected_view_bytes", "unit": "bytes", "observed": "12", "limit": 10,
     "provider_requests": 0},
    {"reason": "selected_view_bytes", "unit": "bytes", "observed": -1, "limit": 10,
     "provider_requests": 0},
    {"reason": "selected_view_bytes", "unit": "bytes", "observed": 1000001,
     "limit": 10, "provider_requests": 0},
    {"reason": "selected_view_bytes", "unit": "bytes", "observed": 12, "limit": 10,
     "provider_requests": True},
    {"reason": "selected_view_bytes", "unit": "bytes", "observed": 12, "limit": 10,
     "provider_requests": 0, "PRIVATE": "task text"},
])
def test_poisoned_budget_result_falls_back_to_valid_generic_notice(tmp_path, poisoned):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    try:
        assert owner.enqueue(evidence(task="safe-budget-source")).status == "ACCEPTED"
        owner.pump()
        job = rows(owner, "jobs")[-1]
        with sqlite3.connect(owner.entry.mailbox) as conn:
            conn.row_factory = sqlite3.Row
            stored_job = dict(conn.execute("SELECT * FROM jobs WHERE job_id=?", (job["job_id"],)).fetchone())
            host = json.loads(stored_job["host_json"])
            payload = owner._notice_payload(
                conn, stored_job, SimpleNamespace(status="BUDGET_REFUSED"),
                {"status": "BUDGET_REFUSED", "detail": "PRIVATE task/path/sql", "budget": poisoned}, host,
            )
        assert payload["version"] == 1
        assert set(payload) == {"version", "kind", "outcome", "action", "name", "reason", "source"}
        assert payload["reason"] == "the bounded review request was refused before a provider call"
        assert "PRIVATE" not in json.dumps(payload)
        assert Owner._valid_notice_payload(payload, "DETAIL")
    finally:
        owner.close()


def test_legacy_generic_budget_detail_remains_deliverable(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    try:
        assert owner.enqueue(evidence(task="legacy-budget-source")).status == "ACCEPTED"
        owner.pump()
        auth = read_auth(roster, owner.entry)
        with sqlite3.connect(owner.entry.mailbox) as conn:
            conn.row_factory = sqlite3.Row
            job = dict(conn.execute("SELECT * FROM jobs ORDER BY seq DESC LIMIT 1").fetchone())
            host = json.loads(job["host_json"])
            owner._store_notice(
                conn, auth, job, SimpleNamespace(status="BUDGET_REFUSED"),
                {"status": "BUDGET_REFUSED", "detail": "legacy unstructured detail"}, host,
            )
        owner.close()
        owner = owner_for(roster, background=False)
        output = StringIO()
        assert owner.deliver_notices(output) == 1
        assert "bounded review request was refused before a provider call" in output.getvalue()
        assert "reason=" not in output.getvalue() and "legacy unstructured detail" not in output.getvalue()
    finally:
        owner.close()


def test_actual_no_call_budget_refusal_has_specific_durable_diagnostic(tmp_path, capsys):
    from test_skill_evidence_budget import eligible_evidence

    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    calls = []
    try:
        oversized = eligible_evidence("budget-source", padding="x" * (200 * 1024))
        assert owner.capture_turn(oversized).status == "ELIGIBLE"
        owner.flush_session(oversized.session_id)
        owner.pump()
        assert ReviewService(roster, provider=lambda request: calls.append(request) or {"action": "NONE"}).once() == 1
        service_stderr = capsys.readouterr().err
        result_job = rows(owner, "jobs")[-1]
        result = json.loads(result_job["result_json"])
        budget = result["budget"]
        assert budget["reason"] == "selected_view_bytes"
        assert budget["unit"] == "bytes"
        assert type(budget["observed"]) is int and budget["observed"] > budget["limit"]
        assert budget["limit"] == 64 * 1024 and budget["provider_requests"] == 0
        assert "Skill review: historical stage=service.preparation" in service_stderr
        assert "reason=selected_view_bytes" in service_stderr
        assert f"observed_bytes={budget['observed']} limit_bytes={budget['limit']}" in service_stderr
        assert f"profile={roster.profiles[0].profile_id}" in service_stderr
        assert f"job={result_job['job_id']}" in service_stderr
        assert "zero provider requests" in service_stderr
        owner.pump()
        assert calls == []
        final_job = rows(owner, "jobs")[-1]
        assert final_job["status"] == "BUDGET_REFUSED"
        assert f"observed_bytes={budget['observed']} limit_bytes={budget['limit']}" in final_job["detail"]
        assert rows(owner, "review_fingerprints") == []
        assert not [source for source in rows(owner, "episode_sources") if source["job_id"]]
        notice = json.loads(rows(owner, "notices")[-1]["payload_json"])
        assert notice["version"] == 2 and notice["budget"] == budget
        owner.close()
        owner = owner_for(roster, background=False)
        text = rendered(owner)
        assert text.index("Review started") < text.index(
            "bounded review request was refused before a provider call")
        assert "bounded review request was refused before a provider call" in text
        assert "reason=selected_view_bytes" in text
        assert f"observed={budget['observed']} bytes; limit={budget['limit']} bytes" in text
        assert "provider requests=0" in text
        assert "no skill change" not in text
        assert "Created skill" not in text and "Updated skill" not in text
    finally:
        owner.close()


def test_one_success_nonroutine_query_reaches_none_notice_only_after_final_result(tmp_path):
    import agent as agent_module
    from test_skill_review_integration import (
        answer_message,
        make_agent,
        sql_tool_message,
        verified_result,
    )

    instance, roster = make_agent(tmp_path)
    owner = instance.skill_review_owner
    requests = []
    sql = "SELECT customer_id, SUM(amount) OVER (PARTITION BY customer_id) FROM sales"
    try:
        with patch.object(instance, "step", side_effect=[
                sql_tool_message(sql), answer_message("Completed from one query.")]), \
                patch.object(instance, "manage_context"), \
                patch.object(agent_module.registry, "execute", return_value=verified_result()):
            assert instance.run("Calculate reusable customer sales totals") == \
                "Completed from one query."

        assert instance.last_skill_admission.status == "ELIGIBLE"
        assert rows(owner, "notices") == []
        foreground_messages = json.dumps(instance.messages, sort_keys=True)

        owner.flush_session(instance.session_id)
        owner.pump()
        assert rows(owner, "jobs")[-1]["status"] == "PREPARED"
        assert rows(owner, "notices") == []

        assert ReviewService(
            roster, provider=lambda request: requests.append(request) or {"action": "NONE"}
        ).once() == 1
        assert rows(owner, "jobs")[-1]["status"] == "RESULT"
        assert rows(owner, "notices") == []

        owner.pump()
        assert rows(owner, "jobs")[-1]["status"] == "NONE"
        output = StringIO()
        assert instance.deliver_skill_review_notices(output) == 2
        assert "Review completed with no skill change" in output.getvalue()
        assert len(requests) == 1 and "business_result_omitted" in requests[0]
        assert json.dumps(instance.messages, sort_keys=True) == foreground_messages
        assert "[Skill Review]" not in foreground_messages
    finally:
        instance.shutdown_skill_reviews()


def test_real_writer_export_reaches_success_notice_without_csv_contents(tmp_path):
    import agent as agent_module
    import db_tools
    from test_csv_export_review_capture import native_export_message
    from test_skill_review_integration import answer_message, make_agent

    sql = (
        "SELECT a.customer_id, a.amount FROM synthetic a "
        "JOIN synthetic b ON b.customer_id = a.customer_id ORDER BY a.customer_id"
    )
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute("CREATE TABLE synthetic(customer_id INTEGER, amount TEXT)")
        connection.executemany("INSERT INTO synthetic VALUES (?, ?)", [
            (1, "PRIVATE-CSV-ALPHA"), (2, "PRIVATE-CSV-BETA"),
        ])
        raw_manifest = db_tools._stream_csv(
            connection.execute(sql), str(tmp_path / "complete.csv"), 2,
            "warehouse", "Impala", sql, str(tmp_path), False,
        )
    finally:
        connection.close()

    instance, roster = make_agent(tmp_path)
    owner = instance.skill_review_owner
    requests = []
    try:
        with patch.object(instance, "step", side_effect=[
                native_export_message("export_impala_csv", sql),
                answer_message("Complete CSV exported."),
        ]), patch.object(instance, "manage_context"), \
                patch.object(agent_module.registry, "execute", return_value=raw_manifest):
            assert instance.run("Export the reusable joined report") == "Complete CSV exported."

        assert instance.last_skill_admission.status == "ELIGIBLE"
        source = rows(owner, "episode_sources")[0]
        assert "exported_result_omitted" in source["messages_json"]
        assert "PRIVATE-CSV-ALPHA" not in source["messages_json"]
        assert "PRIVATE-CSV-BETA" not in source["messages_json"]
        assert rows(owner, "notices") == []
        foreground_messages = json.dumps(instance.messages, sort_keys=True)

        owner.flush_session(instance.session_id)
        owner.pump()
        assert rows(owner, "notices") == []
        assert ReviewService(
            roster,
            provider=lambda request: requests.append(request) or create_proposal("export_notice"),
        ).once() == 1
        assert rows(owner, "jobs")[-1]["status"] == "RESULT"
        assert rows(owner, "notices") == []

        owner.pump()
        assert rows(owner, "jobs")[-1]["status"] == "APPLIED"
        output = StringIO()
        assert instance.deliver_skill_review_notices(output) == 2
        assert "Created skill 'export_notice'" in output.getvalue()
        assert len(requests) == 1 and "exported_result_omitted" in requests[0]
        assert "PRIVATE-CSV-ALPHA" not in requests[0]
        assert "PRIVATE-CSV-BETA" not in requests[0]
        assert json.dumps(instance.messages, sort_keys=True) == foreground_messages
        assert "[Skill Review]" not in foreground_messages
    finally:
        instance.shutdown_skill_reviews()


def test_rich_and_untrusted_diagnostics_end_in_private_generic_failure_notices(tmp_path):
    from test_skill_review_diagnostics import fake_sdk_response

    private = "PRIVATE-PROVIDER-BODY SELECT * FROM secret_table"
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    try:
        assert owner.enqueue(evidence(task="trusted-diagnostic")).status == "ACCEPTED"
        owner.pump()
        client = MagicMock()
        client.with_options.return_value = client
        client.chat.completions.create.return_value = fake_sdk_response(
            private, "length",
            SimpleNamespace(prompt_tokens=123, completion_tokens=17, total_tokens=140),
        )
        with patch("openai.OpenAI", return_value=client):
            assert ReviewService(roster, timeout=1.25, output_tokens=512).once() == 1
        assert rows(owner, "jobs")[-1]["status"] == "RESULT"
        assert rows(owner, "notices") == []

        owner.pump()
        trusted_job = rows(owner, "jobs")[-1]
        assert trusted_job["status"] == "INVALID"
        for token in (
            "reason=non_stop_finish", "request_attempted=true",
            "response_received=true", "finish_reason=length", "output_tokens=512",
            "usage_prompt_tokens=123", "usage_completion_tokens=17",
            "usage_total_tokens=140",
        ):
            assert token in trusted_job["detail"]
        trusted_notice = json.dumps(rows(owner, "notices"), sort_keys=True)
        trusted_output = rendered(owner)
        assert "No skill was published: the review result or publication request was invalid" in trusted_output
        assert "provider." not in trusted_notice + trusted_output
        assert private not in trusted_notice + trusted_output

        assert owner.enqueue(evidence(task="untrusted-diagnostic")).status == "ACCEPTED"
        owner.pump()
        forged = ValueError(private)
        forged.reason = "malformed_json"
        forged.metadata = {
            "request_attempted": True,
            "response_received": True,
            "finish_reason": "length",
            "output_tokens": 512,
        }
        assert ReviewService(
            roster, provider=lambda _: (_ for _ in ()).throw(forged)
        ).once() == 1
        owner.pump()
        untrusted_job = rows(owner, "jobs")[-1]
        assert untrusted_job["status"] == "INVALID"
        assert "reason=" not in untrusted_job["detail"]
        assert "request_attempted=" not in untrusted_job["detail"]
        untrusted_notice = json.dumps(
            [row for row in rows(owner, "notices") if row["delivered"] is None],
            sort_keys=True,
        )
        untrusted_output = rendered(owner)
        assert "No skill was published: the review result or publication request was invalid" in untrusted_output
        assert "provider." not in untrusted_notice + untrusted_output
        assert "request_attempted" not in untrusted_notice + untrusted_output
        assert private not in untrusted_notice + untrusted_output
    finally:
        owner.close()


def test_additive_schema_migration_preserves_jobs_and_does_not_invent_historical_backfill(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    complete(owner, roster, lambda _: {"action": "NONE"}, task="already-terminal")
    assert rows(owner, "jobs")[-1]["status"] == "NONE"
    assert owner.enqueue(evidence(task="prepared-before-migration")).status == "ACCEPTED"
    owner.pump()
    assert rows(owner, "jobs")[-1]["status"] == "PREPARED"
    owner.close()
    with sqlite3.connect(roster.profiles[0].mailbox) as conn:
        conn.execute("DROP TABLE notices")
        conn.execute("DROP TABLE episode_sources")
        conn.execute("DROP TABLE episodes")
        conn.execute("DROP TABLE review_fingerprints")
    owner = owner_for(roster, background=False)
    try:
        assert [row["status"] for row in rows(owner, "jobs")] == ["NONE", "PREPARED"]
        assert rows(owner, "notices") == []
        assert owner.deliver_notices(StringIO()) == 0

        assert ReviewService(roster, provider=lambda _: {"action": "NONE"}).once() == 1
        owner.pump()
        assert [row["kind"] for row in rows(owner, "notices")] == ["START", "DETAIL"]
    finally:
        owner.close()


@pytest.mark.parametrize("settings", [
    dict(auto_learn_skills=False),
    dict(auto_learn_skills=True, read_only=True),
    dict(auto_learn_skills=True, stateless=True),
    dict(auto_learn_skills=True, enable_skills=False),
])
def test_disabled_readonly_stateless_and_no_skills_startup_create_no_notice_state(tmp_path, settings):
    import agent as agent_module

    roster = setup_roster(tmp_path)
    previous = agent_module.skill_store.storage_dir
    instance = None
    try:
        agent_module.skill_store.storage_dir = str(tmp_path / "profile-0" / "skills")
        with patch.object(agent_module, "ACTIVE_HISTORY_DB", str(tmp_path / "history.db")):
            instance = agent_module.HermesCodingAgent(
                model=roster.model, base_url=roster.base_url, enable_memory=False,
                review_roster=roster, review_profile="user-0", **settings,
            )
        assert instance.deliver_skill_review_notices(StringIO()) == 0
        assert not Path(roster.profiles[0].mailbox).exists()
    finally:
        if instance:
            instance.shutdown_skill_reviews()
        agent_module.skill_store.storage_dir = previous


def test_reenable_invalidates_old_generation_notice_without_replay(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    try:
        complete(owner, roster, lambda _: create_proposal("old_generation"))
        assert rows(owner, "notices")[0]["delivered"] is None
        owner.set_enabled(False)
        owner.set_enabled(True)
        assert rows(owner, "notices") == []
        assert owner.deliver_notices(StringIO()) == 0
    finally:
        owner.close()


def test_successful_output_detail_is_not_reintroduced_by_concurrent_compaction(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    try:
        for index in range(NOTICE_DETAIL_COUNT // 2):
            complete(owner, roster, lambda _: {"action": "NONE"}, task=f"shown-{index}")

        class ArrivalDuringOutput(StringIO):
            def write(self, value):
                written = super().write(value)
                complete(owner, roster, lambda _: {"action": "NONE"}, task="new-after-snapshot")
                return written

        first = ArrivalDuringOutput()
        assert owner.deliver_notices(first) == NOTICE_DETAIL_COUNT
        second = StringIO()
        assert owner.deliver_notices(second) == 2
        assert "new-after-snapshot" in second.getvalue()
        assert "compacted" not in second.getvalue()
    finally:
        owner.close()


def test_successful_output_summary_is_not_mutated_by_concurrent_compaction(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    try:
        for index in range(NOTICE_DETAIL_COUNT // 2 + 1):
            complete(owner, roster, lambda _: {"action": "NONE"}, task=f"old-{index}")

        class BacklogDuringOutput(StringIO):
            def write(self, value):
                written = super().write(value)
                for index in range(NOTICE_DETAIL_COUNT // 2 + 1):
                    complete(owner, roster, lambda _: {"action": "NONE"}, task=f"new-{index}")
                return written

        first = BacklogDuringOutput()
        assert owner.deliver_notices(first) == NOTICE_DETAIL_COUNT + 1
        assert "2 older review events were compacted" in first.getvalue()

        second = StringIO()
        assert owner.deliver_notices(second) == NOTICE_DETAIL_COUNT + 1
        output = second.getvalue()
        assert "2 older review events were compacted" in output
        assert "28 older review events were compacted" not in output
        assert "new-12" in output
        assert "old-" not in output
    finally:
        owner.close()
