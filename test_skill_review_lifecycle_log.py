"""Caller-visible terminal and private JSONL review lifecycle contracts."""
from io import StringIO
import json
import os
from pathlib import Path
import sqlite3
import pytest
from unittest.mock import patch

from skill_review import NOTICE_DETAIL_COUNT, Owner, ReviewService
from test_async_skill_review import create_proposal, evidence, owner_for, rows, setup_roster


def events(owner):
    path = Path(owner.entry.mailbox).parent / "logs" / "skill_reviews.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def complete(owner, roster, provider, task):
    assert owner.enqueue(evidence(task=task)).status == "ACCEPTED"
    owner.pump()
    assert ReviewService(roster, provider=provider).once() == 1
    owner.pump()


def test_real_terminal_and_private_log_cover_requested_started_and_final(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    try:
        complete(owner, roster, lambda _: create_proposal("safe_name"), "private-business-row")
        output = StringIO()
        assert owner.deliver_notices(output) == 3
        text = output.getvalue()
        assert text.index("Review requested") < text.index("Review started") < text.index("Created skill")
        job_id = rows(owner, "jobs")[-1]["job_id"]
        assert text.count(job_id) == 3

        saved = events(owner)
        assert [item["event"] for item in saved] == ["REQUESTED", "STARTED", "DECISION"]
        assert [item["event_id"] for item in saved] == [
            f"{job_id}:requested", f"{job_id}:started", f"{job_id}:decision"]
        assert all(item["review_id"] == job_id and item["timestamp"].endswith("Z") for item in saved)
        assert saved[0]["status"] == "REQUESTED" and saved[0]["skill_name"] == ""
        assert saved[1]["status"] == "STARTED" and saved[1]["skill_name"] == ""
        assert saved[2]["status"] == "APPLIED" and saved[2]["skill_name"] == "safe_name"
        assert saved[2]["reason"] == "explanation unavailable"
        assert "private-business-row" not in json.dumps(saved)
    finally:
        owner.close()


def test_quiet_suppresses_terminal_only_and_none_has_unavailable_explanation(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False, quiet=True)
    try:
        complete(owner, roster, lambda _: {"action": "NONE"}, "quiet-task")
        output = StringIO()
        assert owner.deliver_notices(output) == 0
        assert output.getvalue() == ""
        saved = events(owner)
        assert [item["event"] for item in saved] == ["REQUESTED", "STARTED", "DECISION"]
        assert saved[-1]["status"] == "NONE"
        assert saved[-1]["reason"] == "explanation unavailable"
    finally:
        owner.close()


def test_failure_is_safe_and_profiles_are_isolated(tmp_path):
    roster = setup_roster(tmp_path, count=2)
    alice = owner_for(roster, 0, background=False)
    bob = owner_for(roster, 1, background=False)
    try:
        def failed(_):
            raise TimeoutError("raw provider body and secret business row")

        complete(alice, roster, failed, "alice-sensitive-task")
        complete(bob, roster, lambda _: {"action": "NONE"}, "bob-sensitive-task")
        alice_events, bob_events = events(alice), events(bob)
        assert len(alice_events) == len(bob_events) == 3
        assert alice_events[-1]["status"] == "FAILED"
        assert alice_events[-1]["reason"] == "the review or publication attempt failed"
        assert "raw provider body" not in json.dumps(alice_events)
        assert {item["review_id"] for item in alice_events}.isdisjoint(
            {item["review_id"] for item in bob_events})
        assert "bob-sensitive-task" not in json.dumps(alice_events)
        assert "alice-sensitive-task" not in json.dumps(bob_events)
    finally:
        alice.close()
        bob.close()


def test_update_log_uses_host_target_name_and_unavailable_explanation(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    try:
        owner.store.save_skill("existing", "Workflow", "Old steps.")

        def update(request):
            target = json.loads(request)["catalog"]["targets"][0]
            return dict(action="UPDATE", target_id=target["target_id"], description="Workflow",
                        instructions="Replacement private instructions.", complete=True)

        complete(owner, roster, update, "update-private-task")
        decision = events(owner)[-1]
        assert decision["status"] == "APPLIED"
        assert decision["action"] == "UPDATE"
        assert decision["skill_name"] == "existing"
        assert decision["reason"] == "explanation unavailable"
        assert "Replacement private instructions" not in json.dumps(events(owner))
    finally:
        owner.close()


def test_create_log_accepts_full_host_rendered_name_contract(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    name = "x" * 124 + "\u200b"
    try:
        complete(owner, roster, lambda _: create_proposal(name), "long-create-name")
        decision = events(owner)[-1]
        assert decision["event"] == "DECISION"
        assert decision["status"] == "APPLIED"
        assert decision["action"] == "CREATE"
        assert decision["skill_name"] == Owner._notice_name(name)
        assert all(row["appended"] != -1 for row in rows(owner, "review_log_events"))
    finally:
        owner.close()


def test_update_log_accepts_long_real_target_with_non_bmp_json_expansion(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    name = "x" * 123 + "\u200b\U0001f600"
    try:
        assert "successfully saved" in owner.store.save_skill(name, "Workflow", "Old steps.")

        def update(request):
            target = json.loads(request)["catalog"]["targets"][0]
            return dict(action="UPDATE", target_id=target["target_id"], description="Workflow",
                        instructions="Replacement steps.", complete=True)

        complete(owner, roster, update, "long-update-name")
        decision = events(owner)[-1]
        assert decision["event"] == "DECISION"
        assert decision["status"] == "APPLIED"
        assert decision["action"] == "UPDATE"
        assert decision["skill_name"] == Owner._notice_name(name)
        assert len(decision["skill_name"]) > 128
        assert all(row["appended"] != -1 for row in rows(owner, "review_log_events"))
    finally:
        owner.close()


def test_create_and_none_log_explicit_actions(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    try:
        complete(owner, roster, lambda _: create_proposal("made"), "create-action")
        complete(owner, roster, lambda _: {"action": "NONE"}, "none-action")
        decisions = [item for item in events(owner) if item["event"] == "DECISION"]
        assert [(item["status"], item["action"]) for item in decisions] == [
            ("APPLIED", "CREATE"), ("NONE", "NONE")]
    finally:
        owner.close()


def test_log_append_failure_retains_pending_then_reconnect_recovers_without_duplicates(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    try:
        assert owner.enqueue(evidence(task="retry-task")).status == "ACCEPTED"
        with patch.object(Owner, "_append_review_log", side_effect=OSError("private raw failure")):
            owner.pump()
        assert events(owner) == []
        assert any(row["appended"] is None for row in rows(owner, "review_log_events"))
        assert "private raw failure" not in owner.last_error
        owner.close()

        recovered = owner_for(roster, background=False)
        try:
            recovered.pump()
            first = events(recovered)
            assert [item["event"] for item in first] == ["REQUESTED"]
            recovered.pump()
            assert events(recovered) == first
            assert all(row["appended"] is not None for row in rows(recovered, "review_log_events"))
        finally:
            recovered.close()
    finally:
        if not owner.closed:
            owner.close()


def test_partial_jsonl_fails_closed_and_does_not_acknowledge(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    try:
        path = owner.review_log_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"partial":', encoding="utf-8")
        assert owner.enqueue(evidence(task="partial-retry")).status == "ACCEPTED"
        owner.pump()
        pending = rows(owner, "review_log_events")
        assert pending and all(row["appended"] is None for row in pending)
        assert path.read_text(encoding="utf-8") == '{"partial":'
        assert "owner.review_log" in owner.last_error
    finally:
        owner.close()


def test_same_event_id_with_different_record_never_acknowledges(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    try:
        assert owner.enqueue(evidence(task="conflict-retry")).status == "ACCEPTED"
        with patch.object(Owner, "_append_review_log", side_effect=OSError("blocked")):
            owner.pump()
        pending = rows(owner, "review_log_events")[0]
        forged = json.loads(pending["payload_json"])
        forged["reason"] = "forged"
        path = owner.review_log_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(forged) + "\n", encoding="utf-8")
        owner.pump()
        assert rows(owner, "review_log_events")[0]["appended"] is None
        assert len(path.read_text(encoding="utf-8").splitlines()) == 1
    finally:
        owner.close()


def test_fsync_failure_reconciles_exact_record_on_retry(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    try:
        assert owner.enqueue(evidence(task="fsync-retry")).status == "ACCEPTED"
        with patch.object(os, "fsync", side_effect=OSError("fsync blocked")):
            owner.pump()
        assert len(events(owner)) == 1
        assert rows(owner, "review_log_events")[0]["appended"] is None
        owner.pump()
        assert len(events(owner)) == 1
    finally:
        owner.close()


def test_actual_log_open_failure_retains_pending(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    try:
        assert owner.enqueue(evidence(task="open-failure")).status == "ACCEPTED"
        with patch.object(Owner, "_append_review_log", side_effect=OSError("prepare retry")):
            owner.pump()
        pending = rows(owner, "review_log_events")
        with patch.object(Path, "open", side_effect=OSError("write blocked")):
            with pytest.raises(OSError):
                Owner._append_review_log(owner.review_log_path, pending)
        assert rows(owner, "review_log_events")[0]["appended"] is None
        assert "write blocked" not in owner.last_error
    finally:
        owner.close()


def test_hardlinked_log_target_is_rejected_and_left_pending(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    try:
        complete(owner, roster, lambda _: {"action": "NONE"}, "initial")
        path = owner.review_log_path
        outside = tmp_path / "outside.jsonl"
        path.replace(outside)
        os.link(outside, path)
        assert owner.enqueue(evidence(task="hardlink-attempt")).status == "ACCEPTED"
        owner.pump()
        assert any(row["appended"] is None for row in rows(owner, "review_log_events"))
        assert outside.read_bytes() == path.read_bytes()
    finally:
        owner.close()


def test_dangling_symlink_log_target_is_rejected_before_append(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    outside = tmp_path / "outside.jsonl"
    try:
        assert owner.enqueue(evidence(task="dangling-link-attempt")).status == "ACCEPTED"
        path = owner.review_log_path
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.symlink_to(outside)
        except OSError as exc:
            pytest.skip(f"host does not permit symlink creation: {type(exc).__name__}")
        owner.pump()
        assert not outside.exists()
        assert path.is_symlink()
        assert any(row["appended"] is None for row in rows(owner, "review_log_events"))
        assert "owner.review_log" in owner.last_error
    finally:
        owner.close()


def test_existing_log_entry_detection_fails_closed_before_append(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    try:
        assert owner.enqueue(evidence(task="injected-link-boundary")).status == "ACCEPTED"
        with patch("skill_review.os.path.lexists", return_value=True), \
                patch.object(Owner, "_append_review_log") as append:
            owner.pump()
        append.assert_not_called()
        assert any(row["appended"] is None for row in rows(owner, "review_log_events"))
        assert "owner.review_log" in owner.last_error
    finally:
        owner.close()


def test_tampered_pending_payload_is_quarantined_without_file_io(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    try:
        assert owner.enqueue(evidence(task="tamper-outbox")).status == "ACCEPTED"
        with patch.object(Owner, "_append_review_log", side_effect=OSError("blocked")):
            owner.pump()
        with sqlite3.connect(owner.entry.mailbox) as conn:
            conn.execute("UPDATE review_log_events SET payload_json=? WHERE appended IS NULL",
                         ('{"raw_business_rows":"MUST_NOT_LOG"}',))
        with patch.object(Owner, "_append_review_log") as append:
            owner.pump()
            append.assert_not_called()
        assert not owner.review_log_path.exists()
        quarantined = rows(owner, "review_log_events")[0]
        assert quarantined["appended"] == -1
        assert quarantined["payload_json"] == "{}"
        assert "MUST_NOT_LOG" not in owner.last_error
    finally:
        owner.close()


def test_successful_outbox_rows_are_pruned_but_jsonl_is_complete(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    try:
        for index in range(50):
            complete(owner, roster, lambda _: {"action": "NONE"}, f"bounded-{index}")
        assert len(events(owner)) == 150
        assert len(rows(owner, "review_log_events")) <= 32
    finally:
        owner.close()


def test_complete_json_record_without_line_terminator_retains_next_event(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    try:
        complete(owner, roster, lambda _: {"action": "NONE"}, "first-review")
        original = owner.review_log_path.read_bytes().rstrip(b"\n")
        owner.review_log_path.write_bytes(original)
        assert owner.enqueue(evidence(task="next-review")).status == "ACCEPTED"
        owner.pump()
        assert owner.review_log_path.read_bytes() == original
        assert any(row["appended"] is None for row in rows(owner, "review_log_events"))
        assert "retained for retry" in owner.last_error
        # Once the incomplete physical line is repaired, retry publishes it.
        owner.review_log_path.write_bytes(original + b"\n")
        owner.pump()
        assert len(events(owner)) == 4
    finally:
        owner.close()


def test_terminal_compaction_never_erases_per_request_file_history(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    try:
        for index in range(NOTICE_DETAIL_COUNT + 5):
            complete(owner, roster, lambda _: {"action": "NONE"}, f"task-{index}")
        assert len([row for row in rows(owner, "notices") if row["delivered"] is None]) <= NOTICE_DETAIL_COUNT + 1
        saved = events(owner)
        assert len(saved) == 3 * (NOTICE_DETAIL_COUNT + 5)
        assert len({item["event_id"] for item in saved}) == len(saved)
    finally:
        owner.close()
