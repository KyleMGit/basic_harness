"""Private durable owner/service contracts; deterministic providers only."""
import json
from pathlib import Path
import sqlite3
import threading
import time
from unittest.mock import patch

import pytest

import skill_review as review
from skill_review import Evidence, Owner, Roster, ReviewService, load_roster
from skills import SkillStore


def wait_for(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError("condition did not become true")


def setup_roster(tmp_path, count=1):
    entries = []
    for i in range(count):
        root = tmp_path / f"profile-{i}"
        entries.append(dict(profile_id=f"user-{i}", mailbox=str(root / "skill_review.db"),
                            store_id=SkillStore(str(root / "skills")).store_id))
    path = tmp_path / "roster.json"
    path.write_text(json.dumps(dict(control_dir=str(tmp_path / "control"), model="test-model",
                                   base_url="http://127.0.0.1:1/v1", profiles=entries)))
    return load_roster(path)


def evidence(session="session", task="task", text="Perform reusable work"):
    item = Evidence(session, task)
    item.add({"role": "system", "content": "SYSTEM MUST NEVER PERSIST"})
    item.add({"role": "user", "content": text})
    item.add({"role": "assistant", "content": "Completed successfully."})
    return item.finish()


def owner_for(roster, index=0, **kwargs):
    entry = roster.profiles[index]
    return Owner(roster, entry.profile_id, SkillStore(str(Path(entry.mailbox).parent / "skills")), **kwargs)


def rows(owner, table):
    with sqlite3.connect(owner.entry.mailbox) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(f"SELECT * FROM {table}")]


def create_proposal(name="learned"):
    return dict(action="CREATE", name=name, description="Repeat the checked workflow",
                instructions="1. Run the verification.\n2. Inspect the output.", complete=True)


def profile_bytes(owner):
    root = Path(owner.entry.mailbox).parent
    paths = [root, *root.rglob("*")] if root.exists() else []
    return {str(p.relative_to(root)): (p.read_bytes() if p.is_file() else None, p.stat().st_mtime_ns)
            for p in paths}


def test_durable_evidence_is_redacted_bounded_oldest_first_and_session_separated(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False, max_records=3, batch_count=2)
    try:
        for i in range(3):
            assert owner.enqueue(evidence(f"session-{i}", f"task-{i}", "password=hidden-value token=secret-value")).status == "ACCEPTED"
        assert owner.enqueue(evidence(task="overflow")).status == "OVERFLOW"
        saved = rows(owner, "evidence")
        assert [r["task_id"] for r in saved] == ["task-0", "task-1", "task-2"]
        assert "SYSTEM" not in str(saved) and "hidden-value" not in str(saved) and "secret-value" not in str(saved)
        owner.pump()
        job = rows(owner, "jobs")[0]
        assert [r["session_id"] for r in json.loads(job["evidence_json"])] == ["session-0", "session-1"]
        assert len(rows(owner, "evidence")) == 3
        service = ReviewService(roster, provider=lambda _: {"action": "NONE"})
        assert service.once() == 1
        owner.pump()
        assert [r["task_id"] for r in rows(owner, "evidence")] == ["task-2"]
        assert rows(owner, "jobs")[0]["status"] == "NONE"
    finally:
        owner.close()


@pytest.mark.parametrize("messages", [[], [{"role":"user","content":"unfinished"}],
    [{"role":"user","content":"u"},{"role":"assistant","content":"calling", "tool_calls":[{"id":"x"}]}],
    [{"role":"user","content":"u"},{"role":"assistant","content":"a"},{"role":"tool","content":"end"}]])
def test_incomplete_evidence_excluded(messages):
    item = Evidence("s", "t")
    for msg in messages:
        item.add(msg)
    assert item.finish().status == "INCOMPLETE"


def test_evidence_overflow_is_explicit_and_minimal_completed_task_is_valid():
    assert evidence().status == "READY"
    item = Evidence("s", "t", max_bytes=256)
    item.add({"role":"user","content":"x" * 1000})
    item.add({"role":"assistant","content":"Done"})
    assert item.finish().status == "OVERFLOW"


def test_foreground_enqueue_does_not_scan_catalog_and_has_busy_refusal(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    try:
        with patch.object(owner.store, "prepare_review", side_effect=AssertionError("foreground scan")):
            assert owner.enqueue(evidence()).status == "ACCEPTED"
        with sqlite3.connect(owner.entry.mailbox) as blocker:
            blocker.execute("BEGIN EXCLUSIVE")
            started = time.monotonic()
            assert owner.enqueue(evidence(task="busy")).status == "FAILED"
            assert time.monotonic() - started < 1.5
    finally:
        owner.close()


def test_sqlite_busy_timeout_and_real_contention_use_approved_half_second(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    blocker = sqlite3.connect(owner.entry.mailbox, check_same_thread=False)
    try:
        assert review.BUSY_SECONDS == .5
        with review.connect(owner.entry) as conn:
            assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 500

        blocker.execute("BEGIN EXCLUSIVE")
        released = threading.Event()
        def release_old_limit_contention():
            time.sleep(.15)  # Longer than the former 50 ms bound, within 500 ms.
            blocker.rollback()
            released.set()
        thread = threading.Thread(target=release_old_limit_contention)
        thread.start()
        assert owner.enqueue(evidence(task="released-within-new-bound")).status == "ACCEPTED"
        thread.join(2)
        assert released.is_set()

        blocker.execute("BEGIN EXCLUSIVE")
        started = time.monotonic()
        assert owner.enqueue(evidence(task="held-beyond-new-bound")).status == "FAILED"
        assert time.monotonic() - started < 1.5
    finally:
        blocker.rollback()
        blocker.close()
        owner.close()


def test_private_context_global_rebind_idle_delivery_and_disconnected_reconnect(tmp_path):
    roster = setup_roster(tmp_path, 2)
    alice, bob = owner_for(roster, 0), owner_for(roster, 1)
    import tools
    previous = tools.skill_store.storage_dir
    seen = []
    try:
        alice.store.save_skill("alice_only", "Alice context", "Alice private steps.")
        bob.store.save_skill("bob_only", "Bob context", "Bob private steps.")
        tools.skill_store.storage_dir = bob.store.storage_dir
        assert alice.enqueue(evidence(task="alice-task", text="ALICE EVIDENCE")).status == "ACCEPTED"
        assert bob.enqueue(evidence(task="bob-task", text="BOB EVIDENCE")).status == "ACCEPTED"
        wait_for(lambda: len(rows(alice, "jobs")) == len(rows(bob, "jobs")) == 1)
        alice.close()

        def provider(request):
            seen.append(request)
            return create_proposal("alice_result" if "ALICE EVIDENCE" in request else "bob_result")

        assert ReviewService(roster, provider=provider).once() == 2
        wait_for(lambda: "bob_result" in bob.store.list_skills())
        assert "alice_result" not in alice.store.list_skills()
        assert "BOB EVIDENCE" not in seen[0] and "bob_only" not in seen[0]
        assert "ALICE EVIDENCE" not in seen[1] and "alice_only" not in seen[1]
        reconnected = owner_for(roster, 0)
        try:
            wait_for(lambda: "alice_result" in reconnected.store.list_skills())
            assert "bob_result" not in reconnected.store.list_skills()
        finally:
            reconnected.close()
    finally:
        tools.skill_store.storage_dir = previous
        alice.close()
        bob.close()


@pytest.mark.parametrize("running", [False, True])
def test_permission_generation_blocks_all_profile_writes_after_ack_and_reenable(tmp_path, running):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster)
    entered, release = threading.Event(), threading.Event()
    service = ReviewService(roster, provider=lambda _: (entered.set(), release.wait(4), create_proposal())[2])
    worker = None
    try:
        owner.enqueue(evidence())
        wait_for(lambda: len(rows(owner, "jobs")) == 1)
        if running:
            worker = threading.Thread(target=service.once)
            worker.start()
            assert entered.wait(2)
        owner.set_enabled(False)
        before = profile_bytes(owner)
        release.set()
        if worker:
            worker.join(3)
            assert not worker.is_alive()
        else:
            assert service.once() == 0
        time.sleep(.12)
        assert before == profile_bytes(owner)
        owner.set_enabled(True)
        time.sleep(.12)
        assert "learned" not in owner.store.list_skills()
        assert owner.enqueue(evidence(task="fresh")).status == "ACCEPTED"
    finally:
        release.set()
        if worker:
            worker.join(4)
        owner.close()


def test_readonly_start_creates_no_profile_and_can_restore_optin(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, enabled=False)
    try:
        assert not Path(owner.entry.mailbox).parent.exists()
        assert not Path(roster.control_dir).exists()
        assert owner.enqueue(evidence()).status == "DISABLED"
        owner.set_enabled(True)
        assert owner.enqueue(evidence()).status == "ACCEPTED"
    finally:
        owner.close()


def test_publication_before_ack_recovers_without_second_update(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    owner.store.save_skill("existing", "Workflow", "Old steps.")
    owner.enqueue(evidence())
    owner.pump()

    def provider(request):
        target = json.loads(request)["catalog"]["targets"][0]["target_id"]
        return dict(action="UPDATE", target_id=target, description="Workflow", instructions="Replacement complete steps.", complete=True)

    assert ReviewService(roster, provider=provider).once() == 1
    with patch.object(owner, "_ack", side_effect=OSError("crash after authoritative commit")):
        owner.pump()
    committed = (Path(owner.store.storage_dir) / "existing.md").read_bytes()
    owner.close()
    restarted = owner_for(roster, background=False)
    try:
        restarted.pump()
        assert (Path(restarted.store.storage_dir) / "existing.md").read_bytes() == committed
        assert rows(restarted, "jobs")[0]["status"] == "DUPLICATE"
    finally:
        restarted.close()


@pytest.mark.parametrize("column,value", [("profile_id","forged"), ("store_id","forged"),
    ("generation","forged"), ("result_id","forged"), ("job_id","forged"),
    ("service_generation","forged")])
def test_forged_result_identity_fails_closed(tmp_path, column, value):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    try:
        owner.enqueue(evidence())
        owner.pump()
        ReviewService(roster, provider=lambda _: create_proposal()).once()
        with sqlite3.connect(owner.entry.mailbox) as conn:
            conn.execute(f"UPDATE jobs SET {column}=?", (value,))
        owner.pump()
        assert not owner.store.get_all_skills()
    finally:
        owner.close()


def test_one_service_pool_fair_progress_error_isolation_and_singleton(tmp_path):
    roster = setup_roster(tmp_path, 4)
    owners = [owner_for(roster, i, background=False) for i in range(4)]
    entered, release = threading.Event(), threading.Event()
    active, peak, calls = 0, 0, 0

    def provider(request):
        nonlocal active, peak, calls
        active += 1
        peak = max(peak, active)
        calls += 1
        try:
            if calls == 1:
                entered.set()
                release.wait(4)
                raise TimeoutError("fake provider timeout")
            return {"action": "NONE"}
        finally:
            active -= 1

    service = ReviewService(roster, provider=provider)
    try:
        for i, owner in enumerate(owners):
            owner.enqueue(evidence(task=f"task-{i}"))
            owner.pump()
        worker = threading.Thread(target=service.once)
        worker.start()
        assert entered.wait(2)
        with pytest.raises(TimeoutError):
            ReviewService(roster, provider=provider).once()
        release.set()
        worker.join(5)
        assert not worker.is_alive() and calls == 4 and peak == 1
        for owner in owners:
            owner.pump()
        assert rows(owners[0], "jobs")[0]["status"] == "FAILED"
        assert all(rows(o, "jobs")[0]["status"] == "NONE" for o in owners[1:])
    finally:
        release.set()
        for owner in owners:
            owner.close()


def test_second_owner_same_thread_is_excluded(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    try:
        with pytest.raises(TimeoutError):
            owner_for(roster, background=False)
    finally:
        owner.close()


def test_hundred_private_profiles_make_bounded_progress(tmp_path):
    roster = setup_roster(tmp_path, 100)
    owners = []
    calls = []
    try:
        for index in range(100):
            owner = owner_for(roster, index, background=False)
            owners.append(owner)
            assert owner.enqueue(evidence(task=f"task-{index}")).status == "ACCEPTED"
            owner.pump()
        def provider(request):
            tasks = json.loads(request)["tasks"]
            assert len(tasks) == 1
            calls.append(tasks[0]["task_id"])
            return {"action":"NONE"}
        assert ReviewService(roster, provider=provider).once() == 100
        assert calls == [f"task-{i}" for i in range(100)]
    finally:
        for owner in owners:
            owner.close()


def test_shutdown_holds_slot_until_worker_exit_and_restart_fences_old_attempt(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    entered, release = threading.Event(), threading.Event()
    service = ReviewService(roster, provider=lambda _: (entered.set(), release.wait(4), create_proposal())[2])
    try:
        owner.enqueue(evidence())
        owner.pump()
        worker = threading.Thread(target=service.once)
        worker.start()
        assert entered.wait(2)
        old_generation = rows(owner, "jobs")[0]["service_generation"]
        service.stop()
        with pytest.raises(TimeoutError):
            ReviewService(roster, provider=lambda _: {"action":"NONE"}).once()
        release.set()
        worker.join(5)
        assert rows(owner, "jobs")[0]["status"] == "RUNNING"
        restarted = ReviewService(roster, provider=lambda _: {"action":"NONE"})
        assert restarted.once() == 1
        assert restarted.generation != old_generation
        assert rows(owner, "jobs")[0]["service_generation"] == restarted.generation
        owner.pump()
        assert not owner.store.get_all_skills()
    finally:
        release.set()
        owner.close()


@pytest.mark.parametrize("column", ["host_json", "catalog_json"])
def test_forged_snapshot_paths_or_target_contents_are_rejected(tmp_path, column):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    try:
        owner.enqueue(evidence())
        owner.pump()
        ReviewService(roster, provider=lambda _: create_proposal()).once()
        with sqlite3.connect(owner.entry.mailbox) as conn:
            conn.execute(f"UPDATE jobs SET {column}=?", ('{"targets":[{"relative_path":"../outside.md"}]}',))
        owner.pump()
        assert not owner.store.get_all_skills()
        assert not (tmp_path / "outside.md").exists()
    finally:
        owner.close()


def test_unanswered_tool_calls_cannot_pass_completion_gate():
    item = Evidence("s", "t")
    item.add({"role":"user", "content":"Task"})
    item.add({"role":"assistant", "content":"", "tool_calls":[{"id":"pending"}]})
    item.add({"role":"assistant", "content":"Claimed done without tool result"})
    assert item.finish().status == "INCOMPLETE"


def test_changed_service_model_cannot_consume_existing_authorization(tmp_path):
    from dataclasses import replace
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    calls = []
    try:
        owner.enqueue(evidence())
        owner.pump()
        changed = replace(roster, model="different-model")
        assert ReviewService(changed, provider=lambda x: calls.append(x) or {"action":"NONE"}).once() == 0
        assert not calls
        assert rows(owner, "jobs")[0]["status"] == "PREPARED"
    finally:
        owner.close()
