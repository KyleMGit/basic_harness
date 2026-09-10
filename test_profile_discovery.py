"""Automatic profile discovery through the public service and owner boundaries."""

from pathlib import Path
import subprocess
import sys
import json
import os
import sqlite3
import threading
import time

import pytest


REPO = Path(__file__).resolve().parent


def test_service_run_help_exposes_automatic_discovery():
    completed = subprocess.run(
        [sys.executable, str(REPO / "skill_review.py"), "run", "--help"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--profiles-dir" in completed.stdout
    assert "--discovery-interval" in completed.stdout


def discovery(root, **kwargs):
    from skill_review import DiscoveryRoster
    return DiscoveryRoster(root, kwargs.get("model", "test-model"),
                           kwargs.get("base_url", "http://127.0.0.1:1/v1"))


def test_missing_root_live_additions_and_stable_private_identities(tmp_path):
    from skill_lock import path_identity
    root = tmp_path / "profiles"
    config = discovery(root)
    assert config.refresh() == ()
    assert not root.exists()
    assert not Path(config.control_dir).exists()
    for name in ("alice", "bob"):
        (root / name).mkdir(parents=True)
    entries = config.refresh()
    assert {p.profile_id for p in entries} == {"alice", "bob"}
    assert config.refresh() == entries
    other = discovery(root)
    assert other.refresh() == entries
    assert other.control_dir == config.control_dir
    assert not Path(config.control_dir).is_relative_to(root)
    for entry in entries:
        assert entry.store_id == path_identity(root / entry.profile_id / "skills")
        assert Path(entry.mailbox) == root / entry.profile_id / "skill_review.db"
    assert list(root.rglob("*")) == [root / "alice", root / "bob"]


def test_metadata_discovery_isolates_bad_children_and_has_no_100_profile_cap(tmp_path):
    root = tmp_path / "profiles"
    root.mkdir()
    (root / "file").write_text("not a directory")
    (root / "bad.name").mkdir()
    (root / "healthy" / "nested").mkdir(parents=True)
    for i in range(105):
        (root / f"user-{i}").mkdir()
    config = discovery(root)
    assert len(config.refresh()) == 106
    assert "file" in config.errors and "bad.name" in config.errors
    assert "nested" not in {p.profile_id for p in config.profiles}
    with pytest.raises(ValueError):
        config.entry("../healthy")


def test_replaced_root_is_not_adopted_by_running_discovery(tmp_path):
    root = tmp_path / "profiles"
    (root / "alice").mkdir(parents=True)
    config = discovery(root)
    config.refresh()
    root.rename(tmp_path / "retired")
    (root / "bob").mkdir(parents=True)
    assert config.refresh() == ()
    assert "root" in config.errors
    with pytest.raises(ValueError, match="changed"):
        config.entry("bob")


def automatic_owner(root, who="alice", **kwargs):
    from skill_review import Owner
    from skills import SkillStore
    (root / who).mkdir(parents=True, exist_ok=True)
    config = discovery(root)
    return Owner(config, who, SkillStore(str(root / who / "skills")), background=False, **kwargs)


def test_disconnected_startup_and_late_interrupted_queue_recovery(tmp_path):
    from skill_review import ReviewService
    from test_async_skill_review import evidence, rows, wait_for
    root = tmp_path / "profiles"
    alice = automatic_owner(root)
    alice.enqueue(evidence(task="existing"))
    alice.pump()
    alice.close()
    config = discovery(root)
    calls = []
    service = ReviewService(config, provider=lambda request: (calls.append(request), {"action": "NONE"})[1], discovery_interval=.05)
    thread = threading.Thread(target=service.run)
    thread.start()
    bob = None
    try:
        wait_for(lambda: rows(alice, "jobs")[0]["status"] == "RESULT")
        # Owner-only prepared queue appears after the service has started. An
        # interrupted prior singleton's attempt must recover without a restart.
        bob = automatic_owner(root, "bob")
        bob.enqueue(evidence(task="late"))
        bob.pump()
        with sqlite3.connect(bob.entry.mailbox) as conn:
            conn.execute("UPDATE jobs SET status='RUNNING',service_generation='exited-service'")
        bob.close()
        wait_for(lambda: rows(bob, "jobs")[0]["status"] == "RESULT")
        assert len(calls) == 2
        assert '"existing"' in calls[0] and '"late"' in calls[1]
        assert service.generation == rows(bob, "jobs")[0]["service_generation"]
        assert not (root / "alice" / "skills").exists()
        assert not (root / "bob" / "skills").exists()
    finally:
        service.stop()
        thread.join(5)
        if bob:
            bob.close()
    assert not thread.is_alive()


@pytest.mark.parametrize("replace", [False, True])
def test_missing_or_replaced_profile_cannot_receive_inflight_result_or_restart_work(tmp_path, replace):
    from skill_review import ReviewService
    from test_async_skill_review import evidence, rows, create_proposal
    root = tmp_path / "profiles"
    alice = automatic_owner(root)
    alice.enqueue(evidence())
    alice.pump()
    alice.close()
    entered, release = threading.Event(), threading.Event()
    calls = []
    service = ReviewService(discovery(root), provider=lambda request: (calls.append(request), entered.set(), release.wait(4), create_proposal())[3])
    thread = threading.Thread(target=service.once)
    thread.start()
    try:
        assert entered.wait(2)
        (root / "alice").rename(tmp_path / "retired-alice")
        if replace:
            (root / "alice").mkdir()
            # Simulate copied stale state in a new directory incarnation.
            (root / "alice" / "skill_review.db").write_bytes((tmp_path / "retired-alice" / "skill_review.db").read_bytes())
        before = tree_bytes(root)
        release.set()
        thread.join(5)
        assert tree_bytes(root) == before
        restarted = ReviewService(discovery(root), provider=lambda request: calls.append(request))
        assert restarted.once() == 0
        assert len(calls) == 1
        assert tree_bytes(root) == before
        if replace:
            assert "alice" in restarted.errors
    finally:
        release.set()
        thread.join(5)


def tree_bytes(root):
    return {str(path.relative_to(root)): (path.read_bytes() if path.is_file() else None, path.stat().st_mtime_ns)
            for path in ([root, *root.rglob("*")] if root.exists() else [])}


def run_agent(root, workspace, *flags):
    return subprocess.run([sys.executable, str(REPO / "agent.py"), "--profile", "alice", "--profiles-dir", str(root),
        "--workspace", str(workspace), "--model", "test-model", "--base-url", "http://127.0.0.1:1/v1", *flags],
        input="", capture_output=True, text=True, timeout=8, cwd=REPO)


@pytest.mark.parametrize("flag", ["--read-only", "--stateless", "--no-skills", "--no-auto-skills"])
def test_disabled_public_launch_revokes_disconnected_auto_work_without_roster(tmp_path, flag):
    from skill_review import ReviewService, read_auth
    from test_async_skill_review import evidence
    root = tmp_path / "profiles"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    # Provision through normal public startup first, then retain disconnected work.
    completed = run_agent(root, workspace, "--auto-skills", "--no-memory")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    alice = automatic_owner(root)
    alice.enqueue(evidence())
    alice.pump()
    alice.close()
    protected = {name: tree_bytes(root / "alice" / name) for name in ("skills",)}
    mailbox = (root / "alice" / "skill_review.db").read_bytes()
    completed = run_agent(root, workspace, flag, "--auto-skills", "--no-memory")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert not read_auth(alice.roster, alice.entry)["enabled"]
    calls = []
    assert ReviewService(discovery(root), provider=lambda request: calls.append(request)).once() == 0
    assert not calls
    assert (root / "alice" / "skill_review.db").read_bytes() == mailbox
    assert {name: tree_bytes(root / "alice" / name) for name in protected} == protected
    if flag in ("--read-only", "--stateless"):
        before = tree_bytes(root)
        assert run_agent(root, workspace, flag, "--no-memory").returncode == 0
        assert tree_bytes(root) == before


def test_policy_mismatch_and_mode_conflicts_fail_before_provisioning(tmp_path):
    from skill_review import ReviewService
    root = tmp_path / "profiles"
    config = discovery(root, model="original")
    assert ReviewService(config, provider=lambda _: {"action": "NONE"}).once() == 0
    control_before = tree_bytes(Path(config.control_dir))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    completed = run_agent(root, workspace, "--auto-skills")
    assert completed.returncode == 2 and "policy mismatch" in completed.stderr
    assert not root.exists()
    assert tree_bytes(Path(config.control_dir)) == control_before
    for args in (("--roster", "missing.json", "--profiles-dir", root),
                 ("--roster", "missing.json", "--model", "test-model"),
                 ("--roster", "missing.json", "--base-url", "http://127.0.0.1:1/v1"),
                 ("--roster", "missing.json", "--discovery-interval", "1")):
        completed = subprocess.run([sys.executable, str(REPO / "skill_review.py"), "once", *map(str, args)],
                                   capture_output=True, text=True, timeout=5)
        assert completed.returncode == 2
        assert "not allowed" in completed.stderr or "conflicts" in completed.stdout
    assert not root.exists()


@pytest.mark.parametrize("placement", ["root", "child", "skills", "memory"])
def test_actual_junction_or_symlink_rejected_before_outside_reads_and_writes(tmp_path, placement):
    root = tmp_path / "profiles"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "sentinel").write_text("PRIVATE OUTSIDE")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    link = {"root": root, "child": root / "alice", "skills": root / "alice" / "skills",
            "memory": root / "alice" / "memories"}[placement]
    link.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        result = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(outside)], capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        link.symlink_to(outside, target_is_directory=True)
    try:
        before = tree_bytes(outside)
        completed = run_agent(root, workspace, "--auto-skills", "--no-memory")
        assert completed.returncode == 2, completed.stdout + completed.stderr
        assert "reparse" in completed.stderr
        if placement == "root":
            with pytest.raises(ValueError, match="reparse"):
                discovery(root)
        else:
            (root / "healthy").mkdir()
            config = discovery(root)
            assert [p.profile_id for p in config.refresh()] == ["healthy"]
            assert "alice" in config.errors
        assert tree_bytes(outside) == before
    finally:
        os.rmdir(link) if os.name == "nt" else link.unlink()


def test_switching_modes_requires_revocation_and_never_revives_old_generations(tmp_path):
    from skill_review import Owner, ProfileEntry, Roster, ReviewService, read_auth
    from skills import SkillStore
    from test_async_skill_review import evidence, rows
    root = tmp_path / "profiles"
    (root / "alice").mkdir(parents=True)
    auto = discovery(root)
    entry = auto.entry("alice")
    static = Roster(str(tmp_path / "static-control"), auto.model, auto.base_url,
                    (ProfileEntry(entry.profile_id, entry.mailbox, entry.store_id),))
    store = SkillStore(str(root / "alice" / "skills"))
    first = Owner(static, "alice", store, background=False)
    first.enqueue(evidence(task="must-not-revive"))
    first.pump()
    old_generation = read_auth(static, static.entry("alice"))["generation"]
    first.close()
    before = tree_bytes(root)
    with pytest.raises(ValueError, match="revoke"):
        Owner(auto, "alice", store, background=False)
    assert tree_bytes(root) == before
    disabled = Owner(static, "alice", store, enabled=False, background=False)
    disabled.close()
    second = Owner(auto, "alice", store, background=False)
    try:
        assert read_auth(auto, entry)["generation"] != old_generation
        assert not rows(second, "jobs")
        second.enqueue(evidence(task="new-auto"))
        second.pump()
        calls = []
        assert ReviewService(static, provider=lambda request: calls.append(request)).once() == 0
        assert not calls  # The old static authority cannot process the new queue.
        second.set_enabled(False)
    finally:
        second.close()
    third = Owner(static, "alice", store, background=False)
    try:
        assert not rows(third, "jobs")
        assert read_auth(static, static.entry("alice"))["generation"] != old_generation
        assert ReviewService(auto, provider=lambda request: calls.append(request)).once() == 0
        assert not calls
    finally:
        third.close()


def test_legacy_unmarked_mailbox_requires_static_revocation_without_migration(tmp_path):
    from skill_review import Owner, ProfileEntry, Roster, ReviewService, authority_path
    from skills import SkillStore
    from test_async_skill_review import evidence
    root = tmp_path / "profiles"
    (root / "alice").mkdir(parents=True)
    auto = discovery(root)
    entry = auto.entry("alice")
    static = Roster(str(tmp_path / "static-control"), auto.model, auto.base_url,
                    (ProfileEntry(entry.profile_id, entry.mailbox, entry.store_id),))
    store = SkillStore(str(root / "alice" / "skills"))
    owner = Owner(static, "alice", store, background=False)
    owner.enqueue(evidence())
    owner.pump()
    owner.close()
    authority_path(entry).unlink()  # Fixture simulates an unchanged pre-discovery mailbox.
    before = tree_bytes(root)
    assert ReviewService(auto, provider=lambda _: pytest.fail("Legacy queue was inferred")).once() == 0
    with pytest.raises(ValueError, match="static roster"):
        Owner(auto, "alice", store, background=False)
    assert tree_bytes(root) == before
    disabled = Owner(static, "alice", store, enabled=False, background=False)
    disabled.close()
    enabled = Owner(auto, "alice", store, background=False)
    enabled.close()


def test_static_allowlist_does_not_discover_unlisted_authorized_automatic_profile(tmp_path):
    from skill_review import Roster, ReviewService
    from test_async_skill_review import evidence, rows
    root = tmp_path / "profiles"
    bob = automatic_owner(root, "bob")
    bob.enqueue(evidence())
    bob.pump()
    bob.close()
    static = Roster(str(tmp_path / "static-control"), "test-model", "http://127.0.0.1:1/v1", ())
    assert ReviewService(static, provider=lambda _: pytest.fail("Unlisted profile admitted")).once() == 0
    assert rows(bob, "jobs")[0]["status"] == "PREPARED"


def test_discovery_does_not_enable_learning_or_open_an_unowned_mailbox(tmp_path):
    from skill_review import ReviewService
    root = tmp_path / "profiles"
    (root / "alice").mkdir(parents=True)
    (root / "alice" / "skill_review.db").write_bytes(b"unowned invalid SQLite contents")
    before = tree_bytes(root)
    service = ReviewService(discovery(root), provider=lambda _: pytest.fail("Directory discovery authorized learning"))
    assert service.once() == 0
    assert service.errors == {}
    assert tree_bytes(root) == before


@pytest.mark.parametrize("name", ["../outside", "nested/alice", "", "bad.name", "a" * 65])
def test_invalid_public_profile_id_refuses_before_writes(tmp_path, name):
    root, workspace = tmp_path / "profiles", tmp_path / "workspace"
    workspace.mkdir()
    completed = run_agent(root, workspace, "--profile", name, "--auto-skills")
    assert completed.returncode == 2 and "1-64" in completed.stderr
    assert not root.exists()


def test_profile_root_is_not_granted_to_workspace_or_read_only_tools(tmp_path):
    root, workspace = tmp_path / "profiles", tmp_path / "workspace"
    workspace.mkdir()
    root.mkdir()
    for flags in (("--workspace", str(tmp_path)), ("--read-only-dir", str(root))):
        completed = run_agent(root, workspace, "--auto-skills", *flags)
        assert completed.returncode == 2 and "protected profiles" in completed.stderr
        assert not (root / "alice").exists()


@pytest.mark.parametrize("flag", ["--read-only", "--stateless"])
def test_explicit_disabled_launch_does_not_provision_a_missing_profile(tmp_path, flag):
    from profile_paths import control_directory
    root, workspace = tmp_path / "profiles", tmp_path / "workspace"
    workspace.mkdir()
    completed = run_agent(root, workspace, flag, "--no-memory")
    assert completed.returncode == 2
    assert not root.exists()
    assert not control_directory(root).exists()


@pytest.mark.parametrize("flag", ["--no-skills", "--no-auto-skills"])
def test_learning_disabled_public_exit_preserves_normal_profile_provisioning(tmp_path, flag):
    from profile_paths import control_directory
    from test_skill_review_process import fake_openai
    root, workspaces = tmp_path / "profiles", tmp_path / "workspaces"
    with fake_openai() as (endpoint, _, _, requests):
        completed = subprocess.run(
            [sys.executable, str(REPO / "agent.py"), "--profile", "alice", "--profiles-dir", str(root),
             "--workspaces-dir", str(workspaces), "--model", "test-model", "--base-url", endpoint,
             "--auto-skills", flag],
            input="exit\n", capture_output=True, text=True, timeout=8, cwd=REPO,
            env=dict(os.environ, OPENAI_API_KEY="local-fake-test-key"),
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert "Exiting agent harness." in completed.stdout
        profile = root / "alice"
        assert (workspaces / "alice").is_dir()
        assert (profile / "memories" / "USER.md").is_file()
        assert (profile / "memories" / "MEMORY.md").is_file()
        with sqlite3.connect((profile / "history.db").as_uri() + "?mode=ro", uri=True) as conn:
            assert conn.execute("PRAGMA quick_check").fetchone() == ("ok",)
        assert (profile / "skills").is_dir()
        assert list((profile / "skills").iterdir()) == []
        assert not (profile / "skill_review.db").exists()
        assert not control_directory(root).exists()
        assert requests == []


def test_disabled_launch_revokes_deleted_profile_authorization_without_recreating_it(tmp_path):
    from skill_review import auth_path
    root, workspace = tmp_path / "profiles", tmp_path / "workspace"
    workspace.mkdir()
    alice = automatic_owner(root)
    alice.close()
    (root / "alice").rename(tmp_path / "retired-alice")
    completed = run_agent(root, workspace, "--read-only", "--no-memory")
    assert completed.returncode == 2
    assert not (root / "alice").exists()
    assert not json.loads(auth_path(alice.roster, alice.entry).read_text())["enabled"]
