import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

import profile_paths
from profile_paths import control_directory
from skill_lock import path_identity, ProcessLock


REPO = Path(__file__).resolve().parent


def tree_bytes(path):
    if not path.exists():
        return None
    return {
        item.relative_to(path).as_posix(): ("dir" if item.is_dir() else item.read_bytes())
        for item in sorted(path.rglob("*"))
    }


def provision(root, name, *, mailbox=True):
    profile = root / name
    (profile / "skills").mkdir(parents=True)
    (profile / "memories").mkdir()
    (profile / "logs").mkdir()
    (profile / "skills" / "procedure.md").write_bytes(b"skill")
    (profile / "memories" / "MEMORY.md").write_bytes(b"memory")
    (profile / "history.db").write_bytes(b"history")
    (profile / "logs" / "events.jsonl").write_bytes(b"log")
    if mailbox:
        for suffix, value in (("", b"db"), ("-wal", b"wal"), ("-shm", b"shm"), ("-journal", b"journal")):
            (profile / ("skill_review.db" + suffix)).write_bytes(value)
    return profile


def discovery_records(root, name, *, mode="discovery", incarnation=None):
    control = control_directory(root)
    control.mkdir(parents=True, exist_ok=True)
    profile = root / name
    store_id = path_identity(profile / "skills")
    authority = control / f"{store_id}.authority.json"
    auth = control / f"{store_id}.json"
    generation = "a" * 32
    origin = {
        "mode": mode,
        "control_dir": str(control),
        "model": "test-model",
        "base_url": "http://127.0.0.1:1/v1",
    }
    authority.write_text(json.dumps({"origin": origin, "generation": generation}), encoding="utf-8")
    value = {
        "profile_id": name,
        "store_id": store_id,
        "mailbox": str(profile / "skill_review.db"),
        "enabled": True,
        "generation": "b" * 32,
        "secret": "c" * 64,
        "authority_generation": generation,
        "model": origin["model"],
        "base_url": origin["base_url"],
        "owner_id": "d" * 32,
    }
    if incarnation is not None:
        value["incarnation"] = incarnation
    auth.write_text(json.dumps(value), encoding="utf-8")
    return control, store_id, auth, authority


def test_delete_is_dry_run_then_archives_only_named_profile_and_current_records(tmp_path):
    import delete_profile

    root = tmp_path / "profiles"
    workspaces = tmp_path / "workspaces"
    alice, bob = provision(root, "alice"), provision(root, "bob")
    workspace = workspaces / "alice"
    workspace.mkdir(parents=True)
    (workspace / "keep.bin").write_bytes(b"workspace")
    control, _, auth, authority = discovery_records(root, "alice")
    unrelated = control / "unrelated.json"
    policy = control / "policy.json"
    unrelated.write_bytes(b"other")
    policy.write_bytes(b"policy")
    before = tree_bytes(tmp_path)

    preview = delete_profile.delete_profile("alice", root)
    assert not preview.applied and preview.backup_path is None
    assert tree_bytes(tmp_path) == before

    result = delete_profile.delete_profile("alice", root, apply=True)
    assert result.applied and result.backup_path.is_dir()
    assert not alice.exists() and bob.is_dir()
    assert (result.backup_path / "profile" / "skills" / "procedure.md").read_bytes() == b"skill"
    assert not auth.exists() and not authority.exists()
    assert (result.backup_path / "control" / auth.name).is_file()
    assert (result.backup_path / "control" / authority.name).is_file()
    assert unrelated.read_bytes() == b"other" and policy.read_bytes() == b"policy"
    assert (workspace / "keep.bin").read_bytes() == b"workspace"

    # Same-name recreation is a fresh profile and needs no manual cleanup.
    recreated = provision(root, "alice", mailbox=False)
    assert recreated.is_dir()


def test_reset_all_is_dry_run_then_archives_mailbox_units_and_current_pairs(tmp_path):
    import reset_profile_locks

    root = tmp_path / "profiles"
    workspaces = tmp_path / "workspaces"
    for name in ("alice", "bob"):
        provision(root, name)
        discovery_records(root, name)
        workspace = workspaces / name
        workspace.mkdir(parents=True)
        (workspace / "keep.bin").write_bytes(name.encode())
    control = control_directory(root)
    policy = control / "policy.json"
    policy.write_bytes(b"policy")
    unrelated = control / "unrelated.json"
    unrelated.write_bytes(b"unrelated")
    before = tree_bytes(tmp_path)

    preview = reset_profile_locks.reset_profile_locks(root)
    assert not preview.applied and preview.backup_path is None
    assert tree_bytes(tmp_path) == before

    result = reset_profile_locks.reset_profile_locks(root, apply=True)
    assert result.applied and result.backup_path.is_dir()
    for name in ("alice", "bob"):
        profile = root / name
        assert not list(profile.glob("skill_review.db*"))
        assert (profile / "skills" / "procedure.md").read_bytes() == b"skill"
        assert (profile / "memories" / "MEMORY.md").read_bytes() == b"memory"
        assert (profile / "history.db").read_bytes() == b"history"
        assert (profile / "logs" / "events.jsonl").read_bytes() == b"log"
        assert (workspaces / name / "keep.bin").read_bytes() == name.encode()
        archived = result.backup_path / "profiles" / name
        assert {p.name for p in archived.iterdir()} == {
            "skill_review.db", "skill_review.db-wal", "skill_review.db-shm", "skill_review.db-journal"
        }
    assert policy.read_bytes() == b"policy" and unrelated.read_bytes() == b"unrelated"
    assert len(list((result.backup_path / "control").iterdir())) == 4

    # Idempotence: no sources means no new backup.
    again = reset_profile_locks.reset_profile_locks(root, apply=True)
    assert again.changed == 0 and again.backup_path is None


def test_reset_archives_orphan_mailbox_but_leaves_dormant_old_control(tmp_path):
    import reset_profile_locks

    old_parent = tmp_path / "old-install"
    old_root = old_parent / "profiles"
    provision(old_root, "alice")
    old_control, _, _, _ = discovery_records(old_root, "alice")
    marker = old_control / "dormant-marker"
    marker.write_bytes(b"untouched")
    new_parent = tmp_path / "new-install"
    old_parent.rename(new_parent)
    new_root = new_parent / "profiles"
    assert control_directory(new_root) != new_parent / old_control.name
    before_dormant = tree_bytes(new_parent / old_control.name)

    result = reset_profile_locks.reset_profile_locks(new_root, apply=True)
    assert result.changed == 4
    assert not list((new_root / "alice").glob("skill_review.db*"))
    assert tree_bytes(new_parent / old_control.name) == before_dormant
    assert not control_directory(new_root).exists()


def test_reset_accepts_stale_incarnation_but_rejects_static_authority_atomically(tmp_path):
    import reset_profile_locks

    root = tmp_path / "profiles"
    provision(root, "alice")
    discovery_records(root, "alice", incarnation={"root": [1, 2, 3], "profile": [4, 5, 6]})
    result = reset_profile_locks.reset_profile_locks(root, apply=True)
    assert result.changed == 6

    for suffix, value in (("", b"db"), ("-wal", b"wal"), ("-shm", b"shm"), ("-journal", b"journal")):
        (root / "alice" / ("skill_review.db" + suffix)).write_bytes(value)
    discovery_records(root, "alice")
    provision(root, "bob")
    _, _, _, authority = discovery_records(root, "bob", mode="roster")
    before = tree_bytes(tmp_path)
    with pytest.raises(ValueError, match="Static/custom"):
        reset_profile_locks.reset_profile_locks(root, apply=True)
    assert authority.exists() and tree_bytes(tmp_path) == before


@pytest.mark.parametrize("operation", ["delete", "reset"])
@pytest.mark.parametrize("lock_kind", ["service", "owner"])
def test_live_service_or_owner_lock_refuses_all_mutation(tmp_path, operation, lock_kind):
    import delete_profile
    import reset_profile_locks

    root = tmp_path / "profiles"
    provision(root, "alice")
    _, store_id, _, _ = discovery_records(root, "alice")
    before = tree_bytes(tmp_path)
    identity = "hermes-skill-review-service" if lock_kind == "service" else "skill-owner:" + store_id
    with ProcessLock(identity, timeout=0):
        with pytest.raises(RuntimeError, match="still running"):
            if operation == "delete":
                delete_profile.delete_profile("alice", root, apply=True)
            else:
                reset_profile_locks.reset_profile_locks(root, apply=True)
    assert tree_bytes(tmp_path) == before


def test_invalid_names_and_missing_roots_refuse_without_mutation(tmp_path):
    import delete_profile
    import reset_profile_locks

    missing = tmp_path / "missing"
    with pytest.raises(ValueError, match="does not exist"):
        reset_profile_locks.reset_profile_locks(missing, apply=True)
    with pytest.raises(ValueError, match="1-64"):
        delete_profile.delete_profile("../alice", missing, apply=True)

    root = tmp_path / "profiles"
    root.mkdir()
    (root / "bad.name").mkdir()
    before = tree_bytes(tmp_path)
    with pytest.raises(ValueError, match="1-64"):
        reset_profile_locks.reset_profile_locks(root, apply=True)
    assert tree_bytes(tmp_path) == before


def test_profile_links_refuse_without_mutation(tmp_path):
    import reset_profile_locks

    root = tmp_path / "profiles"
    root.mkdir()
    target = tmp_path / "elsewhere"
    target.mkdir()
    try:
        (root / "alice").symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks unavailable")
    before = tree_bytes(tmp_path)
    with pytest.raises(ValueError, match="symlinks|reparse"):
        reset_profile_locks.reset_profile_locks(root, apply=True)
    assert tree_bytes(tmp_path) == before


def test_hard_linked_mailbox_refuses_without_mutation(tmp_path):
    import reset_profile_locks

    root = tmp_path / "profiles"
    profile = provision(root, "alice")
    alias = tmp_path / "mailbox-alias"
    os.link(profile / "skill_review.db", alias)
    before = tree_bytes(tmp_path)
    with pytest.raises(ValueError, match="without aliases"):
        reset_profile_locks.reset_profile_locks(root, apply=True)
    assert tree_bytes(tmp_path) == before


@pytest.mark.parametrize("module_name,function_name", [
    ("delete_profile", "delete_profile"),
    ("reset_profile_locks", "reset_profile_locks"),
])
def test_rename_failure_rolls_back_without_partial_mutation(tmp_path, monkeypatch, module_name, function_name):
    module = __import__(module_name)
    import profile_maintenance
    root = tmp_path / "profiles"
    provision(root, "alice")
    discovery_records(root, "alice")
    before = tree_bytes(tmp_path)
    real_rename = profile_maintenance.os.rename
    calls = 0

    def fail_second(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected rename failure")
        return real_rename(source, target)

    monkeypatch.setattr(profile_maintenance.os, "rename", fail_second)
    with pytest.raises(RuntimeError, match="rolled back"):
        if function_name == "delete_profile":
            module.delete_profile("alice", root, apply=True)
        else:
            module.reset_profile_locks(root, apply=True)
    assert tree_bytes(tmp_path) == before


def test_cli_defaults_are_application_anchored_and_dry_run(tmp_path, monkeypatch, capsys):
    import delete_profile
    import reset_profile_locks

    root = tmp_path / "profiles"
    provision(root, "alice")
    monkeypatch.setattr(profile_paths, "DEFAULT_PROFILES_DIR", root)
    monkeypatch.chdir(tmp_path)
    assert delete_profile.main(["alice"]) == 0
    assert reset_profile_locks.main([]) == 0
    assert (root / "alice").is_dir()
    assert "DRY RUN" in capsys.readouterr().out


def run_agent(root, name, *, auto_skills):
    command = [
        sys.executable, str(REPO / "agent.py"), "--profile", name,
        "--profiles-dir", str(root), "--workspaces-dir", str(root.parent / "workspaces"),
        "--no-memory", "--model", "test-model", "--base-url", "http://127.0.0.1:1/v1",
    ]
    if auto_skills:
        command.append("--auto-skills")
    return subprocess.run(
        command, input="exit\n", capture_output=True, text=True, cwd=REPO, timeout=15,
        env=dict(os.environ, OPENAI_API_KEY="not-used"),
    )


def test_public_startup_after_installation_move_fails_then_reset_recovers(tmp_path):
    old_install = tmp_path / "old-install"
    old_root = old_install / "profiles"
    first = run_agent(old_root, "alice", auto_skills=True)
    assert first.returncode == 0, first.stdout + first.stderr
    new_install = tmp_path / "new-install"
    old_install.rename(new_install)
    new_root = new_install / "profiles"

    stale = run_agent(new_root, "alice", auto_skills=True)
    assert stale.returncode == 2
    reset = subprocess.run(
        [sys.executable, str(REPO / "reset_profile_locks.py"), "--profiles-dir", str(new_root), "--apply"],
        capture_output=True, text=True, cwd=REPO, timeout=15,
    )
    assert reset.returncode == 0, reset.stdout + reset.stderr
    recovered = run_agent(new_root, "alice", auto_skills=True)
    assert recovered.returncode == 0, recovered.stdout + recovered.stderr
    nonlearning = run_agent(new_root, "alice", auto_skills=False)
    assert nonlearning.returncode == 0, nonlearning.stdout + nonlearning.stderr
