import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

import profile_paths
from profile_paths import control_directory
from skill_lock import path_identity
from skill_lock import ProcessLock


REPO = Path(__file__).resolve().parent


def records(root, name="alice", *, mode="discovery"):
    control = control_directory(root)
    control.mkdir(parents=True, exist_ok=True)
    store_id = path_identity(root / name / "skills")
    auth = control / f"{store_id}.json"
    authority = control / f"{store_id}.authority.json"
    auth.write_text(json.dumps({"sentinel": "authorization"}), encoding="utf-8")
    authority.write_text(json.dumps({
        "origin": {"mode": mode, "control_dir": str(control),
                   "model": "test-model", "base_url": "http://127.0.0.1:1/v1"},
        "generation": "a" * 32,
    }), encoding="utf-8")
    return control, store_id, auth, authority


def backups(path):
    return list(path.parent.glob(path.name + ".reset-backup-*"))


def test_reset_backs_up_only_exact_deleted_profile_records_and_is_idempotent(tmp_path):
    import reset_profile
    root = tmp_path / "profiles"
    control, _, auth, authority = records(root)
    other = control / "other.json"
    policy = control / "policy.json"
    other.write_bytes(b"other-profile")
    policy.write_bytes(b"shared-policy")

    moved = reset_profile.reset_profile("alice", root)
    assert moved == 2
    assert not auth.exists() and not authority.exists()
    assert [p.read_bytes() for p in backups(auth)] == [b'{"sentinel": "authorization"}']
    assert len(backups(authority)) == 1
    assert other.read_bytes() == b"other-profile"
    assert policy.read_bytes() == b"shared-policy"
    assert reset_profile.reset_profile("alice", root) == 0
    assert len(backups(auth)) == len(backups(authority)) == 1


def test_default_and_override_roots(tmp_path, monkeypatch):
    import reset_profile
    default_root, override_root = tmp_path / "default", tmp_path / "override"
    _, _, default_auth, _ = records(default_root)
    _, _, override_auth, _ = records(override_root)
    monkeypatch.setattr(profile_paths, "DEFAULT_PROFILES_DIR", default_root)
    assert reset_profile.main(["alice"]) == 0
    assert not default_auth.exists() and override_auth.exists()
    assert reset_profile.main(["alice", "--profiles-dir", str(override_root)]) == 0
    assert not override_auth.exists()


@pytest.mark.parametrize("name", ["../alice", "nested/alice", "bad.name", "", "a" * 65])
def test_traversal_and_invalid_names_are_rejected_without_writes(tmp_path, name):
    import reset_profile
    root = tmp_path / "profiles"
    before = set(tmp_path.rglob("*"))
    with pytest.raises(ValueError, match="1-64"):
        reset_profile.reset_profile(name, root)
    assert set(tmp_path.rglob("*")) == before


def test_existing_profile_directory_or_link_is_refused(tmp_path):
    import reset_profile
    root = tmp_path / "profiles"
    _, _, auth, authority = records(root)
    (root / "alice").mkdir(parents=True)
    with pytest.raises(ValueError, match="still exists"):
        reset_profile.reset_profile("alice", root)
    assert auth.exists() and authority.exists()


@pytest.mark.parametrize("authority_value", [
    {"origin": {"mode": "roster", "control_dir": "X", "model": "m", "base_url": "http://x"},
     "generation": "a" * 32},
    {"origin": {"mode": "discovery"}},
    [],
])
def test_static_or_malformed_authority_refused_without_mutation(tmp_path, authority_value):
    import reset_profile
    root = tmp_path / "profiles"
    _, _, auth, authority = records(root)
    authority.write_text(json.dumps(authority_value), encoding="utf-8")
    before = {auth: auth.read_bytes(), authority: authority.read_bytes()}
    with pytest.raises(ValueError):
        reset_profile.reset_profile("alice", root)
    assert {path: path.read_bytes() for path in before} == before
    assert not backups(auth) and not backups(authority)


def test_second_rename_failure_rolls_back_first_and_reports_truthfully(tmp_path, monkeypatch):
    import reset_profile
    root = tmp_path / "profiles"
    _, _, auth, authority = records(root)
    real_rename = os.rename
    calls = 0

    def fail_second(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected second rename failure")
        return real_rename(source, target)

    monkeypatch.setattr(reset_profile.os, "rename", fail_second)
    with pytest.raises(RuntimeError, match="rolled back"):
        reset_profile.reset_profile("alice", root)
    assert auth.exists() and authority.exists()
    assert not backups(auth) and not backups(authority)


def test_live_owner_lock_refuses_without_mutation(tmp_path):
    import reset_profile
    root = tmp_path / "profiles"
    _, store_id, auth, authority = records(root)
    with ProcessLock("skill-owner:" + store_id, timeout=0):
        with pytest.raises(RuntimeError, match="still running"):
            reset_profile.reset_profile("alice", root)
    assert auth.exists() and authority.exists()
    assert not backups(auth) and not backups(authority)


def test_real_cli_same_name_recovery_preserves_other_profile_and_workspace(tmp_path):
    root, workspaces = tmp_path / "profiles", tmp_path / "workspaces"
    workspace = workspaces / "alice"
    workspace.mkdir(parents=True)
    (workspace / "sentinel.bin").write_bytes(b"workspace-private-bytes")
    (root / "bob").mkdir(parents=True)
    (root / "bob" / "sentinel.bin").write_bytes(b"other-profile-bytes")
    before_other = (root / "bob" / "sentinel.bin").read_bytes()
    before_workspace = (workspace / "sentinel.bin").read_bytes()

    command = [sys.executable, str(REPO / "agent.py"), "--profile", "alice", "--profiles-dir", str(root),
               "--workspaces-dir", str(workspaces), "--auto-skills", "--no-memory",
               "--model", "test-model", "--base-url", "http://127.0.0.1:1/v1"]
    created = subprocess.run(command, input="exit\n", capture_output=True, text=True, cwd=REPO, timeout=10,
                             env=dict(os.environ, OPENAI_API_KEY="not-used"))
    assert created.returncode == 0, created.stdout + created.stderr
    shutil.rmtree(root / "alice")
    stale = subprocess.run(
        command,
        input="exit\n", capture_output=True, text=True, cwd=REPO, timeout=10,
        env=dict(os.environ, OPENAI_API_KEY="not-used"),
    )
    assert stale.returncode == 2
    reset = subprocess.run(
        [sys.executable, str(REPO / "reset_profile.py"), "alice", "--profiles-dir", str(root)],
        capture_output=True, text=True, cwd=REPO, timeout=10,
    )
    assert reset.returncode == 0, reset.stdout + reset.stderr
    recreated = subprocess.run(
        command,
        input="exit\n", capture_output=True, text=True, cwd=REPO, timeout=10,
        env=dict(os.environ, OPENAI_API_KEY="not-used"),
    )
    assert recreated.returncode == 0, recreated.stdout + recreated.stderr
    assert (root / "bob" / "sentinel.bin").read_bytes() == before_other
    assert (workspace / "sentinel.bin").read_bytes() == before_workspace
