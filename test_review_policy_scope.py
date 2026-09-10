"""Public non-learning CLI sessions must not inherit another review policy."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from profile_paths import control_directory
from skill_review import Owner, ProfileEntry, Roster, auth_path, authority_path
from skills import SkillStore
from test_profile_discovery import automatic_owner, tree_bytes
from test_skill_review_process import fake_openai


REPO = Path(__file__).resolve().parent


@pytest.fixture
def public_cli(tmp_path):
    root, workspaces = tmp_path / "profiles", tmp_path / "workspaces"
    with fake_openai() as (endpoint, _, _, requests):
        def launch(who, *, model="review-model", base_url=endpoint, flags=()):
            return subprocess.run(
                [sys.executable, "-B", str(REPO / "agent.py"), "--profile", who,
                 "--profiles-dir", str(root), "--workspaces-dir", str(workspaces),
                 "--model", model, "--base-url", base_url, *flags],
                input="exit\n", capture_output=True, text=True, timeout=10, cwd=REPO,
                env=dict(os.environ, OPENAI_API_KEY="local-fake-test-key"),
            )
        yield root, workspaces, endpoint, requests, launch
        assert requests == []


@pytest.mark.parametrize("scope,flags", [
    ("model", ()),
    ("endpoint", ("--no-auto-skills",)),
    ("both", ("--no-skills",)),
])
def test_unrelated_and_revoked_public_sessions_do_not_inherit_learning_policy(public_cli, scope, flags):
    root, workspaces, endpoint, requests, launch = public_cli
    alternative = dict(model="session-model" if scope != "endpoint" else "review-model",
                       base_url=endpoint + "/other" if scope != "model" else endpoint)
    authorized = launch("alice", flags=("--auto-skills",))
    assert authorized.returncode == 0, authorized.stdout + authorized.stderr
    control = control_directory(root)
    before_control, before_profile = tree_bytes(control), tree_bytes(root / "alice")
    active = launch("alice", **alternative, flags=flags)
    assert active.returncode == 2, active.stdout + active.stderr
    assert tree_bytes(control) == before_control
    assert tree_bytes(root / "alice") == before_profile

    unrelated = launch("bob", **alternative, flags=flags)
    assert unrelated.returncode == 0, unrelated.stdout + unrelated.stderr
    assert (workspaces / "bob").is_dir()
    for relative in ("history.db", "memories/USER.md", "memories/MEMORY.md"):
        assert (root / "bob" / relative).is_file()
    assert not (root / "bob" / "skill_review.db").exists()
    assert list((root / "bob" / "skills").iterdir()) == []
    assert tree_bytes(control) == before_control

    learning = launch("charlie", **alternative, flags=("--auto-skills",))
    assert learning.returncode == 2, learning.stdout + learning.stderr
    assert not (root / "charlie").exists()
    assert not (workspaces / "charlie").exists()
    assert tree_bytes(control) == before_control

    revoked = launch("alice", flags=("--no-auto-skills",))
    assert revoked.returncode == 0, revoked.stdout + revoked.stderr
    before_control = tree_bytes(control)
    before_mailbox = (root / "alice" / "skill_review.db").read_bytes()
    before_skills = tree_bytes(root / "alice" / "skills")
    ordinary = launch("alice", **alternative, flags=flags)
    assert ordinary.returncode == 0, ordinary.stdout + ordinary.stderr
    assert tree_bytes(control) == before_control
    assert (root / "alice" / "skill_review.db").read_bytes() == before_mailbox
    assert tree_bytes(root / "alice" / "skills") == before_skills
    assert requests == []


@pytest.mark.parametrize("current_state", ["enabled", "revoked", "missing", "stale_generation", "wrong_identity", "malformed"])
def test_nonlearning_checks_current_authority_instead_of_old_disabled_auth(public_cli, tmp_path, current_state):
    root, _, endpoint, _, launch = public_cli
    old = automatic_owner(root)
    old.set_enabled(False)
    old.close()
    stale_path = auth_path(old.roster, old.entry)
    assert json.loads(stale_path.read_text())["enabled"] is False
    static = Roster(str(tmp_path / "static-control"), "static-model", endpoint,
                    (ProfileEntry("alice", old.entry.mailbox, old.entry.store_id),))
    current = Owner(static, "alice", SkillStore(str(root / "alice" / "skills")), background=False)
    if current_state != "enabled":
        current.set_enabled(False)
    current.close()
    current_path = auth_path(static, static.entry("alice"))
    if current_state == "missing":
        current_path.unlink()
    elif current_state in ("stale_generation", "wrong_identity"):
        auth = json.loads(current_path.read_text())
        auth["authority_generation" if current_state == "stale_generation" else "store_id"] = "0" * 64
        current_path.write_text(json.dumps(auth))
    elif current_state == "malformed":
        current_path.write_text('{"enabled":false}')
    before = {path: tree_bytes(path) for path in (control_directory(root), Path(static.control_dir), root)}
    ordinary = launch("alice", model="ordinary-model", flags=("--no-auto-skills",))
    assert ordinary.returncode == (0 if current_state == "revoked" else 2), ordinary.stdout + ordinary.stderr
    for path in (control_directory(root), Path(static.control_dir)):
        assert tree_bytes(path) == before[path]
    if current_state != "revoked":
        assert tree_bytes(root) == before[root]
    assert json.loads(stale_path.read_text())["enabled"] is False
    assert (root / "alice" / "skill_review.db").read_bytes() == before[root]["alice/skill_review.db".replace("/", os.sep)][0]


@pytest.mark.parametrize("state", ["legacy_mailbox", "orphan_auth"])
def test_unknown_review_state_is_not_proof_of_no_learning(public_cli, state):
    root, _, _, _, launch = public_cli
    owner = automatic_owner(root)
    owner.set_enabled(False)
    owner.close()
    authority_path(owner.entry).unlink()
    if state == "legacy_mailbox":
        auth_path(owner.roster, owner.entry).unlink()
    else:
        Path(owner.entry.mailbox).unlink()
    before_control, before_profile = tree_bytes(control_directory(root)), tree_bytes(root)
    ordinary = launch("alice", model="ordinary-model")
    assert ordinary.returncode == 2, ordinary.stdout + ordinary.stderr
    assert tree_bytes(control_directory(root)) == before_control
    assert tree_bytes(root) == before_profile


def test_revoked_authority_cannot_prove_a_replaced_profile_is_safe(public_cli, tmp_path):
    root, _, _, _, launch = public_cli
    owner = automatic_owner(root)
    owner.set_enabled(False)
    owner.close()
    (root / "alice").rename(tmp_path / "retired-alice")
    (root / "alice").mkdir()
    before_control, before_profile = tree_bytes(control_directory(root)), tree_bytes(root)
    ordinary = launch("alice", model="ordinary-model", flags=("--no-skills",))
    assert ordinary.returncode == 2, ordinary.stdout + ordinary.stderr
    assert tree_bytes(control_directory(root)) == before_control
    assert tree_bytes(root) == before_profile
