import os
from pathlib import Path
import sqlite3

import pytest

from skill_lock import ProcessLock


NOW = 2_000_000_000.0
DAY = 86_400


def make_profile(root, name, *, state=True):
    profile = root / name
    profile.mkdir(parents=True)
    db = profile / "history.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY, start_time REAL, task TEXT, status TEXT, system_prompt TEXT)")
        conn.execute("CREATE TABLE steps (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, step_index INTEGER, role TEXT, content TEXT, tool_calls TEXT, tool_call_id TEXT, timestamp REAL, FOREIGN KEY(session_id) REFERENCES sessions(session_id))")
        if state:
            conn.execute("CREATE TABLE session_state (session_id TEXT PRIMARY KEY, messages_json TEXT, step_counter INTEGER, context_state_json TEXT, updated_at REAL, FOREIGN KEY(session_id) REFERENCES sessions(session_id))")
    return profile, db


def add_session(db, session_id, start, *, status="COMPLETED", steps=(), state_marker=False, updated=None):
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?)",
            (session_id, start, "secret task", status, "secret prompt"),
        )
        for index, timestamp in enumerate(steps):
            conn.execute(
                "INSERT INTO steps (session_id, step_index, role, content, tool_calls, tool_call_id, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session_id, index, "user", "secret content", None, None, timestamp),
            )
        if state_marker:
            conn.execute(
                "INSERT INTO session_state VALUES (?, ?, ?, ?, ?)",
                (session_id, '[{"secret":"message"}]', 1, '{"secret":"context"}', updated),
            )


def ids(db, table="sessions"):
    with sqlite3.connect(db) as conn:
        return {row[0] for row in conn.execute(f"SELECT session_id FROM {table}")}


def backup_dbs(root):
    parent = root.parent / ".profile-maintenance-backups"
    return list(parent.rglob("*.db")) if parent.exists() else []


def test_multi_profile_activity_cutoff_unknown_and_malformed_are_conservative(tmp_path):
    import prune_history

    root = tmp_path / "profiles"
    _, alice = make_profile(root, "alice")
    _, bob = make_profile(root, "bob")
    cutoff = NOW - 30 * DAY
    add_session(alice, "old", cutoff - 1, status="IN_PROGRESS")
    add_session(alice, "boundary", cutoff)
    add_session(alice, "recent-step", cutoff - 100, steps=[cutoff + 1])
    add_session(alice, "recent-state", cutoff - 100, state_marker=True, updated=cutoff + 1)
    add_session(alice, "unknown", None, steps=[None], state_marker=True, updated=None)
    add_session(alice, "malformed", "not-an-epoch")
    add_session(alice, "future", NOW + DAY)
    add_session(bob, "bob-old", cutoff - 50)

    result = prune_history.prune_history(30, root, now=NOW)

    assert not result.applied
    assert result.candidates == 2
    assert result.unknown_time == 2
    assert ids(alice) == {"old", "boundary", "recent-step", "recent-state", "unknown", "malformed", "future"}
    assert ids(bob) == {"bob-old"}


def test_apply_deletes_children_backs_up_recovers_and_is_idempotent(tmp_path):
    import prune_history

    root = tmp_path / "profiles"
    _, db = make_profile(root, "alice")
    old = NOW - 31 * DAY
    add_session(db, "old", old, steps=[old], state_marker=True, updated=old)
    add_session(db, "keep", NOW)

    result = prune_history.prune_history(30, root, apply=True, now=NOW)
    assert result.applied and result.deleted == 1
    assert result.backups and result.backups[0].is_file()
    assert ids(db) == {"keep"}
    assert ids(db, "steps") == set() and ids(db, "session_state") == set()
    with sqlite3.connect(db) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    assert ids(result.backups[0]) == {"old", "keep"}

    before = set(backup_dbs(root))
    again = prune_history.prune_history(30, root, apply=True, now=NOW)
    assert again.deleted == 0 and again.backups == ()
    assert set(backup_dbs(root)) == before


def test_dry_run_is_read_only_and_creates_no_backup(tmp_path):
    import prune_history

    root = tmp_path / "profiles"
    _, db = make_profile(root, "alice")
    add_session(db, "old", NOW - 40 * DAY)
    before = db.read_bytes()
    result = prune_history.prune_history(30, root, now=NOW)
    assert result.candidates == 1 and db.read_bytes() == before
    assert backup_dbs(root) == []


def test_legacy_database_without_session_state(tmp_path):
    import prune_history

    root = tmp_path / "profiles"
    _, db = make_profile(root, "alice", state=False)
    add_session(db, "old", NOW - 40 * DAY)
    result = prune_history.prune_history(30, root, apply=True, now=NOW)
    assert result.deleted == 1 and ids(db) == set()


def test_missing_history_and_empty_root_do_not_create_history(tmp_path):
    import prune_history

    root = tmp_path / "profiles"
    (root / "alice").mkdir(parents=True)
    result = prune_history.prune_history(30, root, now=NOW)
    assert result.missing == 1 and not (root / "alice" / "history.db").exists()
    empty = tmp_path / "empty"
    empty.mkdir()
    assert prune_history.prune_history(30, empty, apply=True, now=NOW).deleted == 0
    assert not list(empty.rglob("history.db"))


def test_does_not_touch_legacy_root_or_recursive_workspace_db(tmp_path):
    import prune_history

    root = tmp_path / "profiles"
    _, db = make_profile(root, "alice")
    add_session(db, "keep", NOW)
    legacy = root / ".agent_history.db"
    legacy.write_bytes(b"legacy")
    nested = root / "alice" / "workspace" / "history.db"
    nested.parent.mkdir()
    nested.write_bytes(b"nested")
    prune_history.prune_history(30, root, apply=True, now=NOW)
    assert legacy.read_bytes() == b"legacy" and nested.read_bytes() == b"nested"


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "3"])
def test_api_rejects_non_positive_integer_days(tmp_path, bad):
    import prune_history

    root = tmp_path / "profiles"
    root.mkdir()
    with pytest.raises(ValueError, match="positive integer"):
        prune_history.prune_history(bad, root)


def test_cli_rejects_invalid_days_and_modes(capsys):
    import prune_history

    with pytest.raises(SystemExit):
        prune_history.main(["--days", "0"])
    with pytest.raises(SystemExit):
        prune_history.main(["--days", "3", "--apply", "--dry-run"])


def test_apply_refuses_incompatible_schema_before_changing_any_db(tmp_path):
    import prune_history

    root = tmp_path / "profiles"
    _, good = make_profile(root, "alice")
    add_session(good, "old", NOW - 40 * DAY)
    bad_profile = root / "bob"
    bad_profile.mkdir()
    with sqlite3.connect(bad_profile / "history.db") as conn:
        conn.execute("CREATE TABLE sessions (wrong TEXT)")
    before = good.read_bytes()
    with pytest.raises(ValueError, match="schema"):
        prune_history.prune_history(30, root, apply=True, now=NOW)
    assert good.read_bytes() == before and backup_dbs(root) == []


@pytest.mark.parametrize("has_candidate", [True, False])
def test_lock_refusal_does_not_mutate_or_backup(tmp_path, has_candidate):
    import prune_history
    import profile_maintenance

    root = tmp_path / "profiles"
    _, db = make_profile(root, "alice")
    timestamp = NOW - 40 * DAY if has_candidate else NOW
    add_session(db, "session", timestamp)
    root_pin = profile_maintenance.selected_root(root)
    store_id = profile_maintenance.all_profiles(root_pin)[0].store_id
    with ProcessLock("skill-owner:" + store_id, timeout=0):
        with pytest.raises(RuntimeError, match="still running"):
            prune_history.prune_history(30, root, apply=True, now=NOW)
    assert ids(db) == {"session"} and backup_dbs(root) == []


def test_delete_failure_rolls_back_database(tmp_path):
    import prune_history

    root = tmp_path / "profiles"
    _, db = make_profile(root, "alice")
    add_session(db, "old", NOW - 40 * DAY, steps=[NOW - 40 * DAY])
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TRIGGER refuse_session_delete BEFORE DELETE ON sessions BEGIN SELECT RAISE(ABORT, 'refuse'); END")
    with pytest.raises(sqlite3.DatabaseError, match="refuse"):
        prune_history.prune_history(30, root, apply=True, now=NOW)
    assert ids(db) == {"old"} and ids(db, "steps") == {"old"}


def test_hardlinked_database_and_sidecar_are_refused(tmp_path):
    import prune_history

    root = tmp_path / "profiles"
    profile, db = make_profile(root, "alice")
    alias = tmp_path / "alias.db"
    try:
        os.link(db, alias)
    except OSError:
        pytest.skip("hard links unavailable")
    with pytest.raises(ValueError, match="aliases"):
        prune_history.prune_history(30, root)

    alias.unlink()
    sidecar = profile / "history.db-journal"
    sidecar.write_bytes(b"journal")
    side_alias = tmp_path / "sidecar-alias"
    os.link(sidecar, side_alias)
    with pytest.raises(ValueError, match="aliases"):
        prune_history.prune_history(30, root)
