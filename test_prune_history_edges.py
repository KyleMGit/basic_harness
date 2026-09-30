"""Independent adversarial regression coverage for history maintenance."""
from contextlib import closing
import sqlite3

import pytest
import prune_history
from test_prune_history import DAY, NOW, make_profile, add_session, ids


def test_nonfinite_step_cannot_be_hidden_by_max(tmp_path):
    root = tmp_path / 'profiles'
    _, db = make_profile(root, 'alice')
    add_session(db, 'invalid', NOW - 40 * DAY,
                steps=[float('-inf'), NOW - 40 * DAY])
    result = prune_history.prune_history(30, root, now=NOW)
    assert result.candidates == 0
    assert result.unknown_time == 1


def test_dry_run_closes_all_database_connections(tmp_path, monkeypatch):
    root = tmp_path / 'profiles'
    make_profile(root, 'alice')
    real_connect = prune_history._connect
    opened = []

    def tracked(*args):
        conn = real_connect(*args)
        opened.append(conn)
        return conn

    monkeypatch.setattr(prune_history, '_connect', tracked)
    prune_history.prune_history(30, root, now=NOW)
    try:
        for conn in opened:
            with pytest.raises(sqlite3.ProgrammingError, match='closed'):
                conn.execute('SELECT 1')
    finally:
        for conn in opened:
            conn.close()


def test_delete_verification_does_not_exceed_sqlite_variable_limit(tmp_path, monkeypatch):
    root = tmp_path / 'profiles'
    _, db = make_profile(root, 'alice')
    with closing(sqlite3.connect(db)) as conn:
        conn.executemany('INSERT INTO sessions(session_id,start_time) VALUES (?,?)',
                         [(str(i), NOW - 40 * DAY) for i in range(60)])
        conn.commit()
    real_connect = prune_history._connect

    def limited(*args):
        conn = real_connect(*args)
        conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 50)
        return conn

    monkeypatch.setattr(prune_history, '_connect', limited)
    deleted, unknown = prune_history._delete(db, NOW - 30 * DAY, True)
    assert len(deleted) == 60 and unknown == 0
    assert ids(db) == set()


def test_cli_failure_reports_recoverable_backups_for_partial_progress(tmp_path, capsys):
    root = tmp_path / 'profiles'
    _, alice = make_profile(root, 'alice')
    _, bob = make_profile(root, 'bob')
    add_session(alice, 'old', 1)
    add_session(bob, 'old', 1)
    with closing(sqlite3.connect(bob)) as conn:
        conn.execute("CREATE TRIGGER refuse BEFORE DELETE ON sessions BEGIN SELECT RAISE(ABORT, 'refuse'); END")
        conn.commit()
    assert prune_history.main(['--days', '30', '--profiles-dir', str(root), '--apply']) == 2
    error = capsys.readouterr().err
    backups = list((root.parent / '.profile-maintenance-backups').rglob('history.db'))
    assert len(backups) == 2
    assert all(str(path) in error for path in backups)
    assert ids(alice) == set() and ids(bob) == {'old'}
