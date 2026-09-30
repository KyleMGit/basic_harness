"""Offline pruning of SQL-agent session history by last recorded activity."""

import argparse
from contextlib import closing
from dataclasses import dataclass
import math
import os
from pathlib import Path
import sqlite3
import sys
import time
import uuid

import profile_paths
from profile_maintenance import locked_profiles, one_profile, selected_root
from profile_paths import PinnedDirectory, plain_stat, validate_profile_id


SIDECARS = ("history.db-journal", "history.db-wal", "history.db-shm")
REQUIRED_COLUMNS = {
    "sessions": {"session_id", "start_time", "task", "status", "system_prompt"},
    "steps": {"id", "session_id", "step_index", "role", "content", "tool_calls", "tool_call_id", "timestamp"},
    "session_state": {"session_id", "messages_json", "step_counter", "context_state_json", "updated_at"},
}


@dataclass(frozen=True)
class ProfilePrune:
    profile: str
    candidates: int
    deleted: int
    unknown_time: int
    missing: bool
    backup_path: Path | None = None


@dataclass(frozen=True)
class PruneResult:
    applied: bool
    reports: tuple[ProfilePrune, ...]

    @property
    def candidates(self):
        return sum(item.candidates for item in self.reports)

    @property
    def deleted(self):
        return sum(item.deleted for item in self.reports)

    @property
    def unknown_time(self):
        return sum(item.unknown_time for item in self.reports)

    @property
    def missing(self):
        return sum(item.missing for item in self.reports)

    @property
    def backups(self):
        return tuple(item.backup_path for item in self.reports if item.backup_path is not None)


def _uri(path, mode):
    return path.as_uri() + f"?mode={mode}"


def _connect(path, mode):
    return sqlite3.connect(_uri(path, mode), uri=True)


def _validated_database(profile):
    profile.paths.validate()
    db = profile.paths.directory.path / "history.db"
    if not os.path.lexists(db):
        return None
    plain_stat(db)
    for name in SIDECARS:
        sidecar = profile.paths.directory.path / name
        if os.path.lexists(sidecar):
            plain_stat(sidecar)
    return db


def _profiles(root_pin):
    """Use maintenance discovery while explicitly excluding the legacy root DB."""
    with os.scandir(root_pin.path) as children:
        names = sorted(child.name for child in children if child.name != ".agent_history.db")
    found = []
    for name in names:
        validate_profile_id(name)
        found.append(one_profile(root_pin, name))
    return tuple(found)


def _schema(conn, db):
    tables = {
        row[0] for row in conn.execute(
            "SELECT name FROM sqlite_schema WHERE type='table'"
        )
    }
    for table in ("sessions", "steps"):
        if table not in tables:
            raise ValueError(f"Incompatible history schema in {db}: missing {table}")
    has_state = "session_state" in tables
    for table in ("sessions", "steps") + (("session_state",) if has_state else ()):
        columns = {row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')}
        missing = REQUIRED_COLUMNS[table] - columns
        if missing:
            raise ValueError(
                f"Incompatible history schema in {db}: {table} missing {', '.join(sorted(missing))}"
            )
    integrity = conn.execute("PRAGMA integrity_check").fetchone()
    if integrity != ("ok",):
        raise ValueError(f"Incompatible or corrupt history database: {db}")
    return has_state


def _usable(value):
    return (type(value) in (int, float) and math.isfinite(value))


def _selection(conn, cutoff, has_state):
    conn.create_function("prune_valid_epoch", 1, _usable)
    state_sql = (
        "(SELECT ss.updated_at FROM session_state ss WHERE ss.session_id=s.session_id)"
        if has_state else "NULL"
    )
    rows = conn.execute(
        "SELECT s.session_id, s.start_time, "
        "(SELECT MAX(st.timestamp) FROM steps st WHERE st.session_id=s.session_id), "
        f"{state_sql}, "
        "EXISTS(SELECT 1 FROM steps bad WHERE bad.session_id=s.session_id "
        "AND bad.timestamp IS NOT NULL AND NOT prune_valid_epoch(bad.timestamp)) "
        "FROM sessions s"
    )
    candidates = []
    unknown = 0
    for session_id, start, step_time, state_time, malformed_step in rows:
        values = (start, step_time, state_time)
        malformed = bool(malformed_step) or any(
            value is not None and not _usable(value) for value in values
        )
        valid = [float(value) for value in values if _usable(value)]
        if malformed or not valid:
            unknown += 1
        elif max(valid) < cutoff:
            candidates.append(session_id)
    return tuple(candidates), unknown


def _preflight(profile, cutoff):
    db = _validated_database(profile)
    if db is None:
        return db, False, (), 0
    with closing(_connect(db, "ro")) as conn:
        has_state = _schema(conn, db)
        selected, unknown = _selection(conn, cutoff, has_state)
    return db, has_state, selected, unknown


def _backup_parent(root_pin):
    parent = root_pin.path.parent
    PinnedDirectory(parent).validate()
    destination = parent / ".profile-maintenance-backups"
    if os.path.lexists(destination):
        PinnedDirectory(destination).validate()
    if destination == root_pin.path or destination.is_relative_to(root_pin.path):
        raise ValueError("Backup location overlaps the profile discovery root")
    return destination


def _backup_database(source, root_pin, profile_name):
    parent = _backup_parent(root_pin)
    parent.mkdir(exist_ok=True)
    run = parent / f"prune-history-{uuid.uuid4().hex}"
    run.mkdir()
    destination_dir = run / profile_name
    destination_dir.mkdir()
    destination = destination_dir / "history.db"
    try:
        with closing(_connect(source, "ro")) as original, closing(sqlite3.connect(destination)) as backup:
            original.backup(backup)
        plain_stat(destination)
        return destination
    except Exception:
        if destination.exists():
            destination.unlink()
        destination_dir.rmdir()
        run.rmdir()
        try:
            parent.rmdir()
        except OSError:
            pass
        raise


def _delete(db, cutoff, expected_has_state):
    conn = _connect(db, "rw")
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("BEGIN IMMEDIATE")
        has_state = _schema(conn, db)
        if has_state != expected_has_state:
            raise ValueError(f"History schema changed during maintenance preflight: {db}")
        selected, unknown = _selection(conn, cutoff, has_state)
        tables = ("steps",) + (("session_state",) if has_state else ()) + ("sessions",)
        if selected:
            conn.execute("CREATE TEMP TABLE prune_selected (session_id TEXT PRIMARY KEY)")
            conn.executemany("INSERT INTO prune_selected VALUES (?)", ((value,) for value in selected))
            for table in tables:
                conn.execute(f"DELETE FROM {table} WHERE session_id IN (SELECT session_id FROM temp.prune_selected)")
            remaining = sum(conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE session_id IN (SELECT session_id FROM temp.prune_selected)"
            ).fetchone()[0] for table in tables)
            if remaining:
                raise RuntimeError("History deletion verification failed; transaction rolled back")
        conn.commit()
        if selected:
            remaining = sum(conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE session_id IN (SELECT session_id FROM temp.prune_selected)"
            ).fetchone()[0] for table in tables)
            if remaining:
                raise RuntimeError("Committed history deletion failed readback verification")
        return selected, unknown
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def prune_history(days, profiles_dir=None, *, apply=False, now=None, on_backup=None):
    if type(days) is not int or days <= 0:
        raise ValueError("--days must be a positive integer")
    current_time = time.time() if now is None else now
    if not _usable(current_time):
        raise ValueError("Current time must be a finite epoch timestamp")
    cutoff = float(current_time) - days * 86_400
    root_pin = selected_root(profiles_dir)
    profiles = _profiles(root_pin)
    preflight = [_preflight(profile, cutoff) for profile in profiles]
    if not apply:
        return PruneResult(False, tuple(
            ProfilePrune(profile.name, len(item[2]), 0, item[3], item[0] is None)
            for profile, item in zip(profiles, preflight)
        ))
    reports = []
    with locked_profiles(root_pin, profiles):
        root_pin.validate()
        current = _profiles(root_pin)
        if [(p.name, p.store_id) for p in current] != [(p.name, p.store_id) for p in profiles]:
            raise ValueError("Profile set or identity changed during maintenance preflight")
        # Revalidate every database before the first mutation.
        current_preflight = [_preflight(profile, cutoff) for profile in current]
        for profile, item in zip(current, current_preflight):
            db, has_state, selected, unknown = item
            if db is None or not selected:
                reports.append(ProfilePrune(profile.name, len(selected), 0, unknown, db is None))
                continue
            backup = _backup_database(db, root_pin, profile.name)
            if on_backup is not None:
                on_backup(profile.name, backup)
            deleted, final_unknown = _delete(db, cutoff, has_state)
            reports.append(ProfilePrune(
                profile.name, len(deleted), len(deleted), final_unknown, False, backup
            ))
    return PruneResult(True, tuple(reports))


def _positive_integer(value):
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a positive integer") from None
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _parser():
    parser = argparse.ArgumentParser(
        description="Prune SQL-agent sessions strictly older than a last-activity cutoff.",
        epilog=("OFFLINE ONLY: stop every agent and review process first. Dry-run is the "
                "default; --apply backs up each changing database before deletion."),
    )
    parser.add_argument("--days", required=True, type=_positive_integer)
    parser.add_argument(
        "--profiles-dir", help=f"Profile root (default: {profile_paths.DEFAULT_PROFILES_DIR})"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="Back up and delete candidates")
    mode.add_argument("--dry-run", action="store_true", help="Explicitly select read-only preview")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        result = prune_history(
            args.days, args.profiles_dir, apply=args.apply,
            on_backup=lambda name, path: print(f"{name}: backup {path}", file=sys.stderr, flush=True),
        )
    except (OSError, ValueError, RuntimeError, sqlite3.DatabaseError) as exc:
        print(f"History pruning failed: {exc}", file=sys.stderr)
        if args.apply:
            print("Earlier profiles may already be pruned; retain the reported backups for recovery.", file=sys.stderr)
        return 2
    prefix = "APPLY" if result.applied else "DRY RUN"
    for report in result.reports:
        if report.missing:
            print(f"{report.profile}: missing history.db; skipped")
        else:
            action = "deleted" if result.applied else "candidates"
            count = report.deleted if result.applied else report.candidates
            print(f"{report.profile}: {count} {action}; {report.unknown_time} unknown-time retained")

    print(f"{prefix}: {result.candidates} candidates; {result.deleted} deleted; "
          f"{result.unknown_time} unknown-time retained; {result.missing} missing databases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
