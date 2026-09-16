"""Offline, recoverable profile maintenance shared by the public CLIs.

Only paths derived from the selected discovery root are inspected.  In
particular, authority JSON never supplies a path that maintenance follows.
"""

from contextlib import ExitStack
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import uuid

import profile_paths
from profile_paths import (PinnedDirectory, ProfilePaths, absolute_path,
                           control_directory, plain_stat, validate_profile_id)
from skill_lock import CatalogLock, ProcessLock, path_identity


MAILBOX_NAMES = (
    "skill_review.db",
    "skill_review.db-wal",
    "skill_review.db-shm",
    "skill_review.db-journal",
)
SERVICE_LOCK_ID = "hermes-skill-review-service"


@dataclass(frozen=True)
class MaintenanceResult:
    applied: bool
    changed: int
    profiles: tuple[str, ...]
    backup_path: Path | None = None


@dataclass(frozen=True)
class _Profile:
    name: str
    paths: ProfilePaths
    store_id: str


@dataclass(frozen=True)
class _Move:
    source: Path
    relative_destination: Path


def selected_root(profiles_dir=None):
    root = absolute_path(
        profile_paths.DEFAULT_PROFILES_DIR if profiles_dir is None else profiles_dir
    )
    if root == root.parent:
        raise ValueError("Profile root cannot be a filesystem root")
    pin = PinnedDirectory(root)
    try:
        pin.validate()
    except FileNotFoundError:
        raise ValueError(f"Profile root does not exist: {root}") from None
    except ValueError as exc:
        if "does not exist" in str(exc):
            raise ValueError(f"Profile root does not exist: {root}") from None
        raise
    return pin


def one_profile(root_pin, name):
    validate_profile_id(name)
    paths = ProfilePaths(root_pin, name)
    if not paths.validate(required=False):
        raise ValueError(f"Named profile does not exist: {paths.directory.path}")
    return _Profile(name, paths, path_identity(paths.directory.path / "skills"))


def all_profiles(root_pin):
    found = []
    with os.scandir(root_pin.path) as children:
        names = sorted(child.name for child in children)
    for name in names:
        # Every immediate child must be an unambiguous valid profile.  Silently
        # skipping an alias during a destructive all-profile operation is unsafe.
        validate_profile_id(name)
        found.append(one_profile(root_pin, name))
    return tuple(found)


def _read_json(path, label):
    plain_stat(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Malformed {label}: {exc}") from None


def _records(root, profile):
    """Return only the exact current discovery pair after validating it."""
    control = control_directory(root)
    auth = control / f"{profile.store_id}.json"
    authority = control / f"{profile.store_id}.authority.json"
    auth_present = os.path.lexists(auth)
    authority_present = os.path.lexists(authority)
    if not auth_present and not authority_present:
        return ()
    if auth_present != authority_present:
        raise ValueError(
            f"Incomplete current discovery authorization pair for profile {profile.name}"
        )

    # Validate the computed control itself before reading either fixed filename.
    PinnedDirectory(control).validate()
    authority_value = _read_json(authority, "discovery authority")
    if (not isinstance(authority_value, dict)
            or set(authority_value) != {"origin", "generation"}
            or not isinstance(authority_value["generation"], str)
            or re.fullmatch(r"[a-f0-9]{32}", authority_value["generation"]) is None):
        raise ValueError("Malformed discovery authority")
    origin = authority_value["origin"]
    if not isinstance(origin, dict) or origin.get("mode") != "discovery":
        raise ValueError(
            f"Static/custom roster authority is unsupported for profile {profile.name}"
        )
    if (set(origin) != {"mode", "control_dir", "model", "base_url"}
            or not all(isinstance(value, str) and value for value in origin.values())
            or origin["control_dir"] != str(control)
            or re.match(r"^https?://", origin["base_url"]) is None):
        raise ValueError(f"Malformed or non-current discovery authority for profile {profile.name}")

    auth_value = _read_json(auth, "discovery authorization")
    expected = (profile.name, profile.store_id, str(profile.paths.directory.path / "skill_review.db"))
    if (not isinstance(auth_value, dict)
            or (auth_value.get("profile_id"), auth_value.get("store_id"), auth_value.get("mailbox")) != expected
            or auth_value.get("authority_generation") != authority_value["generation"]
            or (auth_value.get("model"), auth_value.get("base_url")) != (origin["model"], origin["base_url"])
            or type(auth_value.get("enabled")) is not bool
            or re.fullmatch(r"[a-f0-9]{32}", str(auth_value.get("generation", ""))) is None
            or re.fullmatch(r"[a-f0-9]{64}", str(auth_value.get("secret", ""))) is None):
        raise ValueError(f"Current discovery authorization identity is invalid for profile {profile.name}")
    # An incarnation mismatch is deliberately allowed: copied directories have
    # new identities, and removing this stale record grants no authority.
    return (auth, authority)


def delete_moves(root_pin, profile):
    profile.paths.validate()
    moves = [_Move(profile.paths.directory.path, Path("profile"))]
    moves.extend(
        _Move(path, Path("control") / path.name)
        for path in _records(root_pin.path, profile)
    )
    return tuple(moves)


def reset_moves(root_pin, profiles):
    moves = []
    for profile in profiles:
        profile.paths.validate()
        for name in MAILBOX_NAMES:
            path = profile.paths.directory.path / name
            if os.path.lexists(path):
                plain_stat(path)
                moves.append(_Move(path, Path("profiles") / profile.name / name))
        moves.extend(
            _Move(path, Path("control") / path.name)
            for path in _records(root_pin.path, profile)
        )
    return tuple(moves)


def locked_profiles(root_pin, profiles):
    """Acquire the known cooperating-process locks in a fixed order."""
    stack = ExitStack()
    try:
        stack.enter_context(ProcessLock(SERVICE_LOCK_ID, timeout=0))
        stack.enter_context(CatalogLock(
            identity="discovery-policy:" + path_identity(root_pin.path), timeout=0
        ))
        for profile in sorted(profiles, key=lambda item: item.store_id):
            stack.enter_context(ProcessLock("skill-owner:" + profile.store_id, timeout=0))
            stack.enter_context(CatalogLock(identity=profile.store_id, timeout=0))
    except TimeoutError as exc:
        stack.close()
        raise RuntimeError(
            "Maintenance refused because an agent owner or review service is still running; "
            "stop every old and current process first"
        ) from exc
    return stack


def _backup_parent(root_pin):
    parent = root_pin.path.parent
    PinnedDirectory(parent).validate()
    backup_parent = parent / ".profile-maintenance-backups"
    if os.path.lexists(backup_parent):
        PinnedDirectory(backup_parent).validate()
    if backup_parent == root_pin.path or backup_parent.is_relative_to(root_pin.path):
        raise ValueError("Backup location overlaps the profile discovery root")
    return backup_parent


def apply_moves(root_pin, moves, label):
    if not moves:
        return None
    parent = _backup_parent(root_pin)
    backup = parent / f"{label}-{uuid.uuid4().hex}"
    destinations = [(move.source, backup / move.relative_destination) for move in moves]
    created_parent = not os.path.lexists(parent)
    try:
        parent.mkdir(exist_ok=True)
        backup.mkdir()
        for source, destination in destinations:
            if not os.path.lexists(source):
                raise ValueError(f"Maintenance source changed during preflight: {source}")
            if os.path.lexists(destination):
                raise ValueError(f"Backup destination unexpectedly exists: {destination}")
        for destination in {destination.parent for _, destination in destinations}:
            destination.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError):
        _cleanup_empty(backup, parent if created_parent else None)
        raise

    moved = []
    try:
        for source, destination in destinations:
            os.rename(source, destination)
            moved.append((source, destination))
    except OSError as exc:
        rollback_errors = []
        for source, destination in reversed(moved):
            try:
                os.rename(destination, source)
            except OSError as rollback_exc:
                rollback_errors.append(f"{destination} remains archived: {rollback_exc}")
        if not rollback_errors:
            _cleanup_empty(backup, parent if created_parent else None)
            raise RuntimeError(f"Maintenance rename failed ({exc}); prior rename rolled back") from None
        raise RuntimeError(
            f"Maintenance rename failed ({exc}); rollback incomplete; " + "; ".join(rollback_errors)
        ) from None
    return backup


def _cleanup_empty(backup, removable_parent=None):
    if backup.exists():
        for directory in sorted(
                (item for item in backup.rglob("*") if item.is_dir()),
                key=lambda item: len(item.parts), reverse=True):
            try:
                directory.rmdir()
            except OSError:
                pass
        try:
            backup.rmdir()
        except OSError:
            pass
    if removable_parent is not None:
        try:
            removable_parent.rmdir()
        except OSError:
            pass
