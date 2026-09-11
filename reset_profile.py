"""Remove stale discovery authorization for an already-deleted profile."""

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import re
import sys
import uuid

import profile_paths
from profile_paths import PinnedDirectory, absolute_path, control_directory, plain_stat, validate_profile_id
from skill_lock import CatalogLock, ProcessLock, path_identity


def _authority(path, expected_control):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Malformed discovery authority: {exc}") from None
    if (not isinstance(value, dict) or set(value) != {"origin", "generation"}
            or not isinstance(value["generation"], str)
            or not re.fullmatch(r"[a-f0-9]{32}", value["generation"])):
        raise ValueError("Malformed discovery authority")
    origin = value["origin"]
    if (not isinstance(origin, dict)
            or set(origin) != {"mode", "control_dir", "model", "base_url"}
            or not all(isinstance(item, str) and item for item in origin.values())):
        raise ValueError("Malformed discovery authority origin")
    if origin["mode"] != "discovery":
        raise ValueError("Static/custom authority cannot be reset by this command")
    if origin["control_dir"] != str(expected_control):
        raise ValueError("Discovery authority control directory mismatch")
    if not re.match(r"^https?://", origin["base_url"]):
        raise ValueError("Malformed discovery authority origin")


def _backup_path(path):
    while True:
        candidate = path.with_name(path.name + ".reset-backup-" + uuid.uuid4().hex)
        if not os.path.lexists(candidate):
            return candidate


def reset_profile(profile, profiles_dir=None):
    """Back up the exact stale auth pair; never delete or recreate a profile."""
    validate_profile_id(profile)
    root = absolute_path(profile_paths.DEFAULT_PROFILES_DIR if profiles_dir is None else profiles_dir)
    root_pin = PinnedDirectory(root)
    root_pin.validate(required=False)

    profile_dir = root / profile
    if os.path.lexists(profile_dir):
        # Inspect it so aliases/reparse points also fail through shared validation.
        try:
            plain_stat(profile_dir, directory=True)
        except (OSError, ValueError):
            pass
        raise ValueError(f"Profile directory still exists: {profile_dir}")

    store_id = path_identity(profile_dir / "skills")
    control = control_directory(root)
    auth = control / f"{store_id}.json"
    authority = control / f"{store_id}.authority.json"
    present = [path for path in (auth, authority) if os.path.lexists(path)]
    if not present:
        return 0

    # Pin and validate every existing ancestor and every source before mutation.
    PinnedDirectory(control).validate()
    for path in present:
        plain_stat(path)
    if authority in present:
        _authority(authority, control)
    else:
        raise ValueError("Authorization exists without discovery authority; refusing reset")

    destinations = {path: _backup_path(path) for path in present}
    moved = []
    with ExitStack() as locks:
        try:
            locks.enter_context(ProcessLock("skill-owner:" + store_id, timeout=0))
            locks.enter_context(CatalogLock(identity=store_id, timeout=0))
        except TimeoutError as exc:
            raise RuntimeError(
                "Reset refused because an old agent for this profile is still running; stop it first"
            ) from exc
        # Revalidate all source paths after acquiring the exclusion locks.
        root_pin.validate(required=False)
        if os.path.lexists(profile_dir):
            raise ValueError(f"Profile directory still exists: {profile_dir}")
        for path in present:
            plain_stat(path)
        try:
            for path in present:
                os.rename(path, destinations[path])
                moved.append(path)
        except OSError as exc:
            rollback_errors = []
            for path in reversed(moved):
                try:
                    os.rename(destinations[path], path)
                except OSError as rollback_exc:
                    rollback_errors.append(f"{destinations[path]} remains backed up: {rollback_exc}")
            if rollback_errors:
                raise RuntimeError(
                    f"Reset rename failed ({exc}); rollback incomplete; " + "; ".join(rollback_errors)
                ) from None
            raise RuntimeError(f"Reset rename failed ({exc}); prior rename rolled back") from None
    return len(moved)


def _parser():
    parser = argparse.ArgumentParser(
        description="Back up stale discovery authorization after a profile was deleted for same-name recreation.",
        epilog="Before running, stop the old agent and the shared skill review service. This command never deletes a profile or workspace.",
    )
    parser.add_argument("profile", help="Deleted profile name")
    parser.add_argument("--profiles-dir", help="Profile root (default: profile_paths.DEFAULT_PROFILES_DIR)")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        moved = reset_profile(args.profile, args.profiles_dir)
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        print(f"Profile reset refused: {exc}", file=sys.stderr)
        return 2
    if moved:
        print(f"Backed up {moved} stale authorization records for deleted profile {args.profile}.")
    else:
        print(f"No stale authorization records found for deleted profile {args.profile}; nothing changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
