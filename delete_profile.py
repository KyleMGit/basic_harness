"""Recoverably remove one named profile from active discovery."""

import argparse
import sys

import profile_paths
from profile_paths import validate_profile_id
from profile_maintenance import (MaintenanceResult, apply_moves, delete_moves,
                                 locked_profiles, one_profile, selected_root)


def delete_profile(profile, profiles_dir=None, *, apply=False):
    validate_profile_id(profile)
    root_pin = selected_root(profiles_dir)
    selected = one_profile(root_pin, profile)
    moves = delete_moves(root_pin, selected)
    if not apply:
        return MaintenanceResult(False, len(moves), (profile,))

    with locked_profiles(root_pin, (selected,)):
        root_pin.validate()
        current = one_profile(root_pin, profile)
        if current.store_id != selected.store_id:
            raise ValueError("Profile identity changed during maintenance preflight")
        moves = delete_moves(root_pin, current)
        backup = apply_moves(root_pin, moves, "delete-" + profile)
    return MaintenanceResult(True, len(moves), (profile,), backup)


def _parser():
    parser = argparse.ArgumentParser(
        description="Recoverably archive one named profile and its current discovery authorization.",
        epilog=("OFFLINE ONLY: stop every agent owner and skill review service in both the old and "
                "current deployment. Dry-run is the default; --apply performs the archive. "
                "Workspaces and dormant old control directories are never touched."),
    )
    parser.add_argument("profile", help="Exact profile name")
    parser.add_argument(
        "--profiles-dir",
        help=f"Profile root (default: {profile_paths.DEFAULT_PROFILES_DIR})",
    )
    parser.add_argument("--apply", action="store_true", help="Perform the recoverable archive")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        result = delete_profile(args.profile, args.profiles_dir, apply=args.apply)
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        print(f"Profile deletion refused: {exc}", file=sys.stderr)
        return 2
    if not result.applied:
        print(f"DRY RUN: would archive profile {args.profile} and {result.changed - 1} current authorization records.")
        print("Stop ALL old/current agent and review-service processes, then rerun with --apply.")
    else:
        print(f"Archived profile {args.profile} and its current discovery records to: {result.backup_path}")
        print("Workspace and dormant old control directories were left untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
