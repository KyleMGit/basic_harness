"""Recoverably reset path-bound review state for all discovered profiles."""

import argparse
import sys

import profile_paths
from profile_maintenance import (MaintenanceResult, all_profiles, apply_moves,
                                 locked_profiles, reset_moves, selected_root)


def reset_profile_locks(profiles_dir=None, *, apply=False):
    root_pin = selected_root(profiles_dir)
    profiles = all_profiles(root_pin)
    moves = reset_moves(root_pin, profiles)
    if not apply:
        return MaintenanceResult(False, len(moves), tuple(p.name for p in profiles))
    if not moves:
        return MaintenanceResult(True, 0, tuple(p.name for p in profiles))

    with locked_profiles(root_pin, profiles):
        root_pin.validate()
        current = all_profiles(root_pin)
        if [(p.name, p.store_id) for p in current] != [(p.name, p.store_id) for p in profiles]:
            raise ValueError("Profile set or identity changed during maintenance preflight")
        moves = reset_moves(root_pin, current)
        backup = apply_moves(root_pin, moves, "reset-all")
    return MaintenanceResult(True, len(moves), tuple(p.name for p in profiles), backup)


def _parser():
    parser = argparse.ArgumentParser(
        description="Archive all current profile review mailboxes and discovery authorization records.",
        epilog=("OFFLINE ONLY: stop every agent owner and skill review service in both the old and "
                "current deployment. Dry-run is the default; --apply performs the reset. "
                "Skills, memories, history, logs, workspaces, OS locks, policy.json, and dormant old "
                "control directories are never removed."),
    )
    parser.add_argument(
        "--profiles-dir",
        help=f"Profile root (default: {profile_paths.DEFAULT_PROFILES_DIR})",
    )
    parser.add_argument("--apply", action="store_true", help="Perform the recoverable archive")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        result = reset_profile_locks(args.profiles_dir, apply=args.apply)
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        print(f"Profile lock reset refused: {exc}", file=sys.stderr)
        return 2
    if not result.applied:
        print(f"DRY RUN: {len(result.profiles)} profiles; would archive {result.changed} review-state files.")
        print("Stop ALL old/current agent and review-service processes, then rerun with --apply.")
    elif result.backup_path is None:
        print("No current mailbox or discovery authorization records found; nothing changed.")
    else:
        print(f"Archived {result.changed} review-state files to: {result.backup_path}")
        print("Skills, memories, history, logs, workspaces, and dormant old controls were left untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

