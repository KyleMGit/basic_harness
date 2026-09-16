# Offline profile maintenance

These commands are for a stopped installation. Before even a dry run, stop **all**
agent processes and the skill-review service from both the old and current
deployment. Keep the old deployment stopped. A non-learning agent may not hold a
lifetime owner lock, so a successful lock check is not proof that every process is
stopped. Neither command kills processes.

Both commands default to the `.agent_profiles` directory beside the scripts,
independent of the launch directory. Supplying a missing `--profiles-dir` is an
error. Both are dry-run by default:

```powershell
# Preview, then recoverably remove one profile from active discovery.
python .\delete_profile.py alice --profiles-dir C:\host\profiles
python .\delete_profile.py alice --profiles-dir C:\host\profiles --apply

# Preview, then reset path-bound review state for every immediate profile.
python .\reset_profile_locks.py --profiles-dir C:\host\profiles
python .\reset_profile_locks.py --profiles-dir C:\host\profiles --apply
```

`delete_profile.py` archives the exact named profile directory and its exact
current discovery authorization pair. It does not touch that profile's workspace,
so recreating the same profile name starts with fresh profile state but retains the
workspace.

`reset_profile_locks.py` archives each profile's `skill_review.db` and SQLite
`-wal`, `-shm`, and `-journal` sidecars together with its exact current discovery
authorization pair when present. It preserves skills, memories, history, logs,
workspaces, `policy.json`, and unrelated authorization records. Pending review work
is abandoned in the recoverable backup. Reset does not enable learning; a later
owner started with `--auto-skills` creates fresh credentials normally.

Successful changes are renamed into a unique directory under
`<profiles-parent>/.profile-maintenance-backups/`, outside the discovery root. The
command prints the exact backup path. A failed multi-file rename is rolled back;
if rollback itself fails, the error reports every item that remains in the backup.
Restore only while the installation is fully stopped, by moving the reported
backup items back to their original profile/control paths.

The scripts coordinate with the known service, owner, catalog, and discovery-policy
locks and refuse immediately when one is held. They never delete OS lock files:
Windows mutexes and POSIX `flock` locks are released by the owning process when it
exits. Static/custom-roster authority is intentionally unsupported and is refused
before any profile is changed.

## Moving an installation or profile root

Copy the application files and the complete profile root, including hidden files
and SQLite sidecars. Copy the independent workspace root too if those workspaces
must move. Transfer deployment configuration and secrets through the server's
normal protected configuration process, not through these maintenance commands.

A copied `.skill-review-*` control directory is bound to the old absolute profile
path. Leave it dormant and do not rename it to the new root's computed control
name. `reset_profile_locks.py --apply` touches only the new root's exact computed
records and the mailboxes inside its immediate profiles; sibling controls for the
old path are reported as untouched. After the reset succeeds, start only the new
deployment. Remove old backups or dormant controls later only under a separate,
explicit retention procedure.

