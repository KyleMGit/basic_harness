# Asynchronous private skill learning

Run one persistent `agent.py --profile ...` process per authenticated user and
**one supervised `skill_review.py run` process for the host**. Automatic learning
is opt-in. The owner has a lightweight preparation/delivery thread; only the
separate service schedules inference. Default capacity is one request, with a
fixed pool configurable from 1 to 8 at service startup. No agent starts a service.

Automatic mode derives profile IDs, private mailbox paths and immutable catalog
identities from one host-controlled profile root. Owners that enable learning and
the service use the same host model and endpoint; there is no smaller/second-model dependency. The service reads
sealed prepared snapshots and writes proposals into that user's mailbox. It never
opens, searches or mutates a catalog. No context, catalog, semantic deduplication
or learned state is shared between users. Only the bound owner validates and
publishes CREATE/UPDATE/NONE results.

These are application guarantees among cooperating processes, **not OS process
isolation or a login feature**. The host must authenticate users, select their
profile/root arguments, and protect the entire root, profile directories, any static roster, coordination
secrets and process access using OS accounts/ACLs. Use local storage. Multiple
Windows sessions/accounts must be permitted to coordinate through the same
`Global\HermesSkill-*` named mutexes. POSIX uses flock coordination files in the
OS temp directory. Automatic discovery changes the admission trust scope: any
eligible immediate child created by the trusted host can participate. Models
cannot register profiles, choose these arguments or access profiles/control through
workspace tools. Keep workspace and read-only tool roots separate; overlapping
roots are refused at named CLI startup. Directory discovery does not opt in,
create a queue or authorize a provider call.

## Launch and supervision

From this repository, launch the service and agents with the same trusted root,
model and endpoint. No roster is needed in automatic mode:

```powershell
$profileRoot = 'C:/host/profiles'

# Supervise exactly one of these for the host:
python .\skill_review.py run --profiles-dir $profileRoot --model Qwen-32b --base-url http://localhost:11434/v1

# A separate terminal/process for each authorized user:
python .\agent.py --profile alice --auto-skills --profiles-dir $profileRoot --model Qwen-32b --base-url http://localhost:11434/v1

# Later, while the SAME service stays running:
python .\agent.py --profile bob --auto-skills --profiles-dir $profileRoot --model Qwen-32b --base-url http://localhost:11434/v1

# Read-only status and a finite smoke/drain command (while run is stopped):
python .\skill_review.py status --profiles-dir $profileRoot --model Qwen-32b --base-url http://localhost:11434/v1
python .\skill_review.py once --profiles-dir $profileRoot --model Qwen-32b --base-url http://localhost:11434/v1
python .\skill_review.py --help
python .\skill_review.py once --help
python .\agent.py --help
```

Both commands default to `.agent_profiles` beside `agent.py`, never launch cwd.
Model defaults remain `AGENT_MODEL` or `Qwen-32b`; endpoint defaults remain
`OPENAI_BASE_URL` or `http://localhost:11434/v1`. Supply matching explicit settings
or the same environment to both processes. No hosted endpoint is inferred and no
model is switched automatically. Existing `--workspace`, `--read-only-dir`,
`--max-tokens`, database behavior and normal named-profile provisioning remain.

The service reads only immediate directory metadata during discovery, with no
recursion, catalog scan or profile initialization. Empty/missing roots are valid;
the first writable agent creates its profile through normal startup. `run` polls
with `--discovery-interval 1` (seconds, configurable from 0.05 to 60). Admission
usually follows the next scan; scan time and available service capacity add latency.
There is no profile-count truncation; 100+ profiles are covered by local tests.
Bad children and scan failures are reported on stderr while running, and in status
or final error output. A bad child does not prevent healthy children from running.

IDs are exactly 1–64 ASCII letters, digits, hyphens or underscores. Files, invalid
names, aliases, symlinks, junctions and other reparse paths are refused. Existing
root ancestors and selected directory incarnations are pinned and revalidated.
A replaced root is not adopted by a running process. Missing/replaced profiles
cannot receive results or be recreated by the service; old authorization binds
the directory incarnation across restart. This is not a deletion/restore manager;
retain coherent profile, mailbox and external control state. These checks rely on
the host protecting paths from concurrent filesystem mutation by untrusted users.

Supply a production API key to the service through its supervised environment's
`OPENAI_API_KEY`; do not put credentials in the roster. The service defaults to
`--timeout 30 --output-tokens 4096 --workers 1`, with SDK `max_retries=0`.
`--timeout` is bounded to 120 seconds and the output token setting to 8192.
SDK transport timeouts bound stalled network operations, not an end-to-end SLA.

Use your existing supervisor to launch `run` and send Ctrl+C/SIGTERM for graceful
stop. Stop admission first if retiring a configuration. Shutdown stops new dispatch and
result persistence, waits for its local inference threads, then releases the
singleton. A second scheduler fails while the first owns its slot. A replacement
may start only after the old process and local workers have exited; forced process
termination stops all its local threads. Remote inference might still complete
twice, so exactly-once remote execution is not promised. No OS service is installed
by this implementation.

`once` recovers interrupted attempts and processes at most the oldest prepared
batch present for each eligible profile at entry. It does not wait for disconnected
owners to prepare new work. `run` picks the oldest prepared batch whenever capacity
is available. At most one batch per profile is outstanding; a busy user's next
batch cannot replace another user's older prepared batch. An idle connected owner
polls for delivery approximately every 100 ms. Normal owner exit is a disconnect:
prepared work/results remain authorized and durable, and a later owner can apply
them. An owner lifecycle should be opened/closed on its process's main thread.
Live refresh recovers interrupted attempts from previous service generations,
including late-discovered queues, without resetting current in-flight work.

## Static compatibility and switching configurations

`python skill_review.py run --roster PATH` remains a static allowlist. It never
discovers or admits unlisted directories. Its matching owner command is
`python agent.py --profile alice --auto-skills --skill-review-roster PATH --profiles-dir ROOT --model MODEL --base-url URL`.
Existing roster JSON and `roster-entry --profile ID --state-root ABSOLUTE_PATH`
remain supported; `roster-entry` prints configuration without creating profile
state. Static roster changes still require restarting those configured processes.

On the service, `--roster` conflicts with `--profiles-dir`, `--model`, `--base-url`
and `--discovery-interval`; static model/endpoint settings come from its JSON.
On the agent, `--skill-review-roster` selects static mode, and `--profiles-dir`
still locates that listed runtime profile. Unlisted identities or mismatched
model/endpoint settings fail before profile provisioning. `--no-auto-skills`
continues to override `--auto-skills`.

Shared control lives in the profile root's parent at
`.skill-review-<first 24 hex characters of the normalized root path SHA-256>`.
The full root and model/endpoint are checked in `policy.json`; per-store authority
records fence configuration changes. No per-profile manual configuration is
needed. Control is outside the protected profile root and is never a skill catalog.
A different learning policy is refused, never overwritten by a competing launch.

This policy constrains review learning, not ordinary agent model/endpoint choices.
Without learning, a profile that has no review state, or whose current review
authority is already validly revoked, can use another `--model` or `--base-url`.
Plain launches and `--no-auto-skills`/`--no-skills` retain normal provisioning and
leave review policy, authorization and mailbox bytes unchanged in these cases.
An active authorization still requires its exact mode/model/endpoint to revoke;
a mismatched ordinary launch fails before provisioning and does not claim to have
canceled that work. Revocation proof comes from the current authority's origin,
matching generation and profile/store identity, including automatic directory
incarnation. Missing, malformed, stale or legacy unknown state is not proof of
revocation. A disabled auth file left in another control directory is not used.

Before switching static/automatic mode, revoke every affected profile using its
**old matching configuration** (`--no-auto-skills`, retaining the static roster
flag when leaving static mode), then stop the owners and service. Launch the new
mode only after revocation. A mailbox from the prior implementation with no mode
record is refused in automatic owner startup until a disabled launch with its
original static roster records revocation. Service discovery never imports such
queues or migrates catalogs. Explicit reenable creates a fresh generation and
purges canceled queue generations; changing modes cannot revive old work/results.
Switching back also requires revocation of the currently selected authority.

To change the automatic review service's model/endpoint, first revoke with the old settings
and stop all affected processes. Only then remove the external `policy.json`
for that root and launch both sides with the new matching settings. Keep the
per-store authority/auth files: they prevent revival of earlier configurations.
Do not remove control state or queue files as a shortcut around revocation.

## Admission, privacy and bounds

The foreground preserves the existing answer, completion log and session
persistence. It then runs `--auto-memory` **synchronously, in its original order**,
and performs a short local durable skill enqueue. Memory reflection can still
delay return, and background inference still consumes shared endpoint compute.
The current session's system prompt stays frozen; later constructed prompts can
see published skills. Pre-turn skill retrieval is unchanged.

Admission necessarily includes local I/O latency. SQLite busy timeout and enqueue
coordination acquisition each use 50 ms bounds; these are contention limits, not
a guarantee about OS/filesystem latency. No catalog scan, catalog hash, provider
request or result delivery occurs in foreground enqueue. Snapshot preparation and
publication run in the owner thread, with no inference in that thread.

| Bound | Default/contract |
| --- | --- |
| One task's evidence | 16 KiB, 64 messages; traversal limited to 512 nodes/depth 12 |
| Unconsumed private evidence | 128 tasks or 1 MiB, whichever fills first |
| Prepared batch | Oldest 4 tasks, at most 64 KiB of message evidence |
| Outstanding batch | One per profile, including retained disconnected results |
| Catalog prompt context | At most 128 summaries/16 KiB; overall catalog snapshot under 48 KiB |
| Eligible UPDATE context | At most 8 complete Markdown targets, each at most 12 KiB; never truncated |
| Supported catalog file | Read at most 256 KiB plus one overflow byte per file |
| Prepared inference request | At most 128 KiB; otherwise explicitly INVALID |
| Model result | At most 24 KiB and the configured token cap |
| Terminal job history | Last 32 metadata outcomes; task/catalog/result bodies cleared at acknowledgement |

Capture is incremental at task/message boundaries before context compaction, not
a copy of the current session transcript at completion. It omits system prompts
and uses `TrajectoryLogger._safe` / `ContextManager.redact_sensitive_value` on
bounded values. Completed tasks remain separate across sessions. A minimal
user+final-assistant task is valid; incomplete, tool-ending, unanswered-tool,
empty or length-terminated completions are excluded. Oversized evidence refuses
the task instead of truncating secrets or silently evicting older tasks.

`ACCEPTED` means SQLite committed the bounded task record before return. `OVERFLOW`
means no admission and earlier records remain. `FAILED` means local admission did
not succeed (for example lock contention/storage failure). `DISABLED` and
`INCOMPLETE` are refusals. Failures do not change a successful task answer. This
does not guarantee reviewing every task forever, automatic retry of refused
tasks, or survival beyond normal local SQLite/filesystem durability guarantees.
Only evidence included in an acknowledged batch is consumed. Terminal provider
errors/invalid proposals consume their included batch with an explicit outcome.

## Authorization and publication

Host coordination state lives **outside protected profiles**. Its generation and
secret seal bind every job's profile, store, job ID, evidence, public snapshot and
private target revisions/paths. The result receipt additionally binds the service
generation and exact result. Models receive only task evidence, catalog summaries
and opaque IDs with full eligible target instructions. They cannot supply routing,
revisions, receipts or permissions. A service never follows model-selected paths.

Admission, worker start, result persistence, delivery and publication recheck the
generation under the same per-catalog process/thread lock. Mode changes synchronize
with mutations and rotate/revoke the external authorization **before returning**.
After read-only/stateless/no-skills acknowledgement, background work cannot write
profile queues, caches, logs or skills. There are no profile heartbeats. Lock
coordination and external revocation persistence are distinct from profile writes.
An already admitted request may finish remotely; its invalid generation prevents
result persistence/publication. Reenable purges invalidated generations while
authorized and never revives them. Disabling learning creates no review/cache/queue
state; matching active authorization is revoked in the external control area.

In automatic mode, named disabled/read-only/stateless/no-skills/no-auto-skills
launches locate existing authorization from the same root;
no roster flag is required. A never-enabled disabled launch creates no review
mailbox or review control state. `--no-skills` and `--no-auto-skills` preserve
normal profile, history, memory and workspace provisioning, including an empty
skills directory; they do not enable review requests or publish learned skills.
Only `--read-only`/`--stateless` startup refuses a missing profile after revoking
any existing matching external authorization, without recreating that profile.
In static mode, retain the roster flag on every launch, including
disabled launches, so the process can revoke previously disconnected work.
`/mode normal` restores startup
opt-in eligibility, including after an initially read-only/stateless launch;
`/mode no-skills` disables skill learning. Ordinary disconnect preserves the
generation. Closing an owner fences local delivery and does not join inference.
To change model/endpoint policy, use the revocation and configuration-switching
procedure above; static mode additionally requires editing its host roster.

All supported catalog writers/readers use one confined resolver and mutation lock.
Resolved containment is checked before opening a file, including symlinks and
junctions. Distinct logical targets with the same normalized name are ambiguous.
Direct model-facing `save_skill` is CREATE-only: exact normalized-name collision
refuses without changing existing bytes; similar distinct names remain allowed.
No Jaccard cutoff, overwrite flag or model receipt/update bypass is exposed.

UPDATE checks the host-held revision of only its eligible target; unrelated skill
additions do not invalidate it. Nested `SKILL.md` updates stay at their original
path. Unknown IDs, renames, path attempts, stale/ambiguous resolution, spoofed
metadata, extra/duplicate JSON fields, incomplete JSON/replacements and provider
`finish_reason=length` fail closed. Content uses the existing safety screen.
Completeness/safety checks are syntactic/heuristic; they cannot prove the semantic
quality of arbitrary generated instructions.

Generated Markdown is canonical. One atomic replacement of that Markdown is the
authoritative commit point; a second replacement writes an optional revision-bound
JSON cache. **Those two replacements are not a crash-atomic pair.** Cache failure
after the Markdown commit is reported as committed with an unavailable cache.
Readers derive skills from Markdown and never load marked caches as authority,
including stale/missing/orphaned caches. No startup/read path repairs a cache.
The next authorized publication rebuilds it from canonical fields.

Independent imported JSON-only skills retain their information and bytes, remain
readable, and are ineligible for automatic UPDATE. Undefined/conflicting pairs or
independent JSON occupying a cache slot are also ineligible. Legacy unmarked pairs
are coalesced only when the JSON has exactly the former generator's six fields,
names its Markdown sibling, and re-renders to that entire Markdown (line-ending
normalization only). When publishing from a recognized legacy pair, the canonical
Markdown commit also records `hermes_legacy_cache`: the sibling filename and
SHA-256 of those exact old JSON bytes. If cache replacement fails or the process
exits after that commit, this proof keeps that specific old JSON classified as a
cache across restart and permits receipt-based UPDATE retries without another
publication. Changed unmarked JSON does not match the proof and remains
independent/ambiguous. Readers do not repair it; the next authorized publication
can rebuild the proven cache. There is no name-based exemption or broad migration.

Authoritative Markdown embeds host operation receipts. An owner crash after commit
but before queue acknowledgement recovers by recognizing that exact receipt and
acknowledging DUPLICATE without publishing again, including UPDATE. Later updates
retain receipts. Deletion publishes a canonical tombstone retaining receipts;
readers hide it and stale JSON cannot resurrect it. Recreating a name retains its
receipt history. Receipt growth counts against file/target limits. Unsupported
external edits/removal of authoritative receipts/tombstones bypass this protocol;
backups/restores must preserve catalog and queue/control identity coherently.

## Recovery and acceptance ledger

Service restart under singleton exclusion returns authorized RUNNING jobs to
PREPARED with a new service generation. Already persisted valid results remain
deliverable across restart. Late old attempts cannot overwrite a reassigned job.
Provider timeout/errors become FAILED; bad proposals become INVALID; publication
can be APPLIED, DUPLICATE, NONE, STALE, COLLISION, INVALID, FAILED or CANCELLED.
Revoked records are logically CANCELLED by their obsolete generation without a
post-revocation profile write. They are removed on authorized reenable.

Local mailbox/persistence faults are isolated per profile and reported by the
service's error output or the owner's `last_error`; admitted evidence is retained.
Restart the service after resolving a local result-persistence fault to recover
its RUNNING attempt. Invalid seals/modified queue records fail closed and require
operator diagnosis with the processes stopped; do not edit them into authorization.
`status` reads existing state without initialization or repair. Invalidated rows
can retain their old displayed status until reenable purges them; enabled=false
means the service will not start/persist their work.

| Acceptance item | Evidence / remaining limit |
| --- | --- |
| Actual foreground completion and frozen prompt | `test_skill_review_integration.py`; separate blocked-provider and synchronous-memory cases |
| Bounded private evidence/admission | `test_async_skill_review.py`; redaction, session boundaries, compaction caller, oldest batches, overload and busy refusal |
| Profile binding, revocation and owner lifecycle | Isolation after global rebind, forged metadata/paths, idle delivery, reconnect, exact file/directory mtime after queued/running revocation |
| One service, bounded pool, shutdown/restart | Singleton tests plus real public subprocess owner/service HTTP integration in `test_skill_review_process.py` |
| Automatic discovery and live additions | `test_profile_discovery.py` and `test_profile_discovery_process.py`; real local HTTP and owner/service subprocesses, same PID across additions, empty/missing root, redaction, separate workspaces, blocked inference and revocation |
| Host publication and crash recovery | `test_skill_publication.py`; collisions, eligible full targets, revisions, nested paths, imported JSON, cache interruption, tombstones/duplicate UPDATE |
| Confinement | Actual Windows junction test; symlink creation test skips if the host lacks that privilege |
| Scale scope | 100 private profiles scheduled using deterministic local callbacks; no claim of 100 real simultaneous model requests or production load benchmark |
| Existing behavior | Prior safety/profile/workspace/compaction assertions retained; synchronous extractor tests ported to snapshot/owner contracts, old two-file rollback expectation replaced by before-commit preservation plus after-commit cache recovery |
| Final checks | Focused suite, full pytest, syntax compilation, diff whitespace and public help/once smoke; actual commands/results recorded in the external evidence log |
| Review handoff | Uncommitted implementation; parent verification and independent named review remain external steps, not an approval claimed here |

The original evidence log is `C:/Users/Owner/AppData/Local/Temp/async-skills-clean-rebuild-evidence.md`.
Automatic-discovery RED/GREEN and final verification are recorded in
`C:/Users/Owner/AppData/Local/Temp/async-profile-discovery-evidence.md`.
No production requests, live service installation, credentials or real profile
catalogs are needed for the acceptance tests.
