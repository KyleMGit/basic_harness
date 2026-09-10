# Asynchronous private skill learning

Run one persistent `agent.py --profile ...` process per authenticated user and
**one supervised `skill_review.py run` process for the host**. Automatic learning
is opt-in. The owner has a lightweight lifecycle/snapshot/delivery thread; only the
separate service schedules inference. Default capacity is one request, with a
fixed pool configurable from 1 to 8 at service startup. No agent starts a service.

Automatic mode derives profile IDs, private mailbox paths and immutable catalog
identities from one host-controlled profile root. Owners that enable learning and
the service use the same host model and endpoint; there is no smaller/second-model dependency. The service reads
sealed episode manifests and authorized catalog snapshots, assembles bounded
whole-exchange evidence, and writes proposals into that user's mailbox. It never
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
revision present for each eligible profile at entry. It does not wait for disconnected
owners to prepare new work. `run` picks the oldest prepared revision whenever capacity
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

This release adds owner-created `episodes`, `episode_sources`, bounded
fingerprint tables, and additive anchor/association/whole-unit-retirement columns on `episodes` without
rewriting legacy `evidence`/`jobs`. Existing accepted
legacy jobs keep their processing and acknowledgement contract and are not
retrospectively gated or re-captured from history. Schema initialization occurs
only when an authorized writable owner enables learning; read-only/status/service
startup does not initialize or migrate a profile. Deploy owner and service code
together and perform a coordinated stop/start; mixed old/new binaries processing
new episode manifests are unsupported. Do not delete/revoke queues merely to
upgrade, and preserve/report a pre-existing mailbox above the 16-MiB page ceiling
instead of truncating, resetting, or vacuuming it automatically.

To change the automatic review service's model/endpoint, first revoke with the old settings
and stop all affected processes. Only then remove the external `policy.json`
for that root and launch both sides with the new matching settings. Keep the
per-store authority/auth files: they prevent revival of earlier configurations.
Do not remove control state or queue files as a shortcut around revocation.

## Selective episodes, privacy and bounds

The foreground performs one deterministic, bounded caller-start check for an
explicit procedural challenge before model execution. If matched, the owner
durably marks the current episode revision challenged and supersedes its pending
job before the turn proceeds; this uses no provider or catalog scan. The foreground
otherwise preserves the existing answer, completion log and session persistence.
After completion it runs `--auto-memory` **synchronously, in its original order**,
and performs bounded omission plus a short local durable turn-delta capture. Memory reflection can still
delay return, and background inference still consumes shared endpoint compute.
The current session's system prompt stays frozen; later constructed prompts can
see published skills. Pre-turn skill retrieval is unchanged.

Caller-start invalidation and completion admission necessarily include local I/O
latency. SQLite busy timeout and coordination acquisition each use 50 ms bounds;
these are contention limits, not
a guarantee about OS/filesystem latency. No catalog scan, catalog hash, provider
request or result delivery occurs in foreground capture. Catalog snapshot preparation
and publication run in the owner thread, with no inference in that thread. Evidence
assembly, whole-exchange selection and final request composition run in the shared
service after claim and outside the owner/foreground coordination lock.

A completed turn is only a capture boundary. Deterministic host observations make
an episode eligible after a resolved non-routine SQL/procedural failure, a supported
substantive correction, or an investigated reusable procedure that was actually
verified. Dates, filters, sorting, limits, formatting, routine grouping,
clarifications, acknowledgements and repetitions do not independently qualify.
There is no eligibility/summarizer model request. Conservative signatures can miss
learning or admit a duplicate; the existing reviewer still makes CREATE/UPDATE/NONE.

Related turns stay in one bounded episode. Eligible work dispatches after 120 seconds
idle, a topic/session boundary, normal exit, or at the next completed-turn boundary
once the oldest eligible signal is 900 seconds old. A newly verified substantive
correction is ready immediately. A start-boundary challenge requires explicit
wrong/incorrect language plus procedural or structural terms; routine date, filter,
sort, limit, format and similar parameter wording does not challenge a revision.
A recognized challenge increments the revision and supersedes older
prepared/running/results before model execution. Its durable challenged state
survives a failed/interrupted turn and same-generation reconnect, so completion is
not required to block an obsolete proposal. Remote inference is not claimed to be
cancellable. Already published skills remain linked catalog targets for a later UPDATE.

A routine turn with a changed SQL signature does not revise or invalidate a frozen
PREPARED/RUNNING/RESULT episode and cannot create a second request. A complete
post-dispatch turn containing a new failure, metadata finding, verified non-routine
execution, or correction is retained unbound as a later revision. Acknowledgement
consumes only sources bound to the frozen job, then promotes those deferred sources;
stale/superseded bindings are cleared without publishing or reviving the old result.

After acknowledgment, the owner retains only the original user request as a
separately labelled `context_only` anchor (plus a bounded related-skill association),
not the acknowledged tool history. The anchor is counted in private payload and
episode limits and, when needed by a later correction, in selected-view bytes and
messages. The 16-KiB carry bound applies only after this source becomes a context-only
anchor; a complete first source may use the 64-KiB selected-view budget. An oversized
future anchor is omitted whole and replaced by a bounded `missing_context` record.
If a later revision needs that absent original context, preparation ends with an
explicit `BUDGET_REFUSED` and zero provider calls rather than sending a truncation.

When a DRAFT episode reaches its byte/message limit, the owner may retire oldest
optional routine turns as complete units. It preserves the original turn, failures,
metadata, corrections, verified non-routine support, and every job-bound source.
Requests report retired source/message/byte counts and the retirement reason. A
normal agent shutdown explicitly retires source-free ACKED state and unresolved,
unpinned CHALLENGED state for that session; PREPARED/RUNNING/RESULT work is never
retired there. A plain owner disconnect does not perform that explicit retirement,
so same-generation reconnect keeps its challenge freshness fence and pending work.

| Bound | Default/contract |
| --- | --- |
| One completed turn | 256 KiB / 256 retained messages after SQL business-output omission |
| Raw input inspection | 256 KiB for the complete selected message before JSON/XML/argument/result decoding |
| Decoded structure | 8,192 nodes / depth 24 per message, including JSON decoded from strings |
| One episode | 512 KiB / 512 retained messages, including pinned antecedents |
| Context-only carry anchor | 16 KiB; indispensable larger evidence is refused rather than sliced |
| Pending private capacity | 128 episode/legacy units / 8 MiB total live payload, with 512 KiB reserved inside it for preparation/results |
| Repetition index | 1,024 fingerprints / 128 KiB, counted in pending payload |
| Selected review view | 64 KiB / 128 messages including wrappers, IDs, provenance and omission records |
| Outstanding batch | One per profile, including retained disconnected results |
| Catalog prompt context | At most 128 summaries/16 KiB; overall catalog snapshot under 48 KiB |
| Eligible UPDATE context | At most 8 complete Markdown targets, each at most 12 KiB; never truncated |
| Supported catalog file | Read at most 256 KiB plus one overflow byte per file |
| Prepared user-content JSON | At most 128 KiB after composition |
| Complete SDK JSON body | At most 256 KiB including prompt, settings, envelope and escaping |
| Model result | At most 24 KiB and the configured token cap |
| Private mailbox | 16-MiB SQLite page ceiling; DELETE rollback journal and per-connection journal-size limit |
| Terminal job history | Last 32 metadata outcomes; task/catalog/result bodies cleared at acknowledgement |

Capture is incremental at task/message boundaries before context compaction, not
a copy of the current session transcript at completion. It omits system prompts
and uses `TrajectoryLogger._safe` / `ContextManager.redact_sensitive_value` on
bounded values. Native and Hermes XML SQL results are tied to call IDs. Recognized
business `rows` are removed before admission regardless of size and replaced by a
labelled receipt retaining database, bounded columns, row count and truncation.
Only supported metadata commands or queries whose parsed `FROM`/`JOIN` sources are
all recognized catalog sources may retain bounded results. SQL comments and string
literals are ignored for classification; comma joins, business/catalog joins,
subqueries/CTEs with uncertain sources, and other unknown or mixed SQL output are
not trusted as metadata. Identifiable assistant copies of structured rows/CSV are
omitted whole. Executed arguments, call IDs, redacted errors and useful schema facts
remain; interactive outputs, history, exports and synchronous auto-memory do not
change. This is business-result omission, not comprehensive anonymization of values
inside arbitrary user prose or SQL literals.

Incomplete, tool-ending, unanswered-tool,
empty or length-terminated completions are excluded. Oversized evidence refuses
the task instead of truncating secrets or silently evicting older tasks.

Legacy direct `Owner.enqueue` records retain their earlier processing/consumption
contract. New automatic capture persists only turn deltas and stays quiet for normal
non-eligible work. `ELIGIBLE` means a positive local signal is retained, not that a
provider call or publication has happened. `OVERFLOW`
means no admission and earlier records remain. `FAILED` means local admission did
not succeed (for example lock contention/storage failure). `DISABLED` and
`INCOMPLETE` are refusals. Failures do not change a successful task answer. This
does not guarantee reviewing every task forever, automatic retry of refused
tasks, or survival beyond normal local SQLite/filesystem durability guarantees.
Only references included in an acknowledged result/refusal are consumed. An
indispensable candidate that cannot fit produces `BUDGET_REFUSED` with zero provider
requests and does not block later fitting work. Terminal provider errors/invalid
proposals retain their explicit outcome; transient local preparation/storage faults
retain accepted work for recovery.

SQL-shape fingerprints from successfully completed semantic-review proposals are
consulted across episodes to suppress a fully repeated failure/resolution set.
Zero-request budget refusals and INVALID/FAILED review attempts do not add
fingerprints; their consumed evidence is not retried, while a new fitting episode
remains eligible. A correction or any new shape can still progress. The index is
pruned oldest-first at 1,024 records and 128 KiB; eviction may permit a later
duplicate review but never restores an acknowledged source or result.

## Reading the diagnostics

The agent prints exceptional capture/admission refusals in the conversation. Normal
non-eligible turns are quiet. Example shapes are:

```text
[Skill Review] OVERFLOW: stage=capture.traversal reason=raw_bytes observed_bytes=300028 limit_bytes=262144; NOT queued; no automatic retry; earlier queue work retained.
[Skill Review] OVERFLOW: stage=capture.serialized reason=serialized_bytes observed_bytes=262145 limit_bytes=262144; NOT queued; no automatic retry; earlier queue work retained.
```

The raw count is exact for the complete selected message fields and is checked before
decoding assistant JSON, native tool arguments, Hermes XML calls, or SQL result JSON.
An oversized uninspectable message is refused as a whole; safely inspectable
structured SQL business rows are then projected before admission. The serialized count is the
exact compact retained-message array (the prior separator off-by-one is removed);
JSON escaping can make it larger than raw UTF-8. Structured SQL business rows are
projected before this retained-data path, but projection is not an exemption from
the raw inspection bound.

`capture.traversal` also distinguishes depth 25 against limit 24, node 8193 against
limit 8192, and unsupported data. Depth/node observations describe the partial
traversal. `capture.serialized` reports message count overflow separately from
bytes. `capture.completion` reports incomplete evidence and its captured message
count; two messages alone do not prove a complete task. Decoded JSON inside strings
uses the same node/depth counters, and message 257 is refused.

`admission.lock`, `admission.authorization`, and `admission.sqlite` distinguish
local admission failures. Known acquisition and SQLite busy settings appear as
`coordination_timeout_s=0.05` and `sqlite_busy_timeout_s=0.05`. `admission.queue`
reports `queue_records` or `queue_bytes` with the observed count including the
rejected unit and the applicable limit (128 units or 7864320 admission bytes after
the 512-KiB preparation/result reservation).
Queue byte diagnostics also show the already queued and incoming byte counts.
Every refusal says the task was **NOT queued**, has **no automatic retry**, and
retains earlier queue work. A legacy direct `ACCEPTED` or automatic `ELIGIBLE` is
durable local state only; it does not promise publication. Unavailable guidance points to the automatic launch settings
above; a static roster is optional.

The separate service emits errors immediately on stderr, including in `once`.
These examples came from local fake-provider and lock-failure runs:

```text
Skill review: historical stage=provider.inference error=APITimeoutError profile=user-0 job=fa7e613b794e4287b5f8685fd257f894 provider_timeout_s=1.25; outcome=FAILED; no automatic provider retry; awaiting result persistence and owner acknowledgement.
Skill review: historical stage=result.persistence error=TimeoutError profile=user-0 job=db6d37c0463d488eb07605c5a5eb7be6 coordination_timeout_s=5; RUNNING attempt interrupted; no active retry; resolve coordination/storage failure and restart the service to recover.
```

| Stage | Consequence and operator action |
| --- | --- |
| `pending.scan` / `pending.recovery` | Scanning or recovery was interrupted. Admitted work remains. `run` automatically tries again on a future loop; `once` promises no future run. Check local coordination/storage if repeated. |
| `claim` | Claiming the prepared job was interrupted. The same future-loop guidance applies; a lock timeout is local coordination, not a model timeout. |
| `worker.authorization` / `worker.dispatch` | A claimed RUNNING attempt was interrupted before inference. There is no active retry for that attempt. Resolve the local fault, stop the service, and restart it after its workers exit. |
| `provider.inference` | The provider call failed, including SDK `APITimeoutError`. The configured provider timeout is shown. A persisted FAILED result is acknowledged by the owner and consumes that batch; it is not NONE. |
| `provider.validation` / `provider.inference_validation` | Output was rejected. The latter covers an adapter that performs inference and validation together. The existing INVALID/FAILED outcomes remain; arbitrary validation exception text is omitted. |
| `result.persistence` | The result could not be durably recorded. A RUNNING attempt may remain even after later scans succeed. Resolve the local fault and restart the service; those scans do not retry this attempt. |
| `service.startup.*` / `service.status` / `service.shutdown` | The command reports a safe exception class and launch/storage guidance, without raw exception text. A busy singleton reports its zero-wait setting; let the prior service exit before restarting. |

Timeout fields describe the configured operation whose failure was observed.
Lock acquisition uses 5 seconds in claim, worker authorization and result
persistence; SQLite uses 0.05 seconds. An exception raised outside a known timed
operation does not acquire a guessed timeout. SDK timeouts remain transport
settings, not an end-to-end deadline. No retry or recovery policy changes here.

The final service JSON retains `processed` and `errors`. Each retained error value
is explicitly **historical**: it records a failed attempt, not a claim about current
health or recovery. A successful pending scan does not erase a persistence failure
or announce recovery. Identical diagnostics for a stage/profile/job are suppressed
while retained in a 128-entry in-memory suppression cache. The error map and
discovery error map are also bounded to 128 entries; entries can be evicted, and an
evicted diagnostic can be emitted again. These caches are process-local, not a
complete audit history. Existing `status` fields remain; `status` does not load
another process's error history or create a persisted diagnostic schema.

Owner preparation, delivery, publication and acknowledgement exceptions are
available on the live Python object's `agent.skill_review_owner.last_error` (or
`owner.last_error` for an embedded owner). This single bounded string includes a
historical label, local stage, safe exception class, profile and any available job
ID. It is **not delivered to chat** and is not exposed by service `status`. When a
publication method returns a failure without an exception object, terminal job
metadata records the publication stage/outcome and says exception detail is
unavailable. No class is reconstructed from arbitrary returned text. Successful
CREATE/UPDATE/NONE chat notifications remain outside this implementation.

Diagnostics omit evidence, provider bodies/headers, credentials, SQL, full paths
and traceback bodies. Known exception classes are allowlisted; custom classes use
a safe base class. Invalid or oversized profile/job identifiers become `<invalid>`;
discovery child labels are bounded and restricted to safe ASCII. Diagnostic data
uses the existing console, in-memory error and authorized terminal-job metadata
paths. No file logger or post-revocation profile diagnostic writes are added.

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
Logging TDD and verification are recorded in
`C:/Users/Owner/AppData/Local/Temp/skill-review-logging-evidence.md`;
the focused caller-visible cases are in `test_skill_review_diagnostics.py`.
No production requests, live service installation, credentials or real profile
catalogs are needed for the acceptance tests.
