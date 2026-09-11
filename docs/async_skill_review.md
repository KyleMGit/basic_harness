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
`--timeout 30 --output-tokens 4096 --workers 1`, with SDK `max_retries=0`. Its
independent request resource guards default to `--selected-view-bytes 262144`,
`--prepared-input-bytes 524288`, and `--wire-body-bytes 1048576`. They bound
selected review evidence, prepared evidence plus catalog, and the complete
serialized SDK request respectively; they are byte guards, not model-context
promises. Each accepts a positive integer byte count. `--timeout` is bounded to
120 seconds and the output token setting to 32768.
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

## Asynchronous review lifecycle notices

The owner background thread never prints. When the owner durably prepares each
new job, it records a `REQUEST` terminal event and a private lifecycle-log event.
This means requested/queued, not worker start or provider transmission. An authorized service worker records
the first actual worker entry on the existing mailbox job, after claim and a
fresh profile/store/generation/input-seal authorization check and before local
request preparation. Capture, eligibility, queue preparation, claim, and
executor dispatch alone do not record a start. The signed marker contains only
the job/profile/store/generation binding, first-start time, service generation,
and input seal; the service never writes the owner's private `notices` table or
publishes a skill. `[Skill Review] Review started.` therefore means authorized
local review processing began. It does **not** prove that a provider request was
transmitted, and local preparation or budget refusal can legitimately follow it.

The authorized owner validates that marker and converts it to a sealed private
outbox event while its bound source metadata is still available. It does this
for both RUNNING and RESULT jobs, so a fast transition cannot hide the start and
start is ordered before the final event. The marker is stable once per job:
routine owner pumps, same-generation reconnects, and recovery under a later
service generation do not create another start. Older jobs without a trustworthy
marker receive no inferred historical start. An older owner schema without the
marker columns continues its established final-only service protocol; only a
writable authorized owner adds the columns.

After the owner validates a sealed final result and, where applicable, the
canonical Markdown publication commits, it adds a sealed host-generated final
event in the same transaction that acknowledges the review job and consumes its
bound sources. Lifecycle events contain only an allowlisted outcome, committed
host name for CREATE/UPDATE, bounded opaque source labels/counts/times, start
time, job/episode/revision metadata as applicable. They never retain SQL, task
text, evidence, proposal instructions, arbitrary provider/exception detail, or
a raw proposal body.

The CLI main thread drains that outbox only at terminal-safe boundaries: startup
and immediately before the next prompt; after `input()` returns and before a new
response or command result; after a foreground response and synchronous memory
reflection finish; and during orderly shutdown when practical. A lifecycle event
arriving while the terminal is idle or partially typed therefore waits for
submission. An event arriving during streaming, model work, tool output, or a
terminal approval prompt waits for the corresponding safe boundary. There is no terminal
repaint dependency. Notices are not model messages, instructions, evidence, or
session history and cause no model call.

Representative output is:

```text
[Skill Review] Review requested. Review queued for an authorized local worker. Source scope: selected legacy batch (...). Review job <opaque-job-id>.
[Skill Review] Review started. Recorded authorized local worker entry at 2026-09-10T12:02:00Z; this does not prove a provider request was transmitted. Reason: authorized local worker entered review; provider transmission is not proven. Source scope: selected legacy batch (...). Review job <opaque-job-id>.
[Skill Review] Created skill 'learned_workflow'. Reason: explanation unavailable. Source scope: selected legacy batch (...). Review job <opaque-job-id>.
[Skill Review] Updated skill 'existing'. Reason: explanation unavailable. Source scope: bound episode source set (2 bound source record(s), 2 task(s), 1 session(s); sessions=session-a; tasks=task-a,task-b; episode=<opaque-episode-id> revision=2; 2026-09-10T12:00:00Z to 2026-09-10T12:01:00Z). Review job <opaque-job-id>.
[Skill Review] Review completed with no skill change. Reason: explanation unavailable. Source scope: selected legacy batch (...). Review job <opaque-job-id>.
[Skill Review] Recovered the prior publication receipt for created skill 'recovered'; no new publication was made. Source scope: selected legacy batch (...).
[Skill Review] No skill was published: the review or publication attempt failed. Source scope: selected legacy batch (...).
```

The same unabridged per-review lifecycle is appended to
`<owning-profile-root>/logs/skill_reviews.jsonl`, never to a workspace or shared
root. Each UTF-8 JSON line has exactly `timestamp` (UTC ISO-8601), stable
`event_id` (`<review-id>:requested|started|decision`), `review_id`, `event`
(`REQUESTED`, `STARTED`, or `DECISION`), `action` (empty for lifecycle markers;
otherwise `CREATE`, `UPDATE`, or the allowlisted non-publication decision),
`skill_name` (empty unless applicable), `status`, and a concise host-allowlisted
`reason`. CREATE, UPDATE, and NONE use
`explanation unavailable` because the reviewer protocol supplies no decision
explanation. The log never contains evidence, prompts, SQL, business rows,
proposal instructions, provider bodies, exception bodies, or model reasoning.
`--quiet-skill-reviews` suppresses terminal rendering only; JSONL appends continue.

Pending JSONL events live durably in the owning profile mailbox and are flushed
by the owner pump without requiring a foreground terminal drain. A write or
flush failure leaves them pending, records a fixed `append_<safe exception class>` state and a
safe `owner.review_log` diagnostic, and retries on a later pump or same-generation
reconnect. Prolonged storage failures retain pending rows, subject to the unchanged
finite mailbox/disk budget, and require operator repair. Reconciliation scans
existing stable event IDs before appending, so
ordinary retry/reconnect does not duplicate lines. The file is flushed and
`fsync`ed before mailbox acknowledgement. A crash between that fsync and the
mailbox update is reconciled only when the complete existing record exactly
matches the signed pending record; an ID collision with different content and
any malformed/partial existing line fail closed. Pending rows are bound to the
profile, store, generation, and job and authenticated with the profile secret.
Their exact bounded schema is checked before file I/O. Invalid or legacy-unsealed
rows are quarantined with their payload erased and a fixed diagnostic; they are
never copied to the log. Successfully appended receipts are pruned to the newest
32 rows, independently of the append-only JSONL, while pending retries are never
pruned. Thus a lost or externally damaged JSONL cannot be reconstructed from old
receipts. No stronger cross-filesystem exactly-once claim is made. Terminal
compaction never removes JSONL history, and historical jobs are not backfilled.

CREATE/UPDATE success is emitted only for `APPLIED`; UPDATE names come from the
host-held target rather than model text. A canonical Markdown commit followed by
an optional cache failure is still `APPLIED`. `DUPLICATE` means an authoritative
receipt proves an earlier publication (even if a later canonical update or
deletion retained that receipt), so its wording says recovered rather than newly
published. `NONE` is emitted only for a valid `{"action":"NONE"}` result.
`BUDGET_REFUSED`, `INVALID`, `FAILED`, `STALE`, `COLLISION`, and `CANCELLED`
use compact allowlisted failure wording and are never described as NONE or
success. `SKIPPED`, `ELIGIBLE`, and admission `ACCEPTED` are not final outcomes.
For a newly generated `BUDGET_REFUSED`, the service stderr diagnostic and durable
owner notice also report an allowlisted reason, exact observed/limit integers,
the correct `bytes`, `messages`, or `sources` unit, and zero provider requests.
The only reason keys are `episode_sources`, `carry_anchor_bytes`, `missing_context`,
`selected_messages`, `selected_view_bytes`, and `prepared_bytes`; notably,
`episode_sources` is a source count, not a token or byte count. Unknown, malformed,
legacy, or out-of-range diagnostic data is discarded and the existing generic
failure wording remains deliverable. Raw result detail, task text, paths, SQL,
business data, exception text, and model output are never copied into a notice.
`SUPERSEDED` remains quiet as a final outcome because it is an internal revision
replacement; an already-recorded start remains truthful and can appear without
a fabricated completion, while the fresh revision can later produce its own
start and final events. Disable/revoke may likewise leave a previously delivered
start without a final, but revocation acknowledgement permits no later marker
conversion, delivery, migration, or replay from the invalidated generation.

Episode attribution counts the source records bound to the reviewed revision,
not the smaller evidence subset the service may select to fit its review view.
It therefore distinguishes the older/batched owner work without claiming every
bound turn was included in the provider request.

Output is written and flushed before its rows are durably marked delivered. A
write/flush failure retains all selected rows. A crash after bytes are printed
but before acknowledgement can repeat them; the acknowledgement proves only
successful local output/flush, not that a human saw the bytes. Successful
acknowledgement suppresses ordinary duplicates. Pending rows survive job-history
pruning, source consumption, normal disconnect, and same-generation reconnect,
including reconnect from a different session. A disable/revoke is ordered after
any already-started foreground drain, and no later drain, migration, retention
write, or acknowledgement occurs after the mode acknowledgement. Reenable uses a
new generation and does not replay invalidated rows.

Outside an active terminal-output claim, at most 24 detailed lifecycle events
(requests, starts, and finals) and one summary remain pending. On further disconnected
work, the oldest details are atomically replaced by that durable summary
containing only request/start counts, distinct allowlisted final-outcome counts, and a
time span. The summary total is explicitly a count of review events, not final
outcomes or distinct jobs; FAILED and NONE remain separate final counts. A drain
claims one such bounded set before printing; work that finishes during output
enters a separate independently capped pending set.
The claimed and new sets can therefore coexist until successful acknowledgement,
or until failure/crash recovery releases and recompacts the claim. The summary
explicitly says older names and source identifiers were compacted, that it
counts lifecycle events rather than jobs/completions, and that it is not a named
success notice. After that summary is delivered, a later overflow creates a new
summary cycle. The most
recent 32 delivered/invalid rows are retained for bounded diagnostics. Thus full
named per-event retention across unbounded disconnection is intentionally
impossible, but eviction is never silent. Summary totals and per-outcome counters
saturate at 1,000,000 and then render as explicit `at least`/`>=` lower bounds
instead of overflowing or implying an exact count. The two bounded active sets,
summaries, and history fit inside the existing 512-KiB mailbox headroom; the 8-MiB
live evidence/result budget, review caps, and 16-MiB SQLite page ceiling are
unchanged. Invalid identities, fields, bounds, payloads, marker seals, or notice
seals are quarantined without rendering or coalescing. Existing episode and
pre-episode mailboxes gain the outbox table and start-marker columns additively
only while an authorized owner is writable; terminal history with already-cleared
bodies is not backfilled.

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
fingerprint/outbox tables, additive anchor/association/whole-unit-retirement
columns on `episodes`, and nullable signed first-start columns on `jobs` without
rewriting legacy evidence or reconstructing starts. Existing accepted
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
latency. SQLite busy timeout and short coordination acquisition each use 500 ms
bounds; these are per-contention limits, not a guarantee about total operation-chain
or OS/filesystem latency. The existing 5-second service coordination waits are
unchanged. No catalog scan, catalog hash, provider
request or result delivery occurs in foreground capture. Catalog snapshot preparation
and publication run in the owner thread, with no inference in that thread. Evidence
assembly, whole-exchange selection and final request composition run in the shared
service after claim and outside the owner/foreground coordination lock.

A completed turn is only a capture boundary. Deterministic host observations make
an episode eligible when a successfully completed business SQL event is marked
non-routine, without requiring preceding database metadata, a failure/recovery, or
a correction. Structural completion still applies, and failed-only, schema-only,
metadata-only, plain lookup/count, unfinished, clarification, acknowledgement and
repetition shapes do not independently qualify. Substantive challenges still
invalidate prior conclusions until a later supporting execution verifies the
correction.

Successful `export_impala_csv` and `export_teradata_csv` calls participate in the
same SQL classification only when the result is the exporter's exact completed
manifest, its backend matches the originating tool, and its SQL digest matches the
exact originating SQL. Capture retains bounded manifest metadata in an
`exported_result_omitted` receipt; it does not read the CSV or treat a complete
export as a truncated query preview. Failed, incomplete, malformed, mismatched and
unmatched export results do not establish successful SQL evidence. Routine and
metadata-only exports remain non-eligible under the unchanged policy.

The non-routine marker remains a lightweight SQL-shape heuristic (for example,
`JOIN`, `WITH`, windows, and `CAST`), not a parser or a claim of semantic novelty.
There is no eligibility/summarizer model request. This threshold can increase review
volume and cost, and genuine `NONE` results are expected; the existing reviewer
still makes CREATE/UPDATE/NONE decisions. Conservative signatures can miss learning
or admit a duplicate.

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
Superseded or authorization-revoked jobs can therefore stop at `REQUESTED` or
`STARTED`; no final decision is invented for them.

After acknowledgment, the owner retains only the original user request as a
separately labelled `context_only` anchor (plus a bounded related-skill association),
not the acknowledged tool history. The anchor is counted in private payload and
episode limits and, when needed by a later correction, in selected-view bytes and
messages. The 16-KiB carry bound applies only after this source becomes a context-only
anchor; a complete first source may use the configured selected-view budget. An oversized
future anchor is omitted whole and replaced by a bounded `missing_context` record.
If a later revision needs that absent original context, preparation ends with an
explicit `BUDGET_REFUSED` and zero provider calls rather than sending a truncation.

This example was generated by a synthetic local test execution against a temporary
mailbox with the selected-view guard explicitly set to the former 65536-byte value;
it is **not the user's live job**. Opaque IDs and timestamps are omitted:

```text
Skill review: historical stage=service.preparation reason=selected_view_bytes profile=user-0 job=<synthetic-job-id> observed_bytes=205833 limit_bytes=65536; outcome=BUDGET_REFUSED; zero provider requests; accepted source references await owner acknowledgement.
[Skill Review] No skill was published: the bounded review request was refused before a provider call (reason=selected_view_bytes; observed=205833 bytes; limit=65536 bytes; provider requests=0). Source scope: bound episode source set (...). Review job <synthetic-job-id>.
```

The synthetic provider-call counter was `0`. These diagnostics do not change any
budget, token cap, timeout, eligibility/selection, retention, retry, or source
acknowledgement behavior.

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
| Selected review view | 256 KiB by default (`--selected-view-bytes`); 128 messages unchanged, including wrappers, IDs, provenance and omission records |
| Outstanding batch | One per profile, including retained disconnected results |
| Catalog prompt context | At most 128 summaries/16 KiB; overall catalog snapshot under 48 KiB |
| Eligible UPDATE context | At most 8 complete Markdown targets, each at most 12 KiB; never truncated |
| Supported catalog file | Read at most 256 KiB plus one overflow byte per file |
| Prepared user-content JSON | 512 KiB by default after composition (`--prepared-input-bytes`) |
| Complete SDK JSON body | 1 MiB by default including prompt, settings, envelope and escaping (`--wire-body-bytes`) |
| Model result | At most 24 KiB and the configured token cap |
| Private mailbox | 16-MiB SQLite page ceiling; DELETE rollback journal and per-connection journal-size limit |
| Terminal job history | Last 32 metadata outcomes; task/catalog/result bodies cleared at acknowledgement |
| Active review notices | One claimed and one pending set, each capped at 24 detailed lifecycle-event rows (starts plus finals) plus one explicit overflow summary; no evidence/proposal bodies |
| Delivered notice history | Last 32 delivered/invalid rows; independent of terminal job pruning |

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
`coordination_timeout_s=0.5` and `sqlite_busy_timeout_s=0.5`. `admission.queue`
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
Skill review: historical stage=provider.inference_validation error=ValueError profile=user-0 job=85f04ad2b89e43bcb63c4caf1c9da8c3 reason=non_stop_finish request_attempted=true response_received=true finish_reason=length output_tokens=4096; outcome=INVALID; no automatic provider retry; awaiting result persistence and owner acknowledgement.
Skill review: historical stage=result.persistence error=TimeoutError profile=user-0 job=db6d37c0463d488eb07605c5a5eb7be6 coordination_timeout_s=5; RUNNING attempt interrupted; no active retry; resolve coordination/storage failure and restart the service to recover.
```

| Stage | Consequence and operator action |
| --- | --- |
| `pending.scan` / `pending.recovery` | Scanning or recovery was interrupted. Admitted work remains. `run` automatically tries again on a future loop; `once` promises no future run. Check local coordination/storage if repeated. |
| `claim` | Claiming the prepared job was interrupted. The same future-loop guidance applies; a lock timeout is local coordination, not a model timeout. |
| `worker.authorization` / `worker.start` / `worker.dispatch` | A claimed RUNNING attempt was interrupted before inference. `worker.start` is the signed first-entry persistence seam before local preparation. There is no active retry for that attempt. Resolve the local fault, stop the service, and restart it after its workers exit. |
| `provider.inference` | The provider call failed, including SDK `APITimeoutError` and HTTP status errors. The configured provider timeout is shown. The host SDK adapter reports only its known request/response observations; a persisted FAILED result is acknowledged by the owner and consumes that batch, and is not NONE. |
| `provider.validation` / `provider.inference_validation` | Output was rejected. The latter covers the host adapter that performs inference and validation together. Fixed reasons distinguish `prepared_input_invalid`, `prepared_input_oversized`, `wire_body_oversized`, `no_choices`, `non_stop_finish`, `missing_or_nontext_output`, `output_invalid_encoding`, `output_oversized`, `malformed_json`, `duplicate_fields`, `proposal_not_object`, `invalid_action_or_fields`, `incomplete_complete_flag`, `empty_fields`, `invalid_name`, `invalid_description`, `incomplete_instructions`, and `safety_rejection`. The existing INVALID/FAILED outcomes remain. |
| `result.persistence` | The result could not be durably recorded. A RUNNING attempt may remain even after later scans succeed. Resolve the local fault and restart the service; those scans do not retry this attempt. |
| `service.startup.*` / `service.status` / `service.shutdown` | The command reports a safe exception class and launch/storage guidance, without raw exception text. A busy singleton reports its zero-wait setting; let the prior service exit before restarting. |

Timeout fields describe the configured operation whose failure was observed.
Lock acquisition uses 5 seconds in claim, worker authorization and result
persistence; short foreground/scan coordination and SQLite busy waits use 0.5
seconds. An exception raised outside a known timed
operation does not acquire a guessed timeout. SDK timeouts remain transport
settings, not an end-to-end deadline. No retry or recovery policy changes here.

Host adapter diagnostics can include `request_attempted`, `response_received`, the
configured `output_tokens`, measured byte counts and limits, bounded provider usage
counts, and an allowlisted `finish_reason`. `request_attempted=true` means only that
invocation of the resolved SDK `create` callable was attempted, not that remote
execution occurred. Preflight and SDK setup failures report
`request_attempted=false response_received=false` because they occur before that
invocation. `response_received=true` is limited to a known SDK response: either a
successful response or a trusted SDK HTTP status exception. Omission of
`response_received` means the observation is unknown, not false; create-time parsing
errors and timeouts therefore do not assert that no response arrived. Output checks
report a known received SDK response. Unknown finish reasons use an invalid sentinel, and
unavailable or invalid counts are omitted. Opaque/custom adapter `ValueError`
instances retain the safe generic fallback: their text and claimed metadata are not
serialized. No response text, request content, provider bodies/headers, paths, SQL,
tracebacks, or arbitrary exception text is included. The same sanitized detail is
written to the terminal job after owner acknowledgement; a diagnostic INVALID still
consumes the batch and is not retried automatically.

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
unavailable. No class is reconstructed from arbitrary returned text.

Diagnostics omit evidence, provider bodies/headers, credentials, SQL, full paths
and traceback bodies. Known exception classes are allowlisted; custom classes use
a safe base class. Invalid or oversized profile/job identifiers become `<invalid>`;
discovery child labels are bounded and restricted to safe ASCII. Diagnostic data
uses the existing console, in-memory error and authorized terminal-job metadata
paths. The lifecycle JSONL described above is the sole added file log; no raw
diagnostic file logger or post-revocation profile diagnostic writes are added.

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
| Host lifecycle notices and safe terminal delivery | `test_skill_review_notices.py` and real subprocess cases in `test_skill_review_process.py`; authorized worker starts, final outcomes, source scope, seals, recovery/reconnect, print/ack crash windows, mode races, event overflow summaries, partial input, foreground responses and permission prompts |
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
