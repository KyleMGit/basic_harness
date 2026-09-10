# Selective Skill Review and Bounded SQL Evidence — Approval Plan

**Status:** Design proposal only. This session has made no production-code edits and has launched no writing agent, service, live provider request, database connection, or real harness-profile operation. Concurrent diagnostic edits by another workstream were detected and preserved; see the baseline note below.

**Goal:** Automatically review worthwhile related work, not each completed chat turn, while supporting realistic SQL evidence without moving overflow to another layer.

**Architecture:** Host-owned, generation-bound episode state feeds the existing owner preparation thread and single shared inference service. Deterministic positive signals make an episode eligible; the existing reviewer makes the semantic CREATE/UPDATE/NONE decision. Review views are explicitly selected evidence, not invented complete transcripts.

**Implementation gate:** Obtain the user's approval before Codex writes code. Leave all implementation changes uncommitted. Final review must be Claude Opus 5, verified by runtime model identity, against the frozen exact tree.

## 1. Verified baseline and measurements

Repository: `C:/Users/Owner/.gemini/antigravity/scratch/coding_agent`

Inspected HEAD: `332e9add6e194b4c5a5261d8523851d2618999ac`; initially clean. There is no completed 16-to-64-KiB change in the working tree.

**Concurrent-work note:** After the synthetic measurements, the source hash check detected changes to `skill_review.py`; a later read-only Git inspection also showed `agent.py` modified and new `test_skill_review_diagnostics.py`. The inspected diff adds stage-specific, privacy-safe diagnostics and caller wording; the per-turn admission and existing numeric limits remain. These edits are not this session's implementation, have not been independently tested here, and must be preserved. The source references, hashes and measurement table below describe the initial measured snapshot, not approval of the moving tree. Reuse/extend the settled diagnostic work instead of duplicating or overwriting it. Establish a fresh agreed snapshot before Codex starts.

- `agent.py:750–786,872,967–1015`: fresh per-run capture, before conversation compaction; each normal completion calls synchronous auto-memory and then skill admission. Earlier conversation and the separately injected skill instructions are not part of that turn's capture.
- `skill_review.py:62–129,538–559`: 16-KiB capture and independent 16-KiB admission cap; 64 messages; 512 traversal nodes/depth 12. Capture failure reasons collapse into OVERFLOW.
- `skill_review.py:575–624,683–802`: oldest up to four tasks under a 64-KiB sum of message bytes; wrappers/catalog are outside that sum. The service does not wait for four. A task above the batch cap could block selection indefinitely if only admission were enlarged.
- Queue: 128 unconsumed task records or 1 MiB. One active job per profile. Acknowledgment clears bodies and consumes included evidence; 32 terminal metadata outcomes remain.
- `skill_catalog.py:325–353`: summaries at most 128/16 KiB, up to eight complete eligible targets, each raw target at most 12 KiB, overall snapshot budget 48 KiB. Target selection reserves space for later summaries.
- `skills.py:188–198`: 128-KiB guard covers prepared user-content JSON, not the complete SDK request or reflection prompt. One proposal is returned for a batch.
- `compaction.py:262–311`: displayed tokens are a character-based estimate including a reserve. Redaction parses JSON inside strings; the outer capture traversal does not enforce the same node/depth bounds on those decoded values.

Read-only experiments executed the real `db_tools._serialize_result`, `Evidence.add/finish`, `TrajectoryLogger._safe`, `packed`, `request_json`, and `AutoSkillExtractor.generate_proposal`. The real OpenAI SDK used an in-memory HTTP transport with an explicitly synthetic NONE response. There was no network or catalog/profile mutation. Diagnostic larger-cap Evidence instances were disposable measurement objects, not source changes or tests of a fix.

| Synthetic fixture | Exact redacted message bytes | Messages | Current capture |
|---|---:|---:|---|
| One SQL preview exchange | 15,181 | 4 | READY |
| Four SQL preview exchanges | 60,022 | 10 | OVERFLOW |
| Eight SQL preview exchanges | 119,810 | 18 | OVERFLOW |
| Four Unicode-heavy SQL exchanges | 74,330 | 10 | OVERFLOW |
| Four escaping-heavy SQL exchanges | 83,606 | 10 | OVERFLOW |

Additional observed boundaries:

- 16,383 serialized bytes passed capture; 16,384 failed because current incremental accounting charges one extra byte.
- 64 tiny messages passed; 65 failed despite only 2,467 serialized bytes.
- Outer 600-node and depth-20 structures failed; the equivalent structures inside JSON strings passed. Decoded-input protection must be fixed together with the byte budget.
- 131,072 bytes of prepared JSON passed its guard and became a 132,495-byte SDK body. 131,073 was refused with zero requests.
- An escaping-heavy 130,048-byte prepared string became a 261,471-byte SDK body.
- A separate eight-exchange fixture was 119,206 message bytes. With a synthetic 48,655-byte catalog, prepared input was 167,957 bytes and was refused. A size-only, explicitly labelled row-omission experiment reduced its message evidence to 4,646 bytes, prepared input to 53,397, and SDK body to 56,950. This did not exercise a production selector, catalog preparation, queue, or process pipeline.

These are synthetic byte measurements, not a reconstruction of the user's query. The reported context movement from approximately 5k to 18k tokens is not used to derive any byte cap.

## 2. Proposed eligibility policy — local positive signals, not another model gate

A completed turn is only a capture boundary. It is not independently eligible merely because it completed, used tools, changed SQL, accumulated bytes, or exceeded an age/message count.

Create a candidate only when retained evidence supports at least one of these compound signals:

1. **Resolved procedural failure:** an observed SQL/dialect/function/type/structural failure, followed by a materially changed procedure and a successful non-metadata execution on the related scope. A successful metadata query alone does not resolve the original failure. Network/authentication/permission failures, unchanged retries, empty results alone, or parameter-only adjustments do not qualify.
2. **Verified procedural correction:** an explicit user correction to an approach or existing procedure, followed by an observed non-routine implementation change and successful execution/verification. “Two months instead,” other routine parameters, and acknowledgments are not this signal.
3. **Investigated reusable workflow:** observed investigation of schema/join/dialect/tool behavior, use of a nontrivial procedure derived from it, and successful execution/verification. Examples include a discovered join path or dialect workaround. Merely running metadata and an ordinary aggregation is insufficient.

Group signals concerning the same procedure into one learning candidate. A new signal must add a material evidence delta; it must not be the same failure/success chain rediscovered on every subsequent turn.

Implementation uses bounded host event records from actual tool invocation/results, existing skill retrieval/load information, conservative SQL/tool signatures, and explicit correction/continuation cues. No SQL system-prompt change or model-emitted eligibility field is required. Normalize recognized literal/date/filter values, sort/limit choices, formatting, and routine grouping variations for repetition suppression, while retaining procedure-bearing joins, subqueries, functions, casts and backend identity. Unsupported or ambiguous syntax is UNKNOWN, not proof of novelty. A changed signature alone never qualifies.

Reuse already computed private catalog/loaded-skill information and a bounded profile-local fingerprint index; do not introduce a catalog scan or inference call after every message. Exact/recognized repetitions can be suppressed locally. Keyword similarity and fingerprints cannot prove semantic coverage; ambiguous but positively evidenced candidates can reach the existing reviewer, which may return NONE.

**Limitations:** conservative heuristics will miss some novel procedures and occasionally send already-covered work. There is no random sample of trivial turns, periodic semantic sweep, or every-message eligibility model. A later substantive correction can make a previously skipped episode eligible. Persist counts/reasons so policy misses and false positives can be evaluated without retaining raw diagnostic evidence.

## 3. Episode lifecycle and dispatch

### Identity and association

- An episode begins with a new work request, including the original question before clarification or tools. A trivial-only sequence remains an unreviewable contextual draft.
- Identity is `(host-bound profile/store, authorization generation, session, episode ID)` with persisted turn/candidate/revision IDs. Models cannot set these fields.
- Explicit continuations, clarification answers, and recognized routine variants attach to the current related work. Actual backend/resource/procedure observations confirm association. A clearly unrelated request or explicit topic switch closes the old episode and starts another.
- Where association is uncertain, retain bounded antecedents and mark the association uncertain; do not infer a new skill from uncertainty. No perfect semantic topic detector is claimed.
- Keep the original request, necessary clarifications, relevant loaded procedure references, and the latest verified baseline. Related corrections add to this chain. Never reconstruct missing evidence from the model's compacted session summary or synthetic interrupted-tool messages.

### Retention and review revisions

Use separate states for contextual draft, eligible candidate, sealed ready work, existing PREPARED/RUNNING/RESULT jobs, and acknowledged terminal outcomes.

Context-only routine follow-ups may replace older routine baseline context as whole units, with an explicit omission manifest. This is documented contextual-buffer retirement, not eviction of admitted review work. Once a candidate is eligible, pin its supporting source units until acknowledgment/cancellation. Store new completed-turn deltas rather than repeatedly enqueuing the whole conversation.

Seal an immutable review revision only when BOTH a positive eligibility signal exists AND one of these dispatch boundaries occurs:

- **Two minutes idle** after a completed turn;
- a clear topic/session change;
- normal exit;
- **15 minutes since the oldest undispatched eligible signal**, at the next completed safe turn boundary, for continuously active conversations.

Idle/age thresholds control timing only. They never make trivial evidence eligible. The owner knows whether a foreground turn is running; it must not seal a partial exchange while the user is still working.

After sealing, routine follow-ups do not create another dirty revision. New substantive corrections can create a delta revision with bounded antecedents. One coherent procedure-focused candidate is reviewed per request, rather than mixing four unrelated eligible episodes into a single-proposal request. Independent procedures may require distinct candidates; shared evidence is retained until all referencing candidates terminate.

### Exit, idle, restart, crash, revocation

- **Connected idle:** the existing owner thread seals due candidates, prepares snapshots, and applies results. No new service/timer process.
- **Long conversation:** the maximum eligible deferral prevents starvation; raw capacity pressure does not itself create eligibility. Routine-only material is retired under the documented context policy.
- **Normal exit:** locally seal and durably mark eligible work ready, without waiting for inference. Already prepared jobs can run while disconnected. Work not yet prepared awaits the next authorized owner connection because only the owner can prepare catalog context; no promise of offline catalog preparation by the service.
- **Crash:** recover committed episode/turn boundaries. A partial in-memory turn may be lost or marked interrupted, never fabricated as complete. Previously eligible completed work survives; pending correction chains can continue after resume.
- **Reconnect/resume:** same generation resumes episode IDs and acknowledgment watermarks. A fresh session can close prior completed eligible drafts; an explicitly resumed session can continue its unsealed draft. Already sealed/consumed revisions cannot be replayed as new learning.
- **Revocation/modes:** invalidate episodes and jobs through the existing generation fence before acknowledging the mode change. No profile cleanup writes after revocation. Authorized reenable purges obsolete generations, never reviving canceled evidence. Plain disconnect preserves the generation.
- Existing provider-error/invalid-proposal consumption policy remains explicit at owner acknowledgment. Remote inference can still be repeated after a service crash; do not promise exactly-once provider execution. Owner publication and consumption remain receipt/idempotency protected.

## 4. Evidence capture, selection, and budgets

### Explicit review views, without an inference summarizer

Keep a bounded, redacted source capture separate from its selected review view. Normalize native/XML tool execution into host evidence with actual executed arguments and call IDs; retain all results of a multi-call group together. Repair prompts and recovered placeholders must not masquerade as user requests or successful execution evidence.

For a review view retain:

- original question and required clarification/correction messages;
- full redacted SQL/tool arguments, observed errors and the complete relevant repair/verification groups;
- full metadata/schema results, since their rows may be the actual learning;
- source completeness/truncation flags and relevant loaded-skill identity/revision context;
- complete relevant assistant messages when needed and within budget.

For a **recognized successful business-data SQL preview larger than 4 KiB**, propose replacing its entire row array in the REVIEW VIEW with an explicitly labelled host projection: original result kind, columns, returned-row count, original truncation flags, omitted-row count/reason, and source serialized size. Retain the original bounded redacted result in the pending private source record. Small results are retained whole. Metadata results, errors, unknown tool formats and row-dependent validation evidence are not treated as ordinary business previews.

This projection preserves the tool-call/result relationship, NOT every original result value. It is not a fabricated original tool response or a complete transcript. When classification is uncertain, retain the full exchange or refuse it. Evidence without rows cannot justify row-dependent business/numerical claims or prove semantic query correctness; the reviewer may learn only procedures supported by what it actually receives.

Omit routine repetitions and unrelated exchanges as complete units. Every view declares included source IDs and omitted categories/counts. Do not slice arbitrary strings, partly retain a multi-tool group, or treat a source preview's returned-row count as total database rows. Oversized indispensable user text, metadata, error chains, or dependent tool groups are honestly refused if no complete supported view fits.

There are **zero added summarizer/eligibility model calls**. Selection/redaction runs locally; catalog work remains in the owner background thread. The reflection prompt, not the SQL system prompt, will describe the view's provenance and limits.

### Proposed ceilings for approval

These are coherent starting limits grounded in the experiments, not a claim that all possible SQL questions fit. Centralize them and verify exact serialization at each boundary.

| Layer | Proposed policy |
|---|---|
| Per-turn redacted source | 256 KiB serialized, 256 messages |
| Per-episode retained source | 512 KiB serialized, 512 messages, including pinned antecedents |
| Input safety | 256-KiB raw per-message ceiling before decoding; at most 8,192 decoded nodes and depth 24 per message, including JSON embedded in strings; cumulative byte/message bounds still apply |
| Context-only carry-forward anchor | At most 16 KiB in a review view; required larger context must fit through a complete candidate view or be explicitly refused, not silently shortened |
| Private pending capacity | 128 episode/legacy units and 8 MiB total live payload, including contextual anchors, retained sources and prepared-job copies; reserve 512 KiB within that total for preparation/results so admission cannot fill all headroom |
| Repetition index | At most 1,024 private fingerprints / 128 KiB, bounded retention; eviction may cause a later false-positive review, never revive a consumed revision |
| One review's evidence view | 64 KiB INCLUDING candidate IDs, provenance, omission manifest, antecedents and group wrappers; at most 128 projected messages |
| Catalog | Preserve current 48-KiB overall, 16-KiB/128 summaries, 8 complete targets / 12 KiB each; whole optional entries may be omitted with coverage flags to fit the actual request |
| Prepared user-content JSON | Keep 128 KiB, measured after full composition |
| Complete outbound SDK JSON body | Add exact 256-KiB guard, including reflection prompt, model/settings and escaping; verify the bytes actually sent, not a differently serialized estimate |
| Output/history | Preserve 24-KiB output, current service token/timeout settings, and 32 terminal metadata outcomes |

For queue storage, enforce both logical payload accounting and a SQLite page ceiling of 16 MiB (using actual page size), with rollback-journal storage bounded by that database ceiling. No unbounded sidecar evidence spool. Reserve/check preparation headroom and report storage-capacity failures without deleting older work. A pre-existing database above a new physical ceiling must be preserved and reported, not vacuumed, reset, or truncated automatically. The page limit is for the private review mailbox, not existing trajectory history/catalog storage; no claim of an OS-wide disk quota.

Keep extra reflection memory bounded by the raw/decoded limits, one active turn buffer, bounded episode reads, and one prepared job per profile. Do not load all pending source bodies to count capacity. Apply node/depth limits to decoded JSON as well as outer structures, and correct the measured off-by-one accounting.

Prepared bytes and wire bytes are NOT model tokens. The existing approximate token estimator is not a tokenizer or proof of fit in a 40,960-token context. Projection should reduce normal requests substantially, but actual provider context rejection remains an explicit failure; no silent model switch, tokenizer download, or extra inference retry is introduced. Fake endpoints prove harness limits, not local-model semantic quality/context acceptance.

### Progress and overflow

Fit evidence first, then bounded catalog context, then validate the complete request before dispatch and again at the service boundary. Prefer one complete projected candidate. If an episode has independent learning candidates, prepare separate candidate views, with only their required antecedents; this can cost more than one review for a genuinely multi-procedure episode. Do not split a dependent correction chain into separately presented “complete” histories merely to meet a cap.

If an indispensable candidate cannot fit even alone, produce an explicit terminal `BUDGET_REFUSED` result with zero provider calls. Retain source until its owner acknowledgment, then consume only that candidate's references under the documented terminal policy. Continue to the next oldest ready candidate. Queue-full refusals reject new admission and retain previously accepted work. Transient preparation/storage failures retain accepted work for recovery, rather than masquerading as terminal size failures.

Diagnostics: reason codes for raw/serialized capture bytes, messages, decoded nodes/depth, incomplete exchanges, episode capacity, queue count/payload/page capacity, evidence-view size, catalog fit, prepared input and final wire size. Report byte/count/limit/omission/request counters only—no SQL, rows, error bodies, credentials, or raw value hashes in diagnostic output. Normal skips remain quiet. CREATE/UPDATE/NONE chat notifications remain excluded.

## 5. Implementation and verification after approval

One bounded implementation effort, not review after each small patch:

1. Recheck status/hashes and preserve concurrent edits. Establish the versioned episode/candidate and budget contracts in tests. Have Codex record caller/scheduler RED failures before production changes.
2. Add host event capture/association/eligibility and durable episode transitions; integrate completion, busy/idle, resume, exit and mode revocation without changing synchronous auto-memory ordering.
3. Add bounded source/redaction handling, explicit projections, exact request budgets, fair terminal refusal/acknowledgment and diagnostics. Keep service/owner authorization and publication boundaries.
4. Add backward-compatible owner-only schema initialization. Existing accepted evidence/PREPARED/RUNNING/RESULT jobs retain their original processing/consumption contract; do not retroactively discard them using the new gate or backfill from whole history. Unknown newer schema versions fail closed.
5. Update docs and run focused tests and the full relevant suite. Independently rerun them as parent; freeze hashes/status; obtain one broad read-only Claude Opus 5 review. Maintain one blocker list and use narrow closure reviews for validated blockers only.

Likely files: `agent.py`, `skill_review.py`, `skills.py`, `skill_catalog.py`, new `skill_episodes.py`, `docs/async_skill_review.md`, relevant README text, and tests. Reuse redaction primitives; alter `compaction.py`/`storage.py` only if a concrete shared-boundary necessity is demonstrated, otherwise keep the new bounded evidence logic local. `db_tools.py` is measurement context, not an approved query-behavior change.

Test work includes `test_skill_episodes.py`, `test_skill_evidence_budget.py`, and extensions to `test_async_skill_review.py`, `test_skill_review_integration.py`, `test_skill_review_process.py`, `test_profile_discovery.py`, `test_profile_discovery_process.py`, `test_review_policy_scope.py`, `test_skill_publication.py` and existing profile/mode tests. Include the concurrently added `test_skill_review_diagnostics.py` after its workstream settles, retaining its privacy and stage-reporting coverage while updating old cap assertions only as required by the approved budget policy.

Required real caller/scheduling acceptance:

- Main SQL work plus date/filter/sort/limit/format/routine-grouping follow-ups: no request per turn, including follow-ups arriving after an earlier review completes. Initial simple work may produce zero; eligible initial work yields one consolidated candidate review absent new substantive learning.
- Trivial-only and clarification-only sequences: zero review and eligibility-model requests after idle, maximum age, exit and reconnect.
- Substantive correction: positive host evidence, retained antecedents, eventual one review under each approved dispatch trigger. Metadata-only success cannot clear a failed business query.
- Every-message counters distinguish primary-agent, existing synchronous memory, review and context-compaction requests. No new hidden model call.
- Large SQL-like raw evidence above the old 16-KiB cap and above a single review-view size progresses through actual agent capture, owner preparation, separate service process, fake loopback HTTP SDK, owner acknowledgment and retained-source cleanup. The fake endpoint receives the labelled projection and checks call/result IDs and omissions.
- Native/XML parity, actual executed arguments, multi-call groups, Unicode/escaping, byte boundaries ±1, message boundaries, nested JSON structural limits, catalog/batch/wire overhead and zero-request refusal.
- Queue/pinned-evidence bounds, reservation of preparation space, crash during admission/ack, source references shared by candidates, unfit oldest work followed by schedulable work, service interruption and restart counters.
- Same-generation reconnect does not repeat consumed revisions; revoked generations never revive. Read-only/stateless/no-skills startup and runtime transitions have zero forbidden profile writes. Preserve owner-only publication, receipt recovery and automatic profile discovery.
- No evidence, catalog, fingerprint or result leakage across profiles, including active global manager rebinding and adding profiles to a running service.

Focused command (after test files exist):

`python -m pytest -q test_skill_episodes.py test_skill_evidence_budget.py test_async_skill_review.py test_skill_review_integration.py test_skill_review_process.py test_profile_discovery.py test_profile_discovery_process.py test_review_policy_scope.py test_skill_publication.py test_profiles.py test_skill_review_diagnostics.py`

Full relevant suite: `python -m pytest -q`, plus changed-file syntax checks and `git diff --check`. Use isolated temporary profiles, credentials explicitly set to fake local test values, fake endpoints only. Public subprocess tests must not depend solely on mocked eligibility helpers.

## 6. Tradeoffs, effort, and operator state

- Conservative eligibility saves calls but can miss learning; explicit request/reason counters allow later evidence-based tuning. NONE remains the semantic safeguard for false positives, not an operational failure.
- Two-minute idle batching adds latency. The 15-minute eligible-age boundary prevents an indefinitely active conversation from deferring learning forever.
- Row projection sacrifices large business result values in the review view, not tool linkage. It supports procedural learning, not proof of numeric/business correctness. Oversized indispensable evidence still has an honest refusal path.
- Larger source buffers increase private local storage and bounded I/O. This is not zero-cost foreground admission; the existing synchronous auto-memory call still blocks as before.
- A disconnected owner cannot prepare fresh catalog snapshots; unprepared work waits for reconnect. Exactly-once remote inference across crashes is not promised.

**Estimate:** roughly 3–5 focused development days including RED/GREEN, real process-boundary tests, independent verification and a bounded final review/closure pass. This is a small durable-state feature, not a one-line cap change. Provider/reviewer access problems could extend elapsed time.

**One implementation-blocker list:** (1) user approval of the proposed eligibility, timing, projection and budget policies; (2) settle/reconcile the concurrent diagnostic edits before handing overlapping files to Codex. The earlier tool-consent blocker is resolved. No named-review model probe or implementation review has been run yet.

**Future migration/restart:** expect additive, versioned private mailbox schema changes performed only by an authorized owner. Do not support mixed old/new owner/service binaries processing new episode jobs. A coordinated owner/service restart will be documented for later deployment, without revocation or queue deletion merely to upgrade. No production restart, install, migration, live-provider test, real-profile modification, or commit is authorized by this planning step.

**Current evidence state:** source inspected; synthetic in-memory serialization experiments executed. No implementation, RED/GREEN/full-suite result, process-level fix verification, or reviewer approval is claimed.

### Inspected source hashes (SHA-256)

- `agent.py`: `cd6a953843eb46cef4a0ad63dfbb8d741bff8dd2f63bd5f8456e1bc2059a240d`
- `skill_review.py`: `0563d60ed3ea648f29383d377922b301b602b4d5ae9cd6b716bb39907de5a449`
- `skills.py`: `6030ec2095ae6ca1c9d22132acf2fcacb09331899332633f7172f09fc50fdda1`
- `skill_catalog.py`: `be7effb2aa1fa0620af2b4f9664c4ee59c81c381016650bf3122346786613dbb`
- `compaction.py`: `80ce1c6dec541dd60b38aeb69959cd388da66084085a70b1311d09d8aab2a21f`
- `storage.py`: `ed04f988bd92a424c9a2ea8183574da5bb5166b724d4263c65b207d4ec3f645f`
- `db_tools.py`: `ce3650625ac50f6b1704be7fd643449cc512090d1c4c55d20ed5812a336062c3`
- `docs/async_skill_review.md`: `de4d97227eafa7b5955661cdfd1b20c3094322cc1a73c097e387a723e7cecb75`
