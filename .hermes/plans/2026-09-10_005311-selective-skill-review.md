# Selective Skill Review — Bounded Approval Plan

**Repository:** `C:/Users/Owner/.gemini/antigravity/scratch/coding_agent` — standalone harness, not Hermes desktop.

**Status:** Approved by the user for implementation. The originally proposed higher limits are included. Measurement verifies those limits; there is no separate cap-approval gate. This bounded plan replaces the expanded architecture specification. Implementation/test/reviewer outcomes will be reported separately; approval of this plan is not a claim of completion.

## 1. Changes in scope

1. Review substantive learning across related turns, not every completed message.
2. Omit SQL business-result output from newly captured learning evidence.
3. Put heavy evidence assembly/request preparation in the existing shared review service.
4. Implement the coordinated higher limits below, with real boundary/process tests.

No generalized novelty engine, blanket removal of SQL-related assistant prose, legacy-job reprocessing project, new service/model, or additional approval phase.

**Inspected baseline:** HEAD `8f899ae1882f9ab7ec64365de7e4be425d41588d`; diagnostic improvements are committed, the cap fix is not. Existing capture/admission is 16 KiB per turn with 64 messages, 512 traversal nodes/depth 12; queue 128 tasks/1 MiB; batch evidence 64 KiB; catalog 48 KiB; prepared JSON 128 KiB. Normal opted-in turns attempt admission. Current capture includes full SQL tool responses. Recheck status/hashes before implementation and preserve concurrent work.

## 2. When review runs

Use deterministic host observations, not a new eligibility-model request. A completed turn qualifies only when it contributes evidence of:

- A meaningful SQL/dialect/procedural failure resolved through a non-routine change and successful execution.
- A substantive correction to an approach or existing procedure, followed by supporting execution/verification.
- An investigated reusable procedure actually used successfully, such as a discovered join path or dialect workaround.

Dates, filters, sorting, limits, formatting, routine grouping, clarifications, acknowledgments and ordinary repetitions do not independently qualify. Metadata success alone cannot resolve a failed business query. A changed SQL signature alone is not novelty.

Use conservative signatures and bounded repetition bookkeeping over actual new events. Unsupported/ambiguous syntax is not automatically novel. These heuristics can miss learning or admit duplicates; the existing reviewer makes the semantic CREATE/UPDATE/NONE decision. Later substantive evidence can make retained work eligible. No periodic semantic sweep or extra inference for selection/summarization.

## 3. Related episodes and dispatch

An episode starts with the original work request. Clarifications, explicit continuations and recognized routine variants attach to it using observed backend/resource/procedure context. Clear topic changes close it and start another. Uncertain associations retain bounded antecedents and are labelled uncertain, not treated as proof of learning.

Persist new turn deltas, the original question, necessary clarifications/corrections, relevant skill references and supporting tool exchanges. Do not repeatedly enqueue the growing conversation. Retire only unneeded routine context as whole units; pin eligible support until acknowledgment/cancellation.

**Dispatch requires positive eligibility plus:** two minutes idle after a completed turn, a topic/session change, normal exit, or fifteen minutes since the oldest eligible signal at the next completed-turn boundary. **A newly verified substantive correction becomes ready immediately.** Never seal an incomplete tool exchange.

Two minutes is consolidation, not a cooldown. Eligible initial work plus routine follow-ups produces one consolidated review; routine follow-ups after that review do not create another. Trivial-only work produces zero. A later substantive correction can produce a new revision. Review one coherent procedure per request; independent discoveries may require separate requests.

**Corrections after dispatch:** bind work/results to an episode revision and recheck freshness before owner publication. A recognized substantive challenge invalidates the affected pending revision; a failed/interrupted corrective turn must not silently restore it. Supersede unclaimed work; reject stale running/results without pretending to cancel remote inference. If a skill was already published, link the supported correction to it for a possible UPDATE. No automatic deletion/quarantine. Service capacity can still delay a ready correction.

**Lifecycle:**
- Idle/long sessions use the triggers above; age/capacity alone never creates eligibility.
- Normal exit durably marks complete eligible work ready without waiting for inference. Authorized work with a catalog snapshot can proceed disconnected; work lacking one waits for the owner to reconnect. Publication remains owner-only.
- Crash recovery uses committed complete boundaries; partial turns are lost/marked interrupted, never fabricated. Same-generation reconnect resumes IDs and acknowledgments without replaying consumed work.
- Preserve generation revocation, cancellation and mode write fences. Existing pending jobs retain their processing/consumption contract; no retrospective gate or bulk legacy rewrite. Omission applies to new capture, not a claim to retract old submissions.

## 4. SQL evidence and placement

**Omit business rows/previews and CSV contents before learning-evidence admission/persistence, regardless of result size.** Do not retain a large raw-result copy for later projection.

Keep necessary question/correction context, executed SQL and call IDs, relevant redacted errors, bounded schema/metadata findings, and observed success/failure/completeness indicators. Replace each omitted business result with an explicit labelled receipt tied to its call. Preserve complete call/result groups in native and XML protocols. Unknown/mixed output must not be passed through as metadata merely because the model labels it so.

Avoid duplicating output through identifiable result-bearing assistant messages: omit such messages whole with an omission marker, rather than slicing arbitrary strings. Do not introduce blanket removal of all SQL-related explanations or a general prose-scrubbing subsystem. Arbitrary user text/SQL literals can still contain business values; row omission is not comprehensive anonymization.

Execution success does not prove correct figures. Missing results cannot justify row-dependent claims; retain genuine existing verification facts or skip/refuse unsupported learning. No extra queries or summarizer model. Main-agent results, user answers, exports, history and synchronous auto-memory remain unchanged.

**Work split:** foreground Python performs bounded field omission/redaction and durable new-delta capture. The owner retains authorization, bounded lifecycle bookkeeping/catalog snapshots, publication and acknowledgment. The existing service performs episode assembly, whole-exchange selection and request composition using authorized private inputs—not unrestricted catalog access. Keep heavy work outside coordination locks and recheck generation/revision before dispatch/publication.

This reduces same-process preparation contention, not all overhead: capture, owner snapshots, disk/CPU and the shared model endpoint still cost time. Verify that a subsequent question progresses while service preparation is running.

## 5. Original higher limits — included in the implementation plan

These apply together; source capacity is not the amount automatically sent to inference.

| Boundary | Planned limit |
|---|---|
| Per-turn retained evidence, after business-output omission | **256 KiB / 256 messages** |
| Per-episode retained evidence, including pinned antecedents | **512 KiB / 512 messages** |
| Raw input inspection | **256 KiB per message before decoding**; omitted structured result fields need not be copied into this path |
| Structural traversal | **8,192 nodes / depth 24 per message**, including decoded JSON inside strings |
| Context-only carry-forward anchor in a review | **16 KiB**; indispensable larger context must fit as complete evidence or be refused |
| Private pending capacity | **128 episode/legacy units / 8 MiB total live payload**, including anchors, sources, bookkeeping and prepared copies |
| Preparation/result headroom | **512 KiB reserved within that 8 MiB** |
| Bounded repetition bookkeeping | At most **1,024 fingerprints / 128 KiB**, counted within the payload budget; eviction can permit a later duplicate review, not revive consumed work |
| One selected review evidence view | **64 KiB / 128 messages**, including antecedents, IDs, wrappers and omission records |
| Catalog snapshot | Preserve **48 KiB overall**, **128 summaries / 16 KiB**, **8 complete targets / 12 KiB each** |
| Prepared user-content JSON | **128 KiB**, measured after composition |
| Complete SDK-serialized outbound body | **256 KiB**, including prompt, envelope/settings and escaping |
| Output/history | Preserve **24-KiB proposal limit**, current service tokens/timeouts and **32 terminal metadata outcomes** |
| Private mailbox database | **16-MiB SQLite page ceiling**, with rollback-journal storage bounded by that ceiling; no unbounded sidecar spool |

Centralize limits and fix accounting at capture, admission, episode, queue, preparation and final request boundaries. Bound subordinate records and in-memory reads too; an episode heading must not hide unbounded deltas. Preserve an existing oversized mailbox rather than resetting/truncating it automatically.

**How large episodes progress:** the service selects complete relevant exchanges and required antecedents, omits whole optional repetitions with provenance, and includes complete catalog targets needed for updates. Separate genuinely independent learning where appropriate, never split a dependent chain into misleading histories. An indispensable candidate that cannot fit alone receives an explicit budget refusal with zero model calls; preserve accepted evidence until owner acknowledgment, then release only its references and let later fitting work progress. Queue-full admission rejects new work without deleting pending work. Transient storage/lock errors retain accepted work for recovery.

**Measurement is verification, not another approval gate.** Prior synthetic omission reduced one fixture from 119,206 evidence bytes to 4,646; another measured 130,048-byte prepared string became a 261,471-byte SDK body. These explain why both omission and exact end-to-end accounting matter. Test the chosen limits with synthetic SQL/metadata/corrections, Unicode and escaping; do not derive bytes from the user's 5k → 18k context estimate. Report an actual incompatibility rather than silently raising limits further or claiming byte limits prove model-token fit.

Extend existing privacy-safe diagnostics to distinguish capture bytes, messages/structure, episode/queue/storage capacity and prepared/wire limits. Print reasons and measured counts/limits, never evidence/secrets. Normal skips remain quiet; CREATE/UPDATE/NONE chat notifications are excluded.

## 6. Implementation, verification and preserved boundaries

After design approval, **Codex implements with RED → GREEN at actual caller/scheduling boundaries**. Parent independently reruns focused tests and `python -m pytest -q`, plus syntax checks and `git diff --check`. Freeze the final tree, verify concurrent edits, then obtain **Claude Opus 5** review with runtime identity verified. Keep one concrete blocker list and narrow closure reviews. No review-per-patch loop.

Required tests, using isolated temporary profiles and a fake local endpoint:
- Request counters: eligible SQL plus all routine follow-up types is not reviewed per turn; trivial-only is zero; a substantive correction eventually reviews; no extra eligibility inference.
- Related-turn context, corrections before/during/after review/publication, complete native/XML exchanges and output omission.
- Large SQL-like work through actual capture → owner/service process boundary → real SDK/fake HTTP → acknowledgment; also large retained SQL/metadata, not just easy-to-drop rows.
- Bytes ±1, Unicode/escaping, messages, decoded structure, catalog/wrapper/wire overhead, redaction, queue/storage bounds and oldest-work progress.
- Next-question progress during preparation; measure remaining owner/capture overhead without claiming zero latency.
- Restart/reconnect/revocation, preservation of pending work, no duplicate consumption/publication, all write restrictions, automatic discovery and cross-profile isolation.

Likely files: `agent.py`, `skill_review.py`, `skills.py`, `skill_catalog.py`, `docs/async_skill_review.md`; a small episode helper only if needed. Extend existing reflection, owner/service, diagnostics, profile and process tests; add focused episode/budget tests where useful. No SQL query/system-prompt changes or unrelated cleanup.

Preserve one shared service, automatic profile discovery, private catalogs/queues, host identity/authorization, owner-only publication, generation recovery, read-only/stateless/no-skills, opt-in behavior and synchronous `--auto-memory`. Only minimal additive owner-authorized persistence changes needed for episodes; document compatibility/restart requirements without operating on real profiles or production services.

**Effort:** provisionally a few hours of agent-assisted work, with test/reviewer failures reported as concrete blockers rather than a fixed deadline. The earlier multi-day estimate is withdrawn.

**Deliverable after implementation:** actual request-frequency behavior, final limits/refusal results, real focused/full/process test output, exact-snapshot reviewer verdict, modified files and any restart/migration instructions. Leave everything uncommitted and undeployed unless separately authorized.
