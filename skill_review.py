"""One supervised inference service; private durable mailboxes; owner-only apply.

See docs/async_skill_review.md for authorization, limits and operator commands.
This module never routes using model output. A service receives only sealed,
prepared JSON; only Owner receives a bound catalog object.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import closing, contextmanager, ExitStack, nullcontext
from dataclasses import asdict, dataclass, field
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import signal
import sqlite3
import sys
import tempfile
import threading
import time
import uuid

from skill_catalog import CatalogSnapshot, ReviewTarget, validate_proposal, MAX_OUTPUT_BYTES
from skill_lock import CatalogLock, ProcessLock
from storage import TrajectoryLogger
from profile_paths import (DEFAULT_PROFILES_DIR, PinnedDirectory, ProfilePaths,
                           control_directory, default_model, default_base_url,
                           plain_stat, validate_profile_id)
from skill_lock import path_identity


BUSY_SECONDS = .05
MAX_RECORDS = 128
MAX_QUEUE_BYTES = 1024 * 1024
BATCH_COUNT = 4
BATCH_BYTES = 64 * 1024
HISTORY_COUNT = 32
ACTIVE = ("PREPARED", "RUNNING", "RESULT")
MAX_DIAGNOSTICS = 128
REJECTED = "NOT queued; no automatic retry; earlier queue work retained."
DIAGNOSTIC_STAGES = frozenset((
    "owner.authorization", "owner.mailbox", "owner.delivery", "owner.publication", "owner.acknowledgement",
    "owner.preparation", "owner.preparation_authorization", "owner.preparation_persistence",
    "pending.scan", "pending.recovery", "claim", "worker.authorization", "result.persistence",
    "service.startup.provider", "service.shutdown", "service.status",
))
EXCEPTION_NAMES = {
    "builtins": {"Exception", "OSError", "TimeoutError", "ValueError", "TypeError", "KeyError",
                 "RuntimeError", "RecursionError", "PermissionError", "FileNotFoundError",
                 "UnicodeEncodeError", "UnicodeDecodeError", "OverflowError", "ImportError", "ModuleNotFoundError"},
    "sqlite3": {"Error", "DatabaseError", "OperationalError", "IntegrityError", "ProgrammingError"},
    "json.decoder": {"JSONDecodeError"},
    "openai": {"OpenAIError", "APIError", "APITimeoutError", "APIConnectionError", "APIStatusError",
               "RateLimitError", "AuthenticationError", "PermissionDeniedError", "BadRequestError",
               "NotFoundError", "ConflictError", "UnprocessableEntityError", "InternalServerError"},
}


def diagnostic_id(value, *, job=False, child=False):
    pattern = r"[a-f0-9]{32}" if job else r"[A-Za-z0-9_.-]{1,64}" if child else r"[A-Za-z0-9_-]{1,64}"
    return value if type(value) is str and re.fullmatch(pattern, value) else "<invalid>"


def exception_name(exc):
    # Never inspect exception messages, reprs, requests, responses or headers.
    # Unknown/custom classes fall back to their nearest allowlisted base class.
    for cls in type(exc).__mro__:
        if cls.__name__ in EXCEPTION_NAMES.get(cls.__module__, set()):
            return cls.__name__
    return "Exception"


class DiagnosticFailure(Exception):
    """Transient safe context; never persisted as a new error schema."""
    def __init__(self, stage, exc, job_id=None, **timing):
        super().__init__("Skill review stage failed")
        self.stage, self.error = stage, exception_name(exc)
        self.job_id, self.timing = job_id, timing


@contextmanager
def diagnostic_stage(stage, *, job_id=None, **timing):
    try:
        yield
    except DiagnosticFailure:
        raise
    except Exception as exc:
        raise DiagnosticFailure(stage, exc, job_id, **timing) from None


@contextmanager
def diagnostic_gate(entry, stage, timeout=5, *, job_id=None):
    # Attach the configured lock timeout only to acquisition failures.
    with ExitStack() as stack:
        with diagnostic_stage(stage, job_id=job_id, coordination_timeout_s=timeout):
            stack.enter_context(gate(entry, timeout))
        yield


def error_detail(stage, exc, *, profile_id=None, job_id=None, use_context=True, **timing):
    error = exception_name(exc)
    if (use_context and type(exc) is DiagnosticFailure and type(exc.stage) is str
            and exc.stage in DIAGNOSTIC_STAGES and type(exc.error) is str
            and any(exc.error in names for names in EXCEPTION_NAMES.values())):
        stage, error = exc.stage, exc.error
        job_id = exc.job_id if exc.job_id is not None else job_id
        timing = exc.timing
    parts = [f"stage={stage}", f"error={error}"]
    if profile_id is not None:
        parts.append("profile=" + diagnostic_id(profile_id))
    if job_id is not None:
        parts.append("job=" + diagnostic_id(job_id, job=True))
    for key in ("coordination_timeout_s", "sqlite_busy_timeout_s", "provider_timeout_s"):
        value = timing.get(key)
        if type(value) in (int, float) and 0 <= value <= 120:
            parts.append(f"{key}={value:g}")
    return " ".join(parts)


def remember_diagnostic(mapping, key, value):
    mapping[key] = value
    while len(mapping) > MAX_DIAGNOSTICS:
        del mapping[next(iter(mapping))]


def packed(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class EvidenceResult:
    status: str
    session_id: str
    task_id: str
    messages_json: str = ""
    reason: str = ""
    observed: int | None = None
    limit: int | None = None
    partial: bool = False

    def diagnostic(self):
        reasons = {
            "raw_bytes": ("capture.traversal", "bytes"),
            "serialized_bytes": ("capture.serialized", "bytes"),
            "message_count": ("capture.serialized", "messages"),
            "depth": ("capture.traversal", "depth"),
            "nodes": ("capture.traversal", "nodes"),
            "unsupported_data": ("capture.traversal", None),
        }
        stage, unit = reasons.get(self.reason, ("capture.completion", None))
        reason = self.reason if self.reason in reasons else "incomplete_completion"
        detail = f"stage={stage} reason={reason}"
        if reason == "incomplete_completion" and type(self.observed) is int and 0 <= self.observed <= sys.maxsize:
            detail += f" observed_messages={self.observed} minimum_messages=2"
        if unit and type(self.observed) is int and type(self.limit) is int and 0 <= self.observed <= sys.maxsize and 0 <= self.limit <= sys.maxsize:
            detail += f" observed_{unit}{'>=' if self.partial else '='}{self.observed} limit_{unit}={self.limit}"
        if self.partial:
            detail += " (partial message traversal; lower bound, not a task total)"
        elif reason == "serialized_bytes":
            detail += " (capacity accounting includes a reserved separator byte)"
        elif reason in ("nodes", "depth"):
            detail += " (partial traversal; remaining data was not measured)"
        return detail


@dataclass(frozen=True)
class Admission:
    status: str
    detail: str = ""


class Evidence:
    """Incremental per-task capture; refusal rather than silently losing old data."""
    def __init__(self, session_id, task_id=None, max_bytes=16 * 1024, max_messages=64):
        self.session_id = str(session_id)[:128]
        self.task_id = task_id or uuid.uuid4().hex
        self.max_bytes, self.max_messages = max_bytes, max_messages
        self.messages = []
        self.byte_count = 2
        self.overflow = False
        self.refusal = ("", None, None, False)

    def add(self, message):
        if message.get("role") == "system" or self.overflow:
            return
        if message.get("role") not in ("user", "assistant", "tool"):
            return
        # Bound the traversal BEFORE redaction/serialization. No slicing secrets
        # mid-value; oversized content refuses the whole task explicitly.
        nodes, budget = 0, self.max_bytes

        def refuse(reason, observed=None, limit=None, partial=False):
            self.refusal = reason, observed, limit, partial
            raise ValueError("Evidence refused")

        def bounded(value, depth=0):
            nonlocal nodes, budget
            nodes += 1
            if nodes > 512:
                refuse("nodes", nodes, 512)
            if depth > 12:
                refuse("depth", depth, 12)
            if budget < 0:
                refuse("raw_bytes", self.max_bytes - budget, self.max_bytes, True)
            if isinstance(value, str):
                if len(value) > budget:
                    refuse("raw_bytes", self.max_bytes - budget + len(value), self.max_bytes, True)
                budget -= len(value.encode("utf-8"))
            elif isinstance(value, dict):
                for key, item in value.items():
                    bounded(key, depth + 1)
                    bounded(item, depth + 1)
            elif isinstance(value, list):
                for item in value:
                    bounded(item, depth + 1)
            elif value is not None and not isinstance(value, (bool, int, float)):
                refuse("unsupported_data")

        try:
            selected = {k: message[k] for k in ("role", "content", "tool_calls", "tool_call_id") if k in message}
            bounded(selected)
            safe = TrajectoryLogger._safe(selected)
            size = len(packed(safe).encode()) + 1
            if len(self.messages) >= self.max_messages:
                refuse("message_count", len(self.messages) + 1, self.max_messages)
            if size + self.byte_count > self.max_bytes:
                refuse("serialized_bytes", size + self.byte_count, self.max_bytes)
            self.messages.append(safe)
            self.byte_count += size
        except (ValueError, TypeError, RecursionError):
            self.overflow = True
            if not self.refusal[0]:
                self.refusal = ("unsupported_data", None, None, False)

    def finish(self, completed=True, finish_reason="stop"):
        status = "READY"
        unanswered = set()
        for message in self.messages:
            if message.get("role") == "assistant":
                unanswered.update(call.get("id") for call in message.get("tool_calls", []) if isinstance(call, dict))
            elif message.get("role") == "tool":
                unanswered.discard(message.get("tool_call_id"))
        if self.overflow:
            status = "OVERFLOW"
        elif (not completed or finish_reason != "stop" or unanswered or len(self.messages) < 2
              or self.messages[0].get("role") != "user"
              or self.messages[-1].get("role") != "assistant"
              or self.messages[-1].get("tool_calls")
              or not str(self.messages[-1].get("content") or "").strip()
              or "<tool_call" in str(self.messages[-1].get("content") or "")):
            status = "INCOMPLETE"
        refusal = ("incomplete_completion", len(self.messages), 2, False) if status == "INCOMPLETE" else self.refusal
        return EvidenceResult(status, self.session_id, self.task_id,
                              packed(self.messages) if status == "READY" else "", *refusal)


@dataclass(frozen=True)
class ProfileEntry:
    profile_id: str
    mailbox: str
    store_id: str
    binding: object = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class Roster:
    control_dir: str
    model: str
    base_url: str
    profiles: tuple[ProfileEntry, ...]

    def entry(self, profile_id):
        found = [p for p in self.profiles if p.profile_id == profile_id]
        if len(found) != 1:
            raise ValueError("Profile is not in the host-authorized roster")
        return found[0]


@dataclass(frozen=True, init=False)
class DiscoveryRoster(Roster):
    """Host-controlled immediate children, without granting learning permission."""
    def __init__(self, profiles_dir=DEFAULT_PROFILES_DIR, model=None, base_url=None):
        root = PinnedDirectory(profiles_dir)
        model = default_model() if model is None else model
        base_url = default_base_url() if base_url is None else base_url
        if not model or not re.match(r"^https?://", base_url):
            raise ValueError("Discovery requires a model and HTTP(S) endpoint")
        for name, value in dict(control_dir=str(control_directory(root.path)), model=model,
                                base_url=base_url.rstrip("/"), profiles=(), root=root,
                                known={}, errors={}).items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "control", PinnedDirectory(self.control_dir))

    def check_policy(self, *, create=False):
        self.root.validate(required=False)
        self.control.validate(required=False)
        expected = dict(mode="discovery", root=str(self.root.path), model=self.model, base_url=self.base_url)
        # Atomic replacement makes reads lock-free; enqueue must not wait on a
        # separate policy lock beyond its existing short catalog-gate budget.
        with CatalogLock(identity="discovery-policy:" + path_identity(self.root.path)) if create else nullcontext():
            path = external_path(self.control_dir, "policy.json")
            if path.exists():
                if json.loads(path.read_text(encoding="utf-8")) != expected:
                    raise ValueError("Discovery model/endpoint policy mismatch; revoke and stop before reconfiguring")
            elif create:
                atomic_json(path, expected)

    def entry(self, profile_id):
        validate_profile_id(profile_id)
        self.root.validate(required=False)
        if profile_id not in self.known:
            binding = ProfilePaths(self.root, profile_id)
            root = binding.directory.path
            self.known[profile_id] = ProfileEntry(profile_id, str(root / "skill_review.db"),
                                                 path_identity(root / "skills"), binding)
        entry = self.known[profile_id]
        entry.binding.validate(required=False)
        return entry

    def refresh(self):
        admitted, identities = [], set()
        try:
            self.check_policy()
            if self.root.validate(required=False):
                with os.scandir(self.root.path) as children:
                    for child in children:
                        try:
                            entry = self.entry(child.name)
                            entry.binding.validate()
                            identity = tuple(entry.binding.directory.stamp)
                            if identity in identities:
                                raise ValueError("Duplicate profile directory identity")
                            identities.add(identity)
                            admitted.append(entry)
                            self.errors.pop(diagnostic_id(child.name, child=True), None)
                        except (OSError, ValueError) as exc:
                            remember_diagnostic(self.errors, diagnostic_id(child.name, child=True),
                                                "historical " + error_detail("discovery.child", exc, profile_id=child.name)
                                                + "; child skipped; check profile directory identity and permissions.")
            self.root.validate(required=False)
            self.errors.pop("root", None)
        except (OSError, ValueError) as exc:
            remember_diagnostic(self.errors, "root", "historical " + error_detail("discovery.root", exc)
                                + "; discovery unavailable; check root identity and permissions; restart after resolving replacement.")
            admitted = []
        object.__setattr__(self, "profiles", tuple(sorted(admitted, key=lambda p: p.profile_id)))
        return self.profiles


def load_roster(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(data) != {"control_dir", "model", "base_url", "profiles"}:
        raise ValueError("Roster requires control_dir, model, base_url and profiles")
    if not Path(data["control_dir"]).is_absolute():
        raise ValueError("Roster paths must be absolute")
    control = str(Path(data["control_dir"]).resolve())
    profiles = []
    for item in data["profiles"]:
        if (set(item) != {"profile_id", "mailbox", "store_id"}
                or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", item["profile_id"])
                or not re.fullmatch(r"[a-f0-9]{64}", item["store_id"])
                or not Path(item["mailbox"]).is_absolute()):
            raise ValueError("Invalid host roster entry")
        entry = ProfileEntry(item["profile_id"], str(Path(item["mailbox"]).resolve()), item["store_id"])
        if Path(control).is_relative_to(Path(entry.mailbox).parent):
            raise ValueError("Host coordination state must be outside protected profiles")
        for other in profiles:
            a, b = Path(entry.mailbox).parent, Path(other.mailbox).parent
            if entry.profile_id == other.profile_id or entry.store_id == other.store_id or a.is_relative_to(b) or b.is_relative_to(a):
                raise ValueError("Roster profiles must have distinct private identities and directories")
        profiles.append(entry)
    if not data["model"] or not re.match(r"^https?://", data["base_url"]):
        raise ValueError("Roster requires a model and HTTP(S) endpoint")
    return Roster(control, data["model"], data["base_url"].rstrip("/"), tuple(profiles))


def gate(entry, timeout=5):
    return CatalogLock(identity=entry.store_id, timeout=timeout)


def check_path(path):
    if str(Path(path).resolve()) != str(path):
        raise ValueError("Host-bound path identity changed")


def external_path(control_dir, name):
    PinnedDirectory(control_dir).validate(required=False)
    path = Path(control_dir, name)
    try:
        plain_stat(path)
    except FileNotFoundError:
        pass
    return path


def auth_path(roster, entry):
    return external_path(roster.control_dir, entry.store_id + ".json")


def authority_path(entry):
    return external_path(control_directory(Path(entry.mailbox).parent.parent), entry.store_id + ".authority.json")


def authority_origin(roster):
    return dict(mode="discovery" if isinstance(roster, DiscoveryRoster) else "roster",
                control_dir=roster.control_dir, model=roster.model, base_url=roster.base_url)


def read_authority(entry):
    path = authority_path(entry)
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(value, dict) or set(value) != {"origin", "generation"}
            or not isinstance(value["generation"], str) or not re.fullmatch(r"[a-f0-9]{32}", value["generation"])):
        raise ValueError("Malformed review mode authority")
    origin = value["origin"]
    if (not isinstance(origin, dict) or set(origin) != {"mode", "control_dir", "model", "base_url"}
            or not all(isinstance(item, str) and item for item in origin.values())
            or origin["mode"] not in ("discovery", "roster")
            or not Path(origin["control_dir"]).is_absolute()
            or Path(origin["control_dir"]).is_relative_to(Path(entry.mailbox).parent)
            or not re.match(r"^https?://", origin["base_url"])):
        raise ValueError("Malformed review mode origin")
    if origin["mode"] == "discovery" and origin["control_dir"] != str(control_directory(Path(entry.mailbox).parent.parent)):
        raise ValueError("Discovery authority control directory mismatch")
    return value


def read_current_auth(entry, authority):
    """Validate the current route's record before using it as revocation proof.

    A disabled record left in another mode's control directory proves nothing.
    This read does not select a policy for the caller or grant any permission.
    """
    origin = authority["origin"]
    path = external_path(origin["control_dir"], entry.store_id + ".json")
    if not path.exists():
        raise ValueError("Current review authorization is missing; revocation cannot be established")
    auth = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(auth, dict) or type(auth.get("enabled")) is not bool
            or (auth.get("profile_id"), auth.get("store_id"), auth.get("mailbox")) != (entry.profile_id, entry.store_id, entry.mailbox)
            or auth.get("authority_generation") != authority["generation"]
            or (auth.get("model"), auth.get("base_url")) != (origin["model"], origin["base_url"])
            or not isinstance(auth.get("generation"), str) or not re.fullmatch(r"[a-f0-9]{32}", auth["generation"])
            or not isinstance(auth.get("secret"), str) or not re.fullmatch(r"[a-f0-9]{64}", auth["secret"])):
        raise ValueError("Current review authorization identity/generation is invalid")
    if not auth["enabled"] and origin["mode"] == "discovery":
        binding = entry.binding or ProfilePaths(Path(entry.mailbox).parent.parent, entry.profile_id)
        if auth.get("incarnation") != binding.identity:
            raise ValueError("Authorized profile directory incarnation changed")
    return auth


def check_authority(roster, entry):
    """Read-only preflight, before owner provisioning; the catalog gate fences changes."""
    if isinstance(roster, DiscoveryRoster):
        roster.check_policy()
    value = read_authority(entry)
    if value and value["origin"] != authority_origin(roster):
        prior = read_current_auth(entry, value)
        if prior["enabled"]:
            raise ValueError("Prior review mode/model/endpoint is authorized; revoke with its configuration before switching")
    elif not value and isinstance(roster, DiscoveryRoster) and Path(entry.mailbox).exists():
        raise ValueError("Existing review mailbox has no mode authority; revoke using its static roster before switching")
    return value


def read_auth(roster, entry, *, for_revocation=False):
    if entry.binding:
        entry.binding.validate(required=False)
        roster.check_policy()
    path = auth_path(roster, entry)
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    authority = read_authority(entry)
    if authority and (authority["origin"] != authority_origin(roster)
                      or value.get("authority_generation") != authority["generation"]):
        raise ValueError("Review mode authority changed; old authorization cannot be reused")
    if entry.binding and not for_revocation and value.get("incarnation") != entry.binding.identity:
        raise ValueError("Authorized profile directory incarnation changed")
    if (value.get("profile_id"), value.get("store_id"), value.get("mailbox")) != (entry.profile_id, entry.store_id, entry.mailbox):
        raise ValueError("Authorization identity mismatch")
    if value["enabled"] and (value.get("model"), value.get("base_url")) != (roster.model, roster.base_url):
        raise ValueError("Authorized model/endpoint changed; revoke with the prior roster before reconfiguring")
    return value


def revoke_existing(roster, entry, *, owner_id=None):
    """Host-only external revocation, also before a disabled startup can fail.

    Incarnation validation is unnecessary for revocation: this operation grants
    nothing and never opens or writes a profile, including a missing profile.
    Model/endpoint and mode authority checks still apply.
    """
    with gate(entry):
        if entry.binding:
            entry.binding.validate(required=False)
            roster.control.validate(required=False)
        authority = read_authority(entry)
        if authority is not None:
            if not read_current_auth(entry, authority)["enabled"]:
                return  # Proven revoked: ordinary sessions must not rewrite policy/auth.
        elif isinstance(roster, DiscoveryRoster):
            if Path(entry.mailbox).exists() or auth_path(roster, entry).exists():
                raise ValueError("Existing review state has no mode authority; revoke using its static roster before switching")
            return  # This profile has never held review state; no policy to revoke.
        # Active authorization still requires its exact host mode/model/endpoint.
        authority = check_authority(roster, entry)
        if authority and authority["origin"] != authority_origin(roster):
            return  # Already revoked in its previous mode; do not switch modes.
        auth = read_auth(roster, entry, for_revocation=True)
        if auth:
            if not authority:
                authority = dict(origin=authority_origin(roster), generation=uuid.uuid4().hex)
            auth.update(enabled=False, generation=uuid.uuid4().hex, owner_id=owner_id,
                        secret=secrets.token_hex(32), authority_generation=authority["generation"])
            write_auth(roster, entry, auth)
            atomic_json(authority_path(entry), authority)


def write_auth(roster, entry, value):
    path = auth_path(roster, entry)
    atomic_json(path, value)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False) as handle:
            pending = handle.name
            handle.write(packed(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(pending, path)
    finally:
        if pending and os.path.exists(pending):
            os.unlink(pending)


def connect(entry, readonly=False, *, create=False):
    if entry.binding:
        entry.binding.validate()
    check_path(entry.mailbox)
    target = Path(entry.mailbox).as_uri() + ("?mode=ro" if readonly else "?mode=rwc" if create else "?mode=rw")
    conn = sqlite3.connect(target, timeout=BUSY_SECONDS, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


JOB_FIELDS = ("job_id", "profile_id", "store_id", "generation", "evidence_json", "catalog_json", "host_json")


def seal(auth, value):
    return hmac.new(bytes.fromhex(auth["secret"]), packed(value).encode(), hashlib.sha256).hexdigest()


def job_seal(auth, job):
    return seal(auth, {k: job[k] for k in JOB_FIELDS})


def result_seal(auth, job):
    return seal(auth, dict(input_seal=job["input_seal"], service_generation=job["service_generation"],
                           result_json=job["result_json"]))


def valid_job(auth, entry, job):
    return (auth and auth["enabled"] and (job["profile_id"], job["store_id"], job["generation"]) ==
            (entry.profile_id, entry.store_id, auth["generation"])
            and hmac.compare_digest(job["input_seal"], job_seal(auth, job)))


def request_json(roster, job):
    return packed(dict(catalog=json.loads(job["catalog_json"]), tasks=json.loads(job["evidence_json"])))


class Owner:
    """Lightweight preparation/delivery worker. It owns no inference or SDK client."""
    def __init__(self, roster, profile_id, store, *, enabled=True, background=True,
                 max_records=MAX_RECORDS, batch_count=BATCH_COUNT):
        self.roster, self.entry = roster, roster.entry(profile_id)
        self.store = store.bind()
        if self.store.store_id != self.entry.store_id:
            raise ValueError("Owner catalog does not match the host roster")
        if not 1 <= max_records <= MAX_RECORDS or not 1 <= batch_count <= BATCH_COUNT:
            raise ValueError("Queue limits exceed host bounds")
        self.max_records, self.batch_count = max_records, batch_count
        self.owner_id = uuid.uuid4().hex
        self.enabled = False
        self.closed = False
        self.last_error = ""
        self._wake, self._stop = threading.Event(), threading.Event()
        self._thread = None
        self.store.set_write_guard(lambda: self.enabled and not self.closed)
        self._owner_lock = ProcessLock("skill-owner:" + self.entry.store_id, timeout=0)
        self._owner_lock.__enter__()
        try:
            self.set_enabled(enabled)
            if background:
                self._thread = threading.Thread(target=self._loop, name="skill-mailbox-" + profile_id, daemon=True)
                self._thread.start()
        except BaseException:
            self._owner_lock.__exit__(None, None, None)
            raise

    def _authorized(self, auth):
        return not self.closed and self.enabled and auth and auth["enabled"] and auth.get("owner_id") == self.owner_id

    def _initialize(self, auth):
        if self.entry.binding:
            self.entry.binding.validate()
        check_path(self.entry.mailbox)
        if not self.entry.binding:
            Path(self.entry.mailbox).parent.mkdir(parents=True, exist_ok=True)
        with closing(connect(self.entry, create=True)) as conn, conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS evidence (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT UNIQUE, session_id TEXT,
                    generation TEXT, messages_json TEXT, bytes INTEGER, created REAL, job_id TEXT);
                CREATE TABLE IF NOT EXISTS jobs (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT UNIQUE, profile_id TEXT, store_id TEXT,
                    generation TEXT, evidence_json TEXT, catalog_json TEXT, host_json TEXT, input_seal TEXT,
                    created REAL, status TEXT, service_generation TEXT, result_id TEXT, result_json TEXT,
                    detail TEXT);
            """)
            # Reenable never revives an invalidated generation. Purge only here,
            # while authorized, never during/after the disabled acknowledgement.
            conn.execute("DELETE FROM evidence WHERE generation != ?", (auth["generation"],))
            conn.execute("DELETE FROM jobs WHERE generation != ?", (auth["generation"],))

    def set_enabled(self, enabled):
        with gate(self.entry):
            if self.closed:
                return
            if not enabled:
                self.enabled = False
                revoke_existing(self.roster, self.entry, owner_id=self.owner_id)
                self._wake.set()
                return
            authority = check_authority(self.roster, self.entry)
            switching = authority and authority["origin"] != authority_origin(self.roster)
            auth = None if switching else read_auth(self.roster, self.entry)
            if enabled:
                if self.entry.binding:
                    self.entry.binding.validate()
                    self.roster.check_policy(create=True)
                if not auth or not auth["enabled"]:
                    auth = dict(profile_id=self.entry.profile_id, store_id=self.entry.store_id,
                                mailbox=self.entry.mailbox, enabled=True, generation=uuid.uuid4().hex,
                                model=self.roster.model, base_url=self.roster.base_url,
                                secret=secrets.token_hex(32))
                if not authority or switching:
                    authority = dict(origin=authority_origin(self.roster), generation=uuid.uuid4().hex)
                auth["authority_generation"] = authority["generation"]
                if self.entry.binding:
                    auth["incarnation"] = self.entry.binding.identity
                auth["owner_id"] = self.owner_id
                write_auth(self.roster, self.entry, auth)
                atomic_json(authority_path(self.entry), authority)
                self._initialize(auth)
                self.enabled = True
        self._wake.set()

    def enqueue(self, evidence):
        if self.closed or not self.enabled:
            return Admission("DISABLED", "stage=admission.authorization; " + REJECTED)
        if evidence.status != "READY":
            status = evidence.status if evidence.status in ("OVERFLOW", "INCOMPLETE", "DISABLED", "FAILED") else "FAILED"
            return Admission(status, evidence.diagnostic() + "; " + REJECTED)
        stage = "admission.lock"
        timing = dict(coordination_timeout_s=BUSY_SECONDS)
        try:
            with gate(self.entry, BUSY_SECONDS):
                stage, timing = "admission.authorization", {}
                auth = read_auth(self.roster, self.entry)
                if not self._authorized(auth):
                    return Admission("DISABLED", "stage=admission.authorization; " + REJECTED)
                stage = "admission.serialized"
                size = len(evidence.messages_json.encode())
                if size > 16 * 1024:
                    return Admission("OVERFLOW", f"stage={stage} reason=serialized_bytes observed_bytes={size} limit_bytes=16384; " + REJECTED)
                stage, timing = "admission.sqlite", dict(sqlite_busy_timeout_s=BUSY_SECONDS)
                with closing(connect(self.entry)) as conn, conn:
                    conn.execute("BEGIN IMMEDIATE")
                    count, used = conn.execute("SELECT count(*), coalesce(sum(bytes),0) FROM evidence").fetchone()
                    if count >= self.max_records:
                        return Admission("OVERFLOW", f"stage=admission.queue reason=queue_records observed_records={count + 1} limit_records={self.max_records} (including rejected task); " + REJECTED)
                    if used + size > MAX_QUEUE_BYTES:
                        return Admission("OVERFLOW", f"stage=admission.queue reason=queue_bytes observed_bytes={used + size} limit_bytes={MAX_QUEUE_BYTES} queued_bytes={used} incoming_bytes={size}; " + REJECTED)
                    conn.execute("INSERT INTO evidence(task_id,session_id,generation,messages_json,bytes,created) VALUES(?,?,?,?,?,?)",
                                 (evidence.task_id, evidence.session_id, auth["generation"], evidence.messages_json, size, time.time()))
            self._wake.set()
            return Admission("ACCEPTED", "stage=admission.commit; durably committed to the private local queue; admission only, publication pending review.")
        except (OSError, ValueError, sqlite3.Error, TimeoutError) as exc:
            return Admission("FAILED", error_detail(stage, exc, **timing) + "; " + REJECTED)

    def _ack(self, conn, job, outcome):
        conn.execute("UPDATE jobs SET status=?,detail=?,evidence_json='',catalog_json='',host_json='',result_json='' WHERE job_id=? AND status='RESULT'",
                     (outcome.status, outcome.detail, job["job_id"]))
        conn.execute("DELETE FROM evidence WHERE job_id=?", (job["job_id"],))
        conn.execute("DELETE FROM jobs WHERE status NOT IN ('PREPARED','RUNNING','RESULT') AND seq NOT IN (SELECT seq FROM jobs ORDER BY seq DESC LIMIT ?)", (HISTORY_COUNT,))

    def pump(self):
        if not self.enabled or self.closed:
            return
        try:
            self._pump()
        except Exception as exc:
            if self.enabled and not self.closed:
                self.last_error = "historical " + error_detail("owner.mailbox", exc, profile_id=self.entry.profile_id) + "; owner background attempt interrupted; pending work retained; check local storage/catalog."

    def _pump(self):
        if not self.enabled or self.closed:
            return
        with diagnostic_gate(self.entry, "owner.authorization"):
            with diagnostic_stage("owner.authorization"):
                auth = read_auth(self.roster, self.entry)
            if not self._authorized(auth):
                return
            with diagnostic_stage("owner.mailbox", sqlite_busy_timeout_s=BUSY_SECONDS), closing(connect(self.entry)) as conn, conn:
                for row in conn.execute("SELECT * FROM jobs WHERE status='RESULT' ORDER BY seq").fetchall():
                    job = dict(row)
                    with diagnostic_stage("owner.delivery", job_id=job["job_id"]):
                        if not valid_job(auth, self.entry, job) or not hmac.compare_digest(job["result_id"], result_seal(auth, job)):
                            continue
                        result = json.loads(job["result_json"])
                        from skill_catalog import Publication
                        if result["status"] in ("INVALID", "FAILED"):
                            outcome = Publication(result["status"], result["detail"])
                        else:
                            host = json.loads(job["host_json"])
                            snapshot = CatalogSnapshot(host["store_id"], job["catalog_json"], tuple(ReviewTarget(**t) for t in host["targets"]))
                            with diagnostic_stage("owner.publication", job_id=job["job_id"]):
                                outcome = self.store.apply_review(result["proposal"], snapshot, job["result_id"])
                            if outcome.status in ("INVALID", "FAILED"):
                                # Returned error text has no trustworthy exception class.
                                outcome = Publication(outcome.status, "stage=owner.publication outcome=" + outcome.status
                                                      + "; publication refused; check catalog eligibility and local storage; exception detail unavailable.")
                    with diagnostic_stage("owner.acknowledgement", job_id=job["job_id"], sqlite_busy_timeout_s=BUSY_SECONDS):
                        self._ack(conn, job, outcome)
                if conn.execute("SELECT 1 FROM jobs WHERE status IN ('PREPARED','RUNNING','RESULT') LIMIT 1").fetchone():
                    return
                selected, size = [], 0
                for row in conn.execute("SELECT * FROM evidence WHERE job_id IS NULL AND generation=? ORDER BY seq LIMIT ?", (auth["generation"], self.batch_count)):
                    if size + row["bytes"] > BATCH_BYTES:
                        break
                    selected.append(dict(row))
                    size += row["bytes"]
                generation = auth["generation"]
        if not selected:
            return
        # Heavy catalog work is outside the foreground and carries immutable data.
        with diagnostic_stage("owner.preparation"):
            snapshot = self.store.prepare_review()
            evidence_json = packed([dict(session_id=r["session_id"], task_id=r["task_id"], messages=json.loads(r["messages_json"])) for r in selected])
            host_json = packed(dict(store_id=snapshot.store_id, targets=[asdict(t) for t in snapshot.targets]))
        with diagnostic_gate(self.entry, "owner.preparation_authorization"):
            with diagnostic_stage("owner.preparation_authorization"):
                auth = read_auth(self.roster, self.entry)
            if not self._authorized(auth) or auth["generation"] != generation:
                return
            job = dict(job_id=uuid.uuid4().hex, profile_id=self.entry.profile_id, store_id=self.entry.store_id,
                       generation=generation, evidence_json=evidence_json, catalog_json=snapshot.public_json, host_json=host_json)
            with diagnostic_stage("owner.preparation_persistence", job_id=job["job_id"], sqlite_busy_timeout_s=BUSY_SECONDS), closing(connect(self.entry)) as conn, conn:
                if conn.execute("SELECT 1 FROM jobs WHERE status IN ('PREPARED','RUNNING','RESULT') LIMIT 1").fetchone():
                    return
                conn.execute("INSERT INTO jobs(job_id,profile_id,store_id,generation,evidence_json,catalog_json,host_json,input_seal,created,status) VALUES(?,?,?,?,?,?,?,?,?,'PREPARED')",
                             tuple(job[k] for k in JOB_FIELDS) + (job_seal(auth, job), selected[0]["created"]))
                for record in selected:
                    conn.execute("UPDATE evidence SET job_id=? WHERE seq=? AND generation=? AND job_id IS NULL",
                                 (job["job_id"], record["seq"], generation))

    def _loop(self):
        while not self._stop.is_set():
            self._wake.clear()
            self.pump()
            self._wake.wait(.1)

    def close(self):
        if self.closed:
            return
        with gate(self.entry):
            diagnostics_allowed = self.enabled
            self.closed = True
            self.enabled = False
            try:
                auth = read_auth(self.roster, self.entry)
                if auth and auth.get("owner_id") == self.owner_id:
                    auth["owner_id"] = None  # Disconnect preserves generation/results.
                    write_auth(self.roster, self.entry, auth)
            except (OSError, ValueError) as exc:
                if diagnostics_allowed:
                    self.last_error = "historical " + error_detail("owner.disconnect", exc, profile_id=self.entry.profile_id) + "; disconnect metadata was not updated; check host coordination storage."
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(.25)
        self._owner_lock.__exit__(None, None, None)


class ReviewService:
    """Singleton scheduler with a fixed bounded pool, never a catalog writer."""
    def __init__(self, roster, *, provider=None, workers=1, timeout=30, output_tokens=4096, discovery_interval=1):
        if not 1 <= workers <= 8 or not 0 < timeout <= 120 or not 256 <= output_tokens <= 8192:
            raise ValueError("Invalid fixed service limits")
        if not .05 <= discovery_interval <= 60:
            raise ValueError("Discovery interval must be between 0.05 and 60 seconds")
        self.roster, self.workers = roster, workers
        self.timeout, self.output_tokens = timeout, output_tokens
        self.provider = provider
        self.stop_event = threading.Event()
        self.errors = {}
        self.generation = None
        self.discovery_interval = discovery_interval
        self._next_discovery = 0
        self._reported = {}
        self._once = True
        self._phase, self._phase_timing = "service.startup", {}

    def _record(self, key, detail):
        historical = "historical " + detail
        remember_diagnostic(self.errors, key, historical)
        if historical not in self._reported:
            print("Skill review: " + historical, file=sys.stderr, flush=True)
            remember_diagnostic(self._reported, historical, True)

    def _failure(self, stage, entry, exc, job=None):
        detail = error_detail(stage, exc, profile_id=entry.profile_id,
                              job_id=job["job_id"] if job else None)
        stage = exc.stage if isinstance(exc, DiagnosticFailure) else stage
        if stage in ("pending.scan", "pending.recovery", "claim"):
            detail += "; local attempt interrupted; admitted work retained; "
            detail += ("once has no automatic future run; rerun once or start run after checking coordination/storage."
                       if self._once else "automatic future run-loop retry while the service runs; check coordination/storage if repeated.")
        else:
            detail += "; RUNNING attempt interrupted; no active retry; resolve coordination/storage failure and restart the service to recover."
        self._record(diagnostic_id(entry.profile_id), detail)

    def _refresh(self):
        if isinstance(self.roster, DiscoveryRoster) and time.monotonic() >= self._next_discovery:
            self.roster.refresh()
            self._next_discovery = time.monotonic() + self.discovery_interval
            for key, value in self.roster.errors.items():
                self._record("discovery:" + diagnostic_id(key, child=True), value.removeprefix("historical "))

    def stop(self):
        self.stop_event.set()

    def _pending(self, recover=False):
        candidates = []
        for entry in self.roster.profiles:
            try:
                with diagnostic_gate(entry, "pending.scan", BUSY_SECONDS):
                    with diagnostic_stage("pending.scan"):
                        auth = read_auth(self.roster, entry)
                    if not auth or not auth["enabled"] or not Path(entry.mailbox).exists():
                        continue
                    with diagnostic_stage("pending.scan", sqlite_busy_timeout_s=BUSY_SECONDS), closing(connect(entry, readonly=True)) as conn:
                        interrupted = conn.execute("SELECT job_id FROM jobs WHERE status='RUNNING' AND generation=? AND (service_generation IS NULL OR service_generation != ?) LIMIT 1", (auth["generation"], self.generation)).fetchone()
                    if interrupted:
                        with diagnostic_stage("pending.recovery", job_id=interrupted["job_id"], sqlite_busy_timeout_s=BUSY_SECONDS), closing(connect(entry)) as conn, conn:
                            # Only attempts from an exited singleton generation. A live
                            # rescan never resets any current service attempt.
                            conn.execute("UPDATE jobs SET status='PREPARED',service_generation=NULL,result_id=NULL,result_json=NULL WHERE status='RUNNING' AND generation=? AND (service_generation IS NULL OR service_generation != ?)", (auth["generation"], self.generation))
                    with diagnostic_stage("pending.scan", sqlite_busy_timeout_s=BUSY_SECONDS), closing(connect(entry, readonly=True)) as conn:
                        row = conn.execute("SELECT * FROM jobs WHERE status='PREPARED' AND generation=? ORDER BY seq LIMIT 1", (auth["generation"],)).fetchone()
                        if row:
                            with diagnostic_stage("pending.scan", job_id=row["job_id"]):
                                if valid_job(auth, entry, dict(row)):
                                    candidates.append((row["created"], entry.profile_id, entry, dict(row)))
            except (OSError, ValueError, KeyError, sqlite3.Error, TimeoutError, DiagnosticFailure) as exc:
                self._failure("pending.scan", entry, exc)
        return sorted(candidates, key=lambda v: v[:2])

    def _claim(self, entry, job):
        with diagnostic_gate(entry, "claim", job_id=job["job_id"]):
            with diagnostic_stage("claim", job_id=job["job_id"]):
                auth = read_auth(self.roster, entry)
                if self.stop_event.is_set() or not valid_job(auth, entry, job):
                    return False
            with diagnostic_stage("claim", job_id=job["job_id"], sqlite_busy_timeout_s=BUSY_SECONDS), closing(connect(entry)) as conn, conn:
                changed = conn.execute("UPDATE jobs SET status='RUNNING',service_generation=? WHERE job_id=? AND status='PREPARED' AND input_seal=?", (self.generation, job["job_id"], job["input_seal"])).rowcount
            job["service_generation"] = self.generation
            return bool(changed)

    def _infer(self, entry, job, provider):
        # Start authorization is rechecked in the worker, after scheduling.
        with diagnostic_gate(entry, "worker.authorization", job_id=job["job_id"]):
            with diagnostic_stage("worker.authorization", job_id=job["job_id"]):
                auth = read_auth(self.roster, entry)
                if self.stop_event.is_set() or not valid_job(auth, entry, job):
                    return None
        def failure(stage, status, exc):
            detail = error_detail(stage, exc, profile_id=entry.profile_id, job_id=job["job_id"],
                                  use_context=False,
                                  **(dict(provider_timeout_s=self.timeout) if stage == "provider.inference" else {}))
            return dict(status=status, detail=detail + f"; outcome={status}; no automatic provider retry; awaiting result persistence and owner acknowledgement.")
        try:
            proposal = provider(request_json(self.roster, job))
        except ValueError as exc:
            # The SDK adapter also validates output; preserve its INVALID contract.
            return failure("provider.inference_validation", "INVALID", exc)
        except Exception as exc:
            return failure("provider.inference", "FAILED", exc)
        try:
            validate_proposal(proposal)
            return dict(status="PROPOSAL", proposal=proposal)
        except ValueError as exc:
            return failure("provider.validation", "INVALID", exc)
        except Exception as exc:
            return failure("provider.validation", "FAILED", exc)

    def _finish(self, entry, job, result):
        if result is None:
            return
        with diagnostic_gate(entry, "result.persistence", job_id=job["job_id"]):
            with diagnostic_stage("result.persistence", job_id=job["job_id"]):
                auth = read_auth(self.roster, entry)
                if self.stop_event.is_set() or not valid_job(auth, entry, job):
                    return
                if result["status"] in ("FAILED", "INVALID"):
                    self._record(diagnostic_id(entry.profile_id), result["detail"])
                job["result_json"] = packed(result)
                result_id = result_seal(auth, job)
            with diagnostic_stage("result.persistence", job_id=job["job_id"], sqlite_busy_timeout_s=BUSY_SECONDS), closing(connect(entry)) as conn, conn:
                conn.execute("UPDATE jobs SET status='RESULT',result_json=?,result_id=? WHERE job_id=? AND status='RUNNING' AND service_generation=? AND input_seal=?",
                             (job["result_json"], result_id, job["job_id"], self.generation, job["input_seal"]))

    def once(self):
        return self.run(once=True)

    def run(self, once=False):
        # Held through executor shutdown. No replacement while a local worker
        # still owns capacity, even if a remote request outlives its timeout.
        self._once = once
        self._phase, self._phase_timing = "service.startup.singleton", dict(coordination_timeout_s=0)
        with ProcessLock("hermes-skill-review-service", timeout=0):
            self._phase, self._phase_timing = "service.startup.policy", {}
            self.generation = uuid.uuid4().hex
            if isinstance(self.roster, DiscoveryRoster):
                self.roster.check_policy(create=True)
            self._next_discovery = 0
            self._refresh()
            initial = self._pending(recover=True)
            client = None
            if self.provider is None:
                with diagnostic_stage("service.startup.provider"):
                    from openai import OpenAI
                    from skills import AutoSkillExtractor
                    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY") or "local-no-key-required",
                                    base_url=self.roster.base_url, timeout=self.timeout, max_retries=0)
                provider = lambda request: AutoSkillExtractor.generate_proposal(client, self.roster.model, request, timeout=self.timeout, output_tokens=self.output_tokens)
            else:
                provider = self.provider
            processed, inflight = 0, {}
            self._phase = "service.run"
            try:
                with ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="skill-inference") as pool:
                    while not self.stop_event.is_set():
                        self._refresh()
                        pending = []
                        if len(inflight) < self.workers:
                            pending = initial if once else self._pending()
                        for _, _, entry, job in pending:
                            if len(inflight) >= self.workers:
                                break
                            stage = "claim"
                            try:
                                if self._claim(entry, job):
                                    stage = "worker.dispatch"
                                    inflight[pool.submit(self._infer, entry, job, provider)] = (entry, job)
                            except Exception as exc:
                                self._failure(stage, entry, exc, job)
                        if once:
                            claimed_ids = {job["job_id"] for _, job in inflight.values()}
                            initial = [item for item in initial if item[3]["job_id"] not in claimed_ids]
                        if not inflight:
                            if once:
                                break
                            self.stop_event.wait(.1)
                            continue
                        done, _ = wait(inflight, timeout=.05, return_when=FIRST_COMPLETED)
                        for future in done:
                            entry, job = inflight.pop(future)
                            stage = "worker.authorization"
                            try:
                                result = future.result()
                                stage = "result.persistence"
                                self._finish(entry, job, result)
                            except Exception as exc:
                                self._failure(stage, entry, exc, job)
                            processed += 1
            finally:
                if client:
                    with diagnostic_stage("service.shutdown"):
                        client.close()
            return processed


def main(argv=None):
    parser = argparse.ArgumentParser(description="Supervise ONE private-profile skill review service (no OS installation).")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "once", "status"):
        child = sub.add_parser(command)
        mode = child.add_mutually_exclusive_group()
        mode.add_argument("--roster", help="Static host allowlist JSON; excludes automatic discovery flags")
        mode.add_argument("--profiles-dir", help="Host-controlled profile root (default: .agent_profiles beside agent.py)")
        child.add_argument("--model", help="Same host model as the owners (default: AGENT_MODEL or Qwen-32b)")
        child.add_argument("--base-url", help="Same host endpoint as the owners (default: OPENAI_BASE_URL or http://localhost:11434/v1)")
        child.add_argument("--discovery-interval", type=float, help="Immediate-child poll interval, 0.05-60 seconds (default: 1)")
        if command != "status":
            child.add_argument("--workers", type=int, default=1)
            child.add_argument("--timeout", type=float, default=30)
            child.add_argument("--output-tokens", type=int, default=4096)
    entry_parser = sub.add_parser("roster-entry", help="Print a roster entry without creating profile state")
    entry_parser.add_argument("--profile", required=True)
    entry_parser.add_argument("--state-root", required=True)
    args = parser.parse_args(argv)
    stage = "service.status" if args.command == "status" else "service.startup"
    profile_id = None
    service = None
    try:
        if args.command == "roster-entry":
            validate_profile_id(args.profile)
            from skills import SkillStore
            root = Path(args.state_root).resolve()
            print(packed(dict(profile_id=args.profile, mailbox=str(root / "skill_review.db"), store_id=SkillStore(str(root / "skills")).store_id)))
            return 0
        if args.roster and any(value is not None for value in (args.model, args.base_url, args.discovery_interval)):
            raise ValueError("--roster conflicts with --model, --base-url and --discovery-interval; static settings come from the roster")
        roster = load_roster(args.roster) if args.roster else DiscoveryRoster(
            args.profiles_dir or DEFAULT_PROFILES_DIR, args.model, args.base_url)
        if args.command == "status":
            if isinstance(roster, DiscoveryRoster):
                roster.refresh()
            try:
                with ProcessLock("hermes-skill-review-service", timeout=0):
                    running = False
            except TimeoutError:
                running = True
            profiles = []
            for entry in roster.profiles:
                profile_id = entry.profile_id
                with diagnostic_gate(entry, "service.status"):
                    with diagnostic_stage("service.status"):
                        auth = read_auth(roster, entry)
                    counts = {}
                    if Path(entry.mailbox).exists():
                        with diagnostic_stage("service.status", sqlite_busy_timeout_s=BUSY_SECONDS), closing(connect(entry, readonly=True)) as conn:
                            counts = dict(conn.execute("SELECT status,count(*) FROM jobs GROUP BY status"))
                    profiles.append(dict(profile_id=entry.profile_id, enabled=bool(auth and auth["enabled"]), jobs=counts))
            print(packed(dict(service_running=running, profiles=profiles, discovery_errors=getattr(roster, "errors", {}))))
            return 0
        service = ReviewService(roster, workers=args.workers, timeout=args.timeout, output_tokens=args.output_tokens,
                                discovery_interval=args.discovery_interval if args.discovery_interval is not None else 1)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: service.stop())
        count = service.run(once=args.command == "once")
        print(packed(dict(processed=count, errors=service.errors)))
        return 0
    except Exception as exc:
        timing = {}
        if service is not None:
            stage, timing = service._phase, service._phase_timing
        detail = error_detail(stage, exc, profile_id=profile_id, **timing)
        print("Skill review service: " + detail + "; check launch settings, flag conflicts, profile identity/permissions and host coordination. If the singleton is busy, wait for the prior service to exit before restarting. No automatic retry from this command.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
