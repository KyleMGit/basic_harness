"""One supervised inference service; private durable mailboxes; owner-only apply.

See docs/async_skill_review.md for authorization, limits and operator commands.
This module never routes using model output. A service receives only sealed,
prepared JSON; only Owner receives a bound catalog object.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import closing, nullcontext
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


def packed(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class EvidenceResult:
    status: str
    session_id: str
    task_id: str
    messages_json: str = ""


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

    def add(self, message):
        if message.get("role") == "system" or self.overflow:
            return
        if message.get("role") not in ("user", "assistant", "tool"):
            return
        # Bound the traversal BEFORE redaction/serialization. No slicing secrets
        # mid-value; oversized content refuses the whole task explicitly.
        nodes, budget = 0, self.max_bytes

        def bounded(value, depth=0):
            nonlocal nodes, budget
            nodes += 1
            if nodes > 512 or depth > 12 or budget < 0:
                raise ValueError("Evidence input exceeds limit")
            if isinstance(value, str):
                if len(value) > budget:
                    raise ValueError("Evidence string exceeds limit")
                budget -= len(value.encode("utf-8"))
            elif isinstance(value, dict):
                for key, item in value.items():
                    bounded(key, depth + 1)
                    bounded(item, depth + 1)
            elif isinstance(value, list):
                for item in value:
                    bounded(item, depth + 1)
            elif value is not None and not isinstance(value, (bool, int, float)):
                raise ValueError("Unsupported evidence value")

        try:
            selected = {k: message[k] for k in ("role", "content", "tool_calls", "tool_call_id") if k in message}
            bounded(selected)
            safe = TrajectoryLogger._safe(selected)
            size = len(packed(safe).encode()) + 1
            if len(self.messages) >= self.max_messages or size + self.byte_count > self.max_bytes:
                raise ValueError("Evidence capacity exceeded")
            self.messages.append(safe)
            self.byte_count += size
        except (ValueError, TypeError, RecursionError):
            self.overflow = True

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
        return EvidenceResult(status, self.session_id, self.task_id, packed(self.messages) if status == "READY" else "")


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
                            self.errors.pop(child.name, None)
                        except (OSError, ValueError) as exc:
                            self.errors[child.name] = str(exc)
            self.root.validate(required=False)
            self.errors.pop("root", None)
        except (OSError, ValueError) as exc:
            self.errors["root"] = str(exc)
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
            return Admission("DISABLED")
        if evidence.status != "READY":
            return Admission(evidence.status, "Task evidence was not accepted")
        try:
            with gate(self.entry, BUSY_SECONDS):
                auth = read_auth(self.roster, self.entry)
                if not self._authorized(auth):
                    return Admission("DISABLED")
                size = len(evidence.messages_json.encode())
                if size > 16 * 1024:
                    return Admission("OVERFLOW", "Task evidence exceeds 16 KiB")
                with closing(connect(self.entry)) as conn, conn:
                    conn.execute("BEGIN IMMEDIATE")
                    count, used = conn.execute("SELECT count(*), coalesce(sum(bytes),0) FROM evidence").fetchone()
                    if count >= self.max_records or used + size > MAX_QUEUE_BYTES:
                        return Admission("OVERFLOW", "Private evidence queue is full; older tasks retained")
                    conn.execute("INSERT INTO evidence(task_id,session_id,generation,messages_json,bytes,created) VALUES(?,?,?,?,?,?)",
                                 (evidence.task_id, evidence.session_id, auth["generation"], evidence.messages_json, size, time.time()))
            self._wake.set()
            return Admission("ACCEPTED", "Durably committed to the private local queue")
        except (OSError, ValueError, sqlite3.Error, TimeoutError) as exc:
            return Admission("FAILED", type(exc).__name__)

    def _ack(self, conn, job, outcome):
        conn.execute("UPDATE jobs SET status=?,detail=?,evidence_json='',catalog_json='',host_json='',result_json='' WHERE job_id=? AND status='RESULT'",
                     (outcome.status, outcome.detail, job["job_id"]))
        conn.execute("DELETE FROM evidence WHERE job_id=?", (job["job_id"],))
        conn.execute("DELETE FROM jobs WHERE status NOT IN ('PREPARED','RUNNING','RESULT') AND seq NOT IN (SELECT seq FROM jobs ORDER BY seq DESC LIMIT ?)", (HISTORY_COUNT,))

    def pump(self):
        try:
            self._pump()
        except Exception as exc:
            self.last_error = type(exc).__name__

    def _pump(self):
        if not self.enabled or self.closed:
            return
        with gate(self.entry):
            auth = read_auth(self.roster, self.entry)
            if not self._authorized(auth):
                return
            with closing(connect(self.entry)) as conn, conn:
                for row in conn.execute("SELECT * FROM jobs WHERE status='RESULT' ORDER BY seq").fetchall():
                    job = dict(row)
                    if not valid_job(auth, self.entry, job) or not hmac.compare_digest(job["result_id"], result_seal(auth, job)):
                        continue
                    result = json.loads(job["result_json"])
                    from skill_catalog import Publication
                    if result["status"] in ("INVALID", "FAILED"):
                        outcome = Publication(result["status"], result["detail"])
                    else:
                        host = json.loads(job["host_json"])
                        snapshot = CatalogSnapshot(host["store_id"], job["catalog_json"], tuple(ReviewTarget(**t) for t in host["targets"]))
                        outcome = self.store.apply_review(result["proposal"], snapshot, job["result_id"])
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
        snapshot = self.store.prepare_review()
        evidence_json = packed([dict(session_id=r["session_id"], task_id=r["task_id"], messages=json.loads(r["messages_json"])) for r in selected])
        host_json = packed(dict(store_id=snapshot.store_id, targets=[asdict(t) for t in snapshot.targets]))
        with gate(self.entry):
            auth = read_auth(self.roster, self.entry)
            if not self._authorized(auth) or auth["generation"] != generation:
                return
            job = dict(job_id=uuid.uuid4().hex, profile_id=self.entry.profile_id, store_id=self.entry.store_id,
                       generation=generation, evidence_json=evidence_json, catalog_json=snapshot.public_json, host_json=host_json)
            with closing(connect(self.entry)) as conn, conn:
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
            self.closed = True
            self.enabled = False
            try:
                auth = read_auth(self.roster, self.entry)
                if auth and auth.get("owner_id") == self.owner_id:
                    auth["owner_id"] = None  # Disconnect preserves generation/results.
                    write_auth(self.roster, self.entry, auth)
            except (OSError, ValueError) as exc:
                self.last_error = type(exc).__name__
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

    def _refresh(self):
        if isinstance(self.roster, DiscoveryRoster) and time.monotonic() >= self._next_discovery:
            self.roster.refresh()
            self._next_discovery = time.monotonic() + self.discovery_interval
            for key, value in self.roster.errors.items():
                self.errors["discovery:" + key] = value
        for key, value in self.errors.items():
            if self._reported.get(key) != value:
                print(f"Skill review [{key}]: {value}", file=sys.stderr, flush=True)
                self._reported[key] = value

    def stop(self):
        self.stop_event.set()

    def _pending(self, recover=False):
        candidates = []
        for entry in self.roster.profiles:
            try:
                with gate(entry, BUSY_SECONDS):
                    auth = read_auth(self.roster, entry)
                    if not auth or not auth["enabled"] or not Path(entry.mailbox).exists():
                        continue
                    with closing(connect(entry, readonly=True)) as conn:
                        interrupted = conn.execute("SELECT 1 FROM jobs WHERE status='RUNNING' AND generation=? AND (service_generation IS NULL OR service_generation != ?) LIMIT 1", (auth["generation"], self.generation)).fetchone()
                    if interrupted:
                        with closing(connect(entry)) as conn, conn:
                            # Only attempts from an exited singleton generation. A live
                            # rescan never resets any current service attempt.
                            conn.execute("UPDATE jobs SET status='PREPARED',service_generation=NULL,result_id=NULL,result_json=NULL WHERE status='RUNNING' AND generation=? AND (service_generation IS NULL OR service_generation != ?)", (auth["generation"], self.generation))
                    with closing(connect(entry, readonly=True)) as conn:
                        row = conn.execute("SELECT * FROM jobs WHERE status='PREPARED' AND generation=? ORDER BY seq LIMIT 1", (auth["generation"],)).fetchone()
                        if row and valid_job(auth, entry, dict(row)):
                            candidates.append((row["created"], entry.profile_id, entry, dict(row)))
            except (OSError, ValueError, KeyError, sqlite3.Error, TimeoutError) as exc:
                self.errors[entry.profile_id] = type(exc).__name__
        return sorted(candidates, key=lambda v: v[:2])

    def _claim(self, entry, job):
        with gate(entry):
            auth = read_auth(self.roster, entry)
            if self.stop_event.is_set() or not valid_job(auth, entry, job):
                return False
            with closing(connect(entry)) as conn, conn:
                changed = conn.execute("UPDATE jobs SET status='RUNNING',service_generation=? WHERE job_id=? AND status='PREPARED' AND input_seal=?", (self.generation, job["job_id"], job["input_seal"])).rowcount
            job["service_generation"] = self.generation
            return bool(changed)

    def _infer(self, entry, job, provider):
        # Start authorization is rechecked in the worker, after scheduling.
        with gate(entry):
            auth = read_auth(self.roster, entry)
            if self.stop_event.is_set() or not valid_job(auth, entry, job):
                return None
        try:
            proposal = provider(request_json(self.roster, job))
            validate_proposal(proposal)
            return dict(status="PROPOSAL", proposal=proposal)
        except ValueError as exc:
            return dict(status="INVALID", detail=str(exc)[:256])
        except Exception as exc:
            # Do not retain provider errors that may contain private headers/body.
            return dict(status="FAILED", detail=type(exc).__name__)

    def _finish(self, entry, job, result):
        if result is None:
            return
        with gate(entry):
            auth = read_auth(self.roster, entry)
            if self.stop_event.is_set() or not valid_job(auth, entry, job):
                return
            job["result_json"] = packed(result)
            result_id = result_seal(auth, job)
            with closing(connect(entry)) as conn, conn:
                conn.execute("UPDATE jobs SET status='RESULT',result_json=?,result_id=? WHERE job_id=? AND status='RUNNING' AND service_generation=? AND input_seal=?",
                             (job["result_json"], result_id, job["job_id"], self.generation, job["input_seal"]))

    def once(self):
        return self.run(once=True)

    def run(self, once=False):
        # Held through executor shutdown. No replacement while a local worker
        # still owns capacity, even if a remote request outlives its timeout.
        with ProcessLock("hermes-skill-review-service", timeout=0):
            self.generation = uuid.uuid4().hex
            if isinstance(self.roster, DiscoveryRoster):
                self.roster.check_policy(create=True)
            self._next_discovery = 0
            self._refresh()
            initial = self._pending(recover=True)
            client = None
            if self.provider is None:
                from openai import OpenAI
                from skills import AutoSkillExtractor
                client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY") or "local-no-key-required",
                                base_url=self.roster.base_url, timeout=self.timeout, max_retries=0)
                provider = lambda request: AutoSkillExtractor.generate_proposal(client, self.roster.model, request, timeout=self.timeout, output_tokens=self.output_tokens)
            else:
                provider = self.provider
            processed, inflight = 0, {}
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
                            try:
                                if self._claim(entry, job):
                                    inflight[pool.submit(self._infer, entry, job, provider)] = (entry, job)
                            except (OSError, ValueError, sqlite3.Error, TimeoutError) as exc:
                                self.errors[entry.profile_id] = type(exc).__name__
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
                            try:
                                self._finish(entry, job, future.result())
                            except Exception as exc:
                                self.errors[entry.profile_id] = type(exc).__name__
                            processed += 1
            finally:
                if client:
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
                with gate(entry):
                    auth = read_auth(roster, entry)
                    counts = {}
                    if Path(entry.mailbox).exists():
                        with closing(connect(entry, readonly=True)) as conn:
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
    except (OSError, ValueError, TimeoutError, sqlite3.Error) as exc:
        print(f"Skill review service: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
