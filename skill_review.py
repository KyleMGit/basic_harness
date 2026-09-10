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
TURN_BYTES = 256 * 1024
TURN_MESSAGES = 256
RAW_MESSAGE_BYTES = 256 * 1024
STRUCTURE_NODES = 8192
STRUCTURE_DEPTH = 24
EPISODE_BYTES = 512 * 1024
EPISODE_MESSAGES = 512
MAX_QUEUE_BYTES = 8 * 1024 * 1024
QUEUE_HEADROOM_BYTES = 512 * 1024
REPETITION_COUNT = 1024
REPETITION_BYTES = 128 * 1024
BATCH_COUNT = 4
BATCH_BYTES = 64 * 1024
BATCH_MESSAGES = 128
CARRY_BYTES = 16 * 1024
IDLE_SECONDS = 120
ELIGIBLE_AGE_SECONDS = 900
MAILBOX_BYTES = 16 * 1024 * 1024
HISTORY_COUNT = 32
ACTIVE = ("PREPARED", "RUNNING", "RESULT")
MAX_DIAGNOSTICS = 128
REJECTED = "NOT queued; no automatic retry; earlier queue work retained."
DIAGNOSTIC_STAGES = frozenset((
    "owner.authorization", "owner.mailbox", "owner.delivery", "owner.publication", "owner.acknowledgement",
    "owner.preparation", "owner.preparation_authorization", "owner.preparation_persistence",
    "pending.scan", "pending.recovery", "claim", "worker.authorization", "result.persistence",
    "service.preparation", "service.startup.provider", "service.shutdown", "service.status",
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


def substantive_challenge(value):
    """Recognize explicit procedural challenges without treating parameter edits as corrections."""
    text = re.sub(r"\s+", " ", str(value or "")).casefold()
    correction = bool(re.search(
        r"\b(?:wrong|incorrect|invalid|broken|flawed)\b|\b(?:correction|correct)\s*:", text))
    structural = bool(re.search(
        r"\b(?:join|deduplicat\w*|qualify|window|dialect|workaround|algorithm|"
        r"aggregation|grouping|partition|cte|subquer\w*)\b", text))
    procedure = bool(re.search(r"\b(?:procedure|approach|method|strategy|logic)\b", text))
    parameter_only = bool(re.search(
        r"\b(?:date|day|week|month|year|filter|where|limit|sort|format|column|label|range)\b", text))
    return correction and (structural or (procedure and not parameter_only))


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
    message_count: int = 0
    event_json: str = "{}"

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
        elif reason in ("nodes", "depth"):
            detail += " (partial traversal; remaining data was not measured)"
        return detail


@dataclass(frozen=True)
class Admission:
    status: str
    detail: str = ""


class BudgetRefusal(ValueError):
    def __init__(self, reason, observed, limit):
        super().__init__("Review candidate exceeds a terminal budget")
        self.reason, self.observed, self.limit = reason, observed, limit

    def detail(self):
        unit = "messages" if self.reason == "selected_messages" else "bytes"
        return (f"stage=service.preparation reason={self.reason} observed_{unit}={self.observed} "
                f"limit_{unit}={self.limit}; outcome=BUDGET_REFUSED; zero provider requests; "
                "accepted source references await owner acknowledgement.")


class MailboxCapacityError(ValueError):
    pass


class Evidence:
    """Incremental per-task capture; refusal rather than silently losing old data."""
    SQL_TOOLS = frozenset(("query_teradata", "query_impala"))
    VERIFICATION_TOOLS = frozenset(("read_file", "run_terminal_command"))
    _XML_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.I | re.S)
    _XML_RESULT = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.I | re.S)

    def __init__(self, session_id, task_id=None, max_bytes=TURN_BYTES, max_messages=TURN_MESSAGES):
        self.session_id = str(session_id)[:128]
        self.task_id = task_id or uuid.uuid4().hex
        self.max_bytes, self.max_messages = max_bytes, max_messages
        self.messages = []
        self.byte_count = 2
        self.overflow = False
        self.refusal = ("", None, None, False)
        self._calls = {}
        self._open_calls = set()
        self._events = []
        self._user_text = []
        self._saw_business_result = False

    @staticmethod
    def _metadata_sql(sql):
        # Only executable source tokens participate. Comments and string
        # literals cannot promote business output to trusted metadata, and a
        # mixed metadata/business query is always treated as business output.
        value = re.sub(r"--[^\r\n]*|/\*.*?\*/", " ", str(sql or ""), flags=re.S)
        value = re.sub(r"'(?:''|[^'])*'", " ", value)
        value = re.sub(r"\s+", " ", value.strip()).casefold()
        if re.match(r"^(?:show|describe|desc|pragma)\s+", value):
            return True
        if re.match(r"^help\s+(?:column|table)\b", value):
            return True
        source_clauses = re.findall(
            r"\bfrom\b(.*?)(?=\b(?:where|qualify|group\s+by|order\s+by|limit|fetch|union|except|intersect)\b|;|$)",
            value)
        if any("," in clause for clause in source_clauses):
            return False
        source_slots = re.findall(r"\b(?:from|join)\s+([^\s,;]+)", value)
        if not source_slots or any(source.startswith("(") for source in source_slots):
            return False

        def normalized(source):
            return re.sub(r'[`"\[\]]', "", source).rstrip(",")

        trusted = re.compile(
            r"^(?:information_schema\.(?:columns|tables|views|schemata)|"
            r"dbc\.(?:columns|columnsv|tables|tablesv|indices|databases|databasesv)|"
            r"sys\.(?:columns|tables|schemas|views))$")
        return all(trusted.fullmatch(normalized(source)) for source in source_slots)

    @staticmethod
    def _sql_shape(sql):
        value = re.sub(r"--[^\r\n]*|/\*.*?\*/", " ", str(sql or ""), flags=re.S).casefold()
        value = re.sub(r"'(?:''|[^'])*'|\b\d+(?:\.\d+)?\b", "?", value)
        value = re.sub(r"\bwhere\b.*?(?=\b(?:qualify|order\s+by|limit|fetch|$))", " ", value, flags=re.S)
        value = re.sub(r"\bgroup\s+by\b.*?(?=\b(?:qualify|order\s+by|limit|fetch|$))", " ", value, flags=re.S)
        value = re.sub(r"\border\s+by\b.*?(?=\b(?:limit|fetch|$))", " ", value, flags=re.S)
        value = re.sub(r"\b(?:limit\s+\?|fetch\s+(?:first|next)\s+\?\s+rows\s+only)\b", " ", value)
        value = re.sub(r"\s+", " ", value).strip().rstrip(";")
        return hashlib.sha256(value.encode()).hexdigest()

    @staticmethod
    def _resources(sql):
        names = re.findall(r"\b(?:from|join|update|into|merge\s+into)\s+([A-Za-z_][\w.$-]*)", str(sql or ""), re.I)
        return sorted({hashlib.sha256(name.casefold().encode()).hexdigest()[:24] for name in names})[:32]

    @staticmethod
    def _nonroutine(sql):
        return bool(re.search(r"\b(join|with|qualify|over\s*\(|cast\s*\(|merge|recursive|pivot|unpivot|lateral|regexp|json[_a-z]*\s*\()", str(sql or ""), re.I))

    def _remember_call(self, call_id, name, arguments):
        if not isinstance(call_id, str) or not call_id or not isinstance(name, str):
            return
        args = arguments
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (TypeError, ValueError):
                args = {}
        args = args if isinstance(args, dict) else {}
        info = {"call_id": call_id[:128], "tool": name[:128]}
        if name in self.SQL_TOOLS:
            sql = args.get("sql") if isinstance(args.get("sql"), str) else ""
            info.update(sql=True, backend=name, metadata=self._metadata_sql(sql),
                        signature=self._sql_shape(sql), resources=self._resources(sql),
                        nonroutine=self._nonroutine(sql))
        elif name in self.VERIFICATION_TOOLS:
            info.update(verification=True, nonroutine=True)
        self._calls[call_id] = info
        self._open_calls.add(call_id)

    def _result_projection(self, call_id, content):
        info = self._calls.get(call_id, {})
        if not info.get("sql"):
            if info.get("verification"):
                failure = isinstance(content, str) and bool(re.search(r"\b(failed|error|exception|denied|timeout|not found)\b", content, re.I))
                self._events.append(dict(info, outcome="failure" if failure else "verification_success"))
            self._open_calls.discard(call_id)
            return content
        parsed = None
        try:
            parsed = json.loads(content)
        except (TypeError, ValueError):
            pass
        envelope_keys = {"database", "columns", "rows", "row_count", "truncated"}
        envelope = (isinstance(parsed, dict) and set(parsed) == envelope_keys
                    and isinstance(parsed.get("database"), str)
                    and isinstance(parsed.get("columns"), list) and isinstance(parsed.get("rows"), list)
                    and type(parsed.get("row_count")) is int and isinstance(parsed.get("truncated"), bool))
        event = dict(info)
        if envelope:
            event.update(outcome="metadata_success" if info.get("metadata") else "business_success")
            if info.get("metadata"):
                projected = content
            else:
                columns, column_bytes, columns_truncated = [], 2, False
                for value in parsed["columns"][:256]:
                    if not isinstance(value, str):
                        columns_truncated = True
                        continue
                    value = value[:256]
                    added = len(json.dumps(value, ensure_ascii=False).encode("utf-8")) + 1
                    if column_bytes + added > 4096:
                        columns_truncated = True
                        break
                    columns.append(value)
                    column_bytes += added
                columns_truncated |= len(parsed["columns"]) > len(columns)
                receipt = dict(business_result_omitted=True, call_id=call_id,
                               database=parsed["database"][:256], columns=columns,
                               row_count=parsed["row_count"], truncated=parsed["truncated"])
                if columns_truncated:
                    receipt["columns_truncated"] = True
                projected = packed(receipt)
                self._saw_business_result = True
        else:
            failure = isinstance(content, str) and bool(re.search(r"\b(failed|error|exception|denied|timeout)\b", content, re.I))
            event.update(outcome="failure" if failure else "unknown")
            if failure:
                projected = content
            else:
                projected = packed(dict(unclassified_sql_result_omitted=True, call_id=call_id))
        self._events.append(event)
        self._open_calls.discard(call_id)
        return projected

    def _project_xml(self, content):
        call_index = 0

        def call(match):
            nonlocal call_index
            try:
                payload = json.loads(match.group(1))
                arguments = payload.get("arguments", {})
                call_id = payload.get("tool_call_id") or f"hermes_call_{call_index}"
                call_index += 1
                self._remember_call(call_id,
                                    payload.get("name"), arguments)
            except (TypeError, ValueError):
                pass
            return match.group(0)

        content = self._XML_CALL.sub(call, content)

        def result(match):
            try:
                payload = json.loads(match.group(1))
                call_id = payload.get("tool_call_id")
                name = payload.get("name")
                if call_id not in self._calls and name:
                    self._remember_call(call_id, name, {})
                payload["content"] = self._result_projection(call_id, payload.get("content"))
                return "<tool_response>\n" + packed(payload) + "\n</tool_response>"
            except (TypeError, ValueError):
                return match.group(0)
        return self._XML_RESULT.sub(result, content)

    def _project(self, selected):
        role = selected.get("role")
        content = selected.get("content")
        if role == "assistant":
            for call in selected.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                function = call.get("function") if isinstance(call.get("function"), dict) else {}
                self._remember_call(call.get("id"), function.get("name"), function.get("arguments"))
            if isinstance(content, str):
                content = self._project_xml(content)
                copied = False
                try:
                    decoded = json.loads(content)
                    copied = self._saw_business_result and isinstance(decoded, dict) and isinstance(decoded.get("rows"), list)
                except (TypeError, ValueError):
                    copied = self._saw_business_result and bool(re.search(r"```csv\b", content, re.I))
                if copied:
                    content = packed({"assistant_business_result_copy_omitted": True})
                selected["content"] = content
        elif role == "tool":
            selected["content"] = self._result_projection(selected.get("tool_call_id"), content)
        elif role == "user" and isinstance(content, str):
            projected = self._project_xml(content)
            selected["content"] = projected
            if "<tool_response" not in content.casefold():
                self._user_text.append(content[:8192])
        return selected

    def _event_summary(self):
        text = "\n".join(self._user_text).casefold()
        correction = substantive_challenge(text)
        return packed(dict(version=1, correction=correction, events=self._events[:64]))

    def add(self, message):
        if message.get("role") == "system" or self.overflow:
            return
        if message.get("role") not in ("user", "assistant", "tool"):
            return
        nodes = 0

        def refuse(reason, observed=None, limit=None, partial=False):
            self.refusal = reason, observed, limit, partial
            raise ValueError("Evidence refused")

        def bounded(value, depth=0):
            nonlocal nodes
            nodes += 1
            if nodes > STRUCTURE_NODES:
                refuse("nodes", nodes, STRUCTURE_NODES)
            if depth > STRUCTURE_DEPTH:
                refuse("depth", depth, STRUCTURE_DEPTH)
            if isinstance(value, str):
                stripped = value.strip()
                if stripped[:1] in ("{", "["):
                    try:
                        decoded = json.loads(value)
                    except (TypeError, ValueError):
                        decoded = None
                    if isinstance(decoded, (dict, list)):
                        bounded(decoded, depth + 1)
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
            raw_size = len(json.dumps(selected, ensure_ascii=False, sort_keys=True,
                                      separators=(",", ":")).encode("utf-8"))
            raw_limit = min(RAW_MESSAGE_BYTES, self.max_bytes)
            if raw_size > raw_limit:
                refuse("raw_bytes", raw_size, raw_limit)
            # No JSON string, native argument payload, XML call, or SQL result
            # is decoded before the complete selected message passes this raw
            # inspection bound. Safely inspectable SQL rows are projected next.
            selected = self._project(selected)
            bounded(selected)
            safe = TrajectoryLogger._safe(selected)
            if len(self.messages) >= self.max_messages:
                refuse("message_count", len(self.messages) + 1, self.max_messages)
            candidate = self.messages + [safe]
            size = len(packed(candidate).encode())
            if size > self.max_bytes:
                refuse("serialized_bytes", size, self.max_bytes)
            self.messages.append(safe)
            self.byte_count = size
        except (ValueError, TypeError, RecursionError):
            self.overflow = True
            if not self.refusal[0]:
                self.refusal = ("unsupported_data", None, None, False)

    def finish(self, completed=True, finish_reason="stop"):
        status = "READY"
        unanswered = set(self._open_calls)
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
                              packed(self.messages) if status == "READY" else "", *refusal,
                              len(self.messages), self._event_summary())


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
    page_size = conn.execute("PRAGMA page_size").fetchone()[0]
    page_count = conn.execute("PRAGMA page_count").fetchone()[0]
    page_limit = MAILBOX_BYTES // page_size
    if page_count > page_limit:
        conn.close()
        raise MailboxCapacityError("Existing review mailbox exceeds the configured page ceiling")
    if not readonly:
        conn.execute(f"PRAGMA max_page_count={page_limit}")
        conn.execute(f"PRAGMA journal_size_limit={MAILBOX_BYTES}")
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
                CREATE TABLE IF NOT EXISTS episodes (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, episode_id TEXT UNIQUE, session_id TEXT,
                    generation TEXT, revision INTEGER, state TEXT, eligible INTEGER, correction INTEGER,
                    created REAL, updated REAL, eligible_at REAL, ready_at REAL, resources_json TEXT,
                    signatures_json TEXT, last_job_id TEXT, anchor_json TEXT NOT NULL DEFAULT '',
                    anchor_bytes INTEGER NOT NULL DEFAULT 0, anchor_messages INTEGER NOT NULL DEFAULT 0,
                    anchor_task_id TEXT, related_skills_json TEXT NOT NULL DEFAULT '[]');
                CREATE TABLE IF NOT EXISTS episode_sources (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, source_id TEXT UNIQUE, episode_id TEXT,
                    revision INTEGER, generation TEXT, task_id TEXT, messages_json TEXT, bytes INTEGER,
                    message_count INTEGER, event_json TEXT, source_hash TEXT, created REAL, job_id TEXT);
                CREATE TABLE IF NOT EXISTS review_fingerprints (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, fingerprint TEXT UNIQUE, bytes INTEGER,
                    created REAL);
            """)
            episode_columns = {row[1] for row in conn.execute("PRAGMA table_info(episodes)")}
            additions = {
                "anchor_json": "TEXT NOT NULL DEFAULT ''",
                "anchor_bytes": "INTEGER NOT NULL DEFAULT 0",
                "anchor_messages": "INTEGER NOT NULL DEFAULT 0",
                "anchor_task_id": "TEXT",
                "related_skills_json": "TEXT NOT NULL DEFAULT '[]'",
                "retired_sources": "INTEGER NOT NULL DEFAULT 0",
                "retired_bytes": "INTEGER NOT NULL DEFAULT 0",
                "retired_messages": "INTEGER NOT NULL DEFAULT 0",
            }
            for name, declaration in additions.items():
                if name not in episode_columns:
                    conn.execute(f"ALTER TABLE episodes ADD COLUMN {name} {declaration}")
            page_size = conn.execute("PRAGMA page_size").fetchone()[0]
            page_count = conn.execute("PRAGMA page_count").fetchone()[0]
            page_limit = MAILBOX_BYTES // page_size
            if page_count > page_limit:
                raise MailboxCapacityError("Existing review mailbox exceeds the configured page ceiling")
            conn.execute(f"PRAGMA max_page_count={page_limit}")
            conn.execute("PRAGMA journal_mode=DELETE")
            # Reenable never revives an invalidated generation. Purge only here,
            # while authorized, never during/after the disabled acknowledgement.
            conn.execute("DELETE FROM evidence WHERE generation != ?", (auth["generation"],))
            conn.execute("DELETE FROM jobs WHERE generation != ?", (auth["generation"],))
            conn.execute("DELETE FROM episode_sources WHERE generation != ?", (auth["generation"],))
            conn.execute("DELETE FROM episodes WHERE generation != ?", (auth["generation"],))

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
                if size > TURN_BYTES:
                    return Admission("OVERFLOW", f"stage={stage} reason=serialized_bytes observed_bytes={size} limit_bytes={TURN_BYTES}; " + REJECTED)
                stage, timing = "admission.sqlite", dict(sqlite_busy_timeout_s=BUSY_SECONDS)
                with closing(connect(self.entry)) as conn, conn:
                    conn.execute("BEGIN IMMEDIATE")
                    count, used = conn.execute("SELECT count(*), coalesce(sum(bytes),0) FROM evidence").fetchone()
                    count += conn.execute("SELECT count(*) FROM episodes WHERE state!='ACKED'").fetchone()[0]
                    used = self._payload_usage(conn)
                    if count >= self.max_records:
                        return Admission("OVERFLOW", f"stage=admission.queue reason=queue_records observed_records={count + 1} limit_records={self.max_records} (including rejected task); " + REJECTED)
                    payload_limit = MAX_QUEUE_BYTES - QUEUE_HEADROOM_BYTES
                    if used + size > payload_limit:
                        return Admission("OVERFLOW", f"stage=admission.queue reason=queue_bytes observed_bytes={used + size} limit_bytes={payload_limit} queued_bytes={used} incoming_bytes={size}; " + REJECTED)
                    conn.execute("INSERT INTO evidence(task_id,session_id,generation,messages_json,bytes,created) VALUES(?,?,?,?,?,?)",
                                 (evidence.task_id, evidence.session_id, auth["generation"], evidence.messages_json, size, time.time()))
            self._wake.set()
            return Admission("ACCEPTED", "stage=admission.commit; durably committed to the private local queue; admission only, publication pending review.")
        except (OSError, ValueError, sqlite3.Error, TimeoutError) as exc:
            return Admission("FAILED", error_detail(stage, exc, **timing) + "; " + REJECTED)

    @staticmethod
    def _event_data(value):
        try:
            data = json.loads(value)
        except (TypeError, ValueError):
            return {"version": 1, "correction": False, "events": []}
        if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("events"), list):
            return {"version": 1, "correction": False, "events": []}
        return data

    @classmethod
    def _episode_signal(cls, event_values):
        failures, metadata, successes = [], [], []
        correction_pending = False
        eligible = correction_verified = False
        for raw in event_values:
            data = cls._event_data(raw)
            if data.get("correction") is True:
                # A substantive challenge revokes the prior conclusion until a
                # later supporting execution verifies the correction.
                correction_pending = True
                eligible = correction_verified = False
            for event in data["events"][:64]:
                if not isinstance(event, dict):
                    continue
                outcome = event.get("outcome")
                if outcome == "verification_success" and correction_pending and event.get("nonroutine"):
                    eligible = correction_verified = True
                    correction_pending = False
                    continue
                if event.get("tool") not in Evidence.SQL_TOOLS:
                    continue
                if outcome == "failure":
                    failures.append(event)
                elif outcome == "metadata_success":
                    metadata.append(event)
                elif outcome == "business_success":
                    successes.append(event)
                    related_failure = any(
                        prior.get("backend") == event.get("backend")
                        and prior.get("signature") != event.get("signature")
                        and (not prior.get("resources") or not event.get("resources")
                             or set(prior.get("resources", ())) & set(event.get("resources", ())))
                        for prior in failures
                    )
                    investigated = (bool(event.get("nonroutine"))
                                    and any(prior.get("backend") == event.get("backend") for prior in metadata))
                    corrected = correction_pending and bool(event.get("nonroutine"))
                    if bool(event.get("nonroutine")) and related_failure or investigated or corrected:
                        eligible = True
                    if corrected:
                        correction_verified = True
                        correction_pending = False
        if correction_pending:
            eligible = correction_verified = False
        return eligible, correction_pending, correction_verified

    @staticmethod
    def _event_resources(event_json):
        data = Owner._event_data(event_json)
        resources, signatures = set(), set()
        for event in data["events"][:64]:
            if isinstance(event, dict):
                resources.update(v for v in event.get("resources", ()) if isinstance(v, str))
                if isinstance(event.get("signature"), str):
                    signatures.add(event["signature"])
        return resources, signatures

    @classmethod
    def _source_has_signal(cls, event_json):
        """Whether a complete turn can contribute to a later eligibility proof."""
        data = cls._event_data(event_json)
        if data.get("correction") is True:
            return True
        return any(
            isinstance(event, dict) and (
                event.get("outcome") in ("failure", "metadata_success")
                or (event.get("outcome") in ("business_success", "verification_success")
                    and bool(event.get("nonroutine")))
            )
            for event in data["events"][:64]
        )

    @classmethod
    def _retire_optional_draft_sources(cls, conn, episode_id, incoming_bytes, incoming_messages):
        """Retire oldest optional whole turns until the incoming turn can fit."""
        rows = conn.execute(
            "SELECT * FROM episode_sources WHERE episode_id=? ORDER BY seq", (episode_id,)).fetchall()
        total_bytes = sum(row["bytes"] for row in rows)
        total_messages = sum(row["message_count"] for row in rows)
        retired = []
        first_seq = rows[0]["seq"] if rows else None
        for row in rows:
            if (total_bytes + incoming_bytes <= EPISODE_BYTES
                    and total_messages + incoming_messages <= EPISODE_MESSAGES):
                break
            if row["seq"] == first_seq or row["job_id"] is not None or cls._source_has_signal(row["event_json"]):
                continue
            conn.execute("DELETE FROM episode_sources WHERE seq=? AND job_id IS NULL", (row["seq"],))
            total_bytes -= row["bytes"]
            total_messages -= row["message_count"]
            retired.append(row)
        if retired:
            conn.execute(
                "UPDATE episodes SET retired_sources=retired_sources+?,retired_bytes=retired_bytes+?,"
                "retired_messages=retired_messages+? WHERE episode_id=?",
                (len(retired), sum(row["bytes"] for row in retired),
                 sum(row["message_count"] for row in retired), episode_id),
            )
        return total_bytes, total_messages

    @classmethod
    def _remaining_episode_state(cls, conn, episode):
        """Promote deferred unbound sources after an acknowledged frozen revision."""
        remaining = conn.execute(
            "SELECT revision,event_json FROM episode_sources WHERE episode_id=? AND job_id IS NULL ORDER BY seq",
            (episode["episode_id"],),
        ).fetchall()
        if not remaining:
            conn.execute("UPDATE episodes SET state='ACKED',last_job_id=NULL WHERE episode_id=?",
                         (episode["episode_id"],))
            return
        event_values = [row["event_json"] for row in remaining]
        eligible, correction, correction_verified = cls._episode_signal(event_values)
        state = ("READY" if correction_verified else "ELIGIBLE" if eligible
                 else "CHALLENGED" if correction else "DRAFT")
        resources, signatures = set(), set()
        for raw in event_values:
            found_resources, found_signatures = cls._event_resources(raw)
            resources.update(found_resources)
            signatures.update(found_signatures)
        now = time.time()
        conn.execute(
            "UPDATE episodes SET revision=?,state=?,eligible=?,correction=?,updated=?,eligible_at=?,"
            "ready_at=?,resources_json=?,signatures_json=?,last_job_id=NULL WHERE episode_id=?",
            (max(row["revision"] for row in remaining), state, int(eligible), int(correction), now,
             now if eligible else None, now if state == "READY" else None,
             packed(sorted(resources)), packed(sorted(signatures)), episode["episode_id"]),
        )

    @staticmethod
    def _record_fingerprints(conn, source_rows):
        signatures = set()
        for row in source_rows:
            _, found = Owner._event_resources(row["event_json"])
            signatures.update(found)
        for fingerprint in sorted(signatures)[:REPETITION_COUNT]:
            conn.execute("INSERT OR IGNORE INTO review_fingerprints(fingerprint,bytes,created) VALUES(?,?,?)",
                         (fingerprint, len(fingerprint.encode()), time.time()))
        while conn.execute("SELECT count(*) FROM review_fingerprints").fetchone()[0] > REPETITION_COUNT:
            conn.execute("DELETE FROM review_fingerprints WHERE seq=(SELECT min(seq) FROM review_fingerprints)")
        while conn.execute("SELECT coalesce(sum(bytes),0) FROM review_fingerprints").fetchone()[0] > REPETITION_BYTES:
            conn.execute("DELETE FROM review_fingerprints WHERE seq=(SELECT min(seq) FROM review_fingerprints)")

    @staticmethod
    def _payload_usage(conn):
        legacy = conn.execute("SELECT coalesce(sum(bytes),0) FROM evidence").fetchone()[0]
        sources = conn.execute("SELECT coalesce(sum(bytes + length(CAST(event_json AS BLOB)) + length(CAST(source_hash AS BLOB)) + 256),0) FROM episode_sources").fetchone()[0]
        episodes = conn.execute("SELECT coalesce(sum(length(CAST(episode_id AS BLOB))+length(CAST(session_id AS BLOB))+length(CAST(generation AS BLOB))+length(CAST(resources_json AS BLOB))+length(CAST(signatures_json AS BLOB))+length(CAST(anchor_json AS BLOB))+length(CAST(related_skills_json AS BLOB))+256),0) FROM episodes WHERE state!='ACKED' OR anchor_json!=''").fetchone()[0]
        jobs = conn.execute("SELECT coalesce(sum(length(CAST(evidence_json AS BLOB))+length(CAST(catalog_json AS BLOB))+length(CAST(host_json AS BLOB))+length(CAST(result_json AS BLOB))+512),0) FROM jobs WHERE status IN ('PREPARED','RUNNING','RESULT')").fetchone()[0]
        fingerprints = conn.execute("SELECT coalesce(sum(bytes),0) FROM review_fingerprints").fetchone()[0]
        return legacy + sources + episodes + jobs + fingerprints

    def begin_turn(self, session_id, user_text):
        """Durably invalidate a pending revision at a recognized caller-start boundary."""
        if self.closed or not self.enabled:
            return Admission("DISABLED", "stage=episode.challenge; " + REJECTED)
        if not substantive_challenge(user_text):
            return Admission("SKIPPED", "")
        now = time.time()
        try:
            with gate(self.entry, BUSY_SECONDS):
                auth = read_auth(self.roster, self.entry)
                if not self._authorized(auth):
                    return Admission("DISABLED", "stage=episode.challenge; " + REJECTED)
                with closing(connect(self.entry)) as conn, conn:
                    conn.execute("BEGIN IMMEDIATE")
                    current = conn.execute(
                        "SELECT * FROM episodes WHERE session_id=? AND generation=? ORDER BY seq DESC LIMIT 1",
                        (str(session_id)[:128], auth["generation"])).fetchone()
                    if not current:
                        return Admission("SKIPPED", "")
                    if current["last_job_id"]:
                        conn.execute(
                            "UPDATE jobs SET status='SUPERSEDED',detail='stage=owner.freshness reason=challenge_started',evidence_json='',catalog_json='',host_json='',result_json='' WHERE job_id=? AND status IN ('PREPARED','RUNNING','RESULT')",
                            (current["last_job_id"],))
                        conn.execute("UPDATE episode_sources SET job_id=NULL WHERE episode_id=? AND job_id=?",
                                     (current["episode_id"], current["last_job_id"]))
                    conn.execute(
                        "UPDATE episodes SET revision=revision+1,state='CHALLENGED',eligible=0,correction=1,updated=?,ready_at=NULL,last_job_id=NULL WHERE episode_id=?",
                        (now, current["episode_id"]))
            self._wake.set()
            return Admission("CHALLENGED", "")
        except (OSError, ValueError, sqlite3.Error, TimeoutError) as exc:
            return Admission("FAILED", error_detail("episode.challenge", exc) + "; " + REJECTED)

    def capture_turn(self, evidence):
        """Durably append a completed delta; deterministic signals gate dispatch."""
        if self.closed or not self.enabled:
            return Admission("DISABLED", "stage=episode.authorization; " + REJECTED)
        if evidence.status != "READY":
            status = evidence.status if evidence.status in ("OVERFLOW", "INCOMPLETE") else "FAILED"
            return Admission(status, evidence.diagnostic() + "; " + REJECTED)
        size = len(evidence.messages_json.encode())
        if size > TURN_BYTES or evidence.message_count > TURN_MESSAGES:
            unit, observed, limit = (("messages", evidence.message_count, TURN_MESSAGES)
                                     if evidence.message_count > TURN_MESSAGES else ("bytes", size, TURN_BYTES))
            return Admission("OVERFLOW", f"stage=episode.turn reason=turn_{unit} observed_{unit}={observed} limit_{unit}={limit}; " + REJECTED)
        now = time.time()
        eligible = False
        deferred = False
        try:
            with gate(self.entry, BUSY_SECONDS):
                auth = read_auth(self.roster, self.entry)
                if not self._authorized(auth):
                    return Admission("DISABLED", "stage=episode.authorization; " + REJECTED)
                with closing(connect(self.entry)) as conn, conn:
                    conn.execute("BEGIN IMMEDIATE")
                    conn.execute("UPDATE episodes SET state='READY',ready_at=? WHERE generation=? AND session_id!=? AND state='ELIGIBLE'",
                                 (now, auth["generation"], evidence.session_id))
                    current = conn.execute("SELECT * FROM episodes WHERE session_id=? AND generation=? ORDER BY seq DESC LIMIT 1",
                                           (evidence.session_id, auth["generation"])).fetchone()
                    resources, signatures = self._event_resources(evidence.event_json)
                    if current:
                        old_resources = set(json.loads(current["resources_json"] or "[]"))
                        clear_change = bool(resources and old_resources and not resources.intersection(old_resources))
                        if clear_change:
                            if current["eligible"] and current["state"] == "ELIGIBLE":
                                conn.execute("UPDATE episodes SET state='READY',ready_at=? WHERE episode_id=?", (now, current["episode_id"]))
                            current = None
                    if current and current["state"] == "PREPARED":
                        # A frozen revision remains valid while it is owned by a
                        # job. Optional routine turns are not independent work;
                        # potentially substantive complete turns wait unbound
                        # for acknowledgement of the frozen revision.
                        if not self._source_has_signal(evidence.event_json):
                            return Admission("SKIPPED", "")
                        deferred = True
                    if current and current["state"] == "ACKED":
                        data = self._event_data(evidence.event_json)
                        new_signal = data.get("correction") is True
                        new_signature = bool(signatures - set(json.loads(current["signatures_json"] or "[]")))
                        if not new_signal and not new_signature:
                            return Admission("SKIPPED", "")
                    was_challenged = bool(current and current["state"] == "CHALLENGED")
                    if not current:
                        units = conn.execute("SELECT count(*) FROM episodes WHERE state!='ACKED'").fetchone()[0]
                        units += conn.execute("SELECT count(*) FROM evidence").fetchone()[0]
                        if units >= self.max_records:
                            return Admission("OVERFLOW", f"stage=episode.queue reason=queue_records observed_records={units + 1} limit_records={self.max_records}; " + REJECTED)
                        episode_id = uuid.uuid4().hex
                        revision = 1
                        conn.execute("INSERT INTO episodes(episode_id,session_id,generation,revision,state,eligible,correction,created,updated,eligible_at,ready_at,resources_json,signatures_json) VALUES(?,?,?,?,'DRAFT',0,0,?,?,?,?,?,?)",
                                     (episode_id, evidence.session_id, auth["generation"], revision, now, now,
                                      None, None, packed(sorted(resources)), packed(sorted(signatures))))
                    else:
                        episode_id = current["episode_id"]
                        if deferred:
                            latest = conn.execute(
                                "SELECT coalesce(max(revision),?) FROM episode_sources WHERE episode_id=?",
                                (current["revision"], episode_id),
                            ).fetchone()[0]
                            revision = max(current["revision"], latest) + 1
                        else:
                            revision = current["revision"] + 1
                            resources.update(json.loads(current["resources_json"] or "[]"))
                            signatures.update(json.loads(current["signatures_json"] or "[]"))
                        if (not deferred and self._event_data(evidence.event_json).get("correction") is True
                                and current["last_job_id"]):
                            conn.execute("UPDATE jobs SET status='SUPERSEDED',detail='stage=owner.freshness reason=correction_revision',evidence_json='',catalog_json='',host_json='',result_json='' WHERE job_id=? AND status IN ('PREPARED','RUNNING','RESULT')",
                                         (current["last_job_id"],))
                            conn.execute("UPDATE episode_sources SET job_id=NULL WHERE episode_id=? AND job_id=?",
                                         (episode_id, current["last_job_id"]))
                    episode_bytes, episode_messages = conn.execute(
                        "SELECT coalesce(sum(bytes),0),coalesce(sum(message_count),0) FROM episode_sources WHERE episode_id=?",
                        (episode_id,)).fetchone()
                    if current:
                        episode_bytes += current["anchor_bytes"] or 0
                        episode_messages += current["anchor_messages"] or 0
                    if (current and current["state"] == "DRAFT"
                            and (episode_bytes + size > EPISODE_BYTES
                                 or episode_messages + evidence.message_count > EPISODE_MESSAGES)):
                        episode_bytes, episode_messages = self._retire_optional_draft_sources(
                            conn, episode_id, size, evidence.message_count)
                        episode_bytes += current["anchor_bytes"] or 0
                        episode_messages += current["anchor_messages"] or 0
                    if episode_bytes + size > EPISODE_BYTES or episode_messages + evidence.message_count > EPISODE_MESSAGES:
                        unit, observed, limit = (("messages", episode_messages + evidence.message_count, EPISODE_MESSAGES)
                                                 if episode_messages + evidence.message_count > EPISODE_MESSAGES
                                                 else ("bytes", episode_bytes + size, EPISODE_BYTES))
                        return Admission("OVERFLOW", f"stage=episode.capacity reason=episode_{unit} observed_{unit}={observed} limit_{unit}={limit}; " + REJECTED)
                    used = self._payload_usage(conn)
                    incoming = size + len(evidence.event_json.encode())
                    payload_limit = MAX_QUEUE_BYTES - QUEUE_HEADROOM_BYTES
                    if used + incoming > payload_limit:
                        return Admission("OVERFLOW", f"stage=episode.queue reason=queue_bytes observed_bytes={used + incoming} limit_bytes={payload_limit} queued_bytes={used} incoming_bytes={incoming}; " + REJECTED)
                    source_id = uuid.uuid4().hex
                    source_hash = hashlib.sha256((evidence.messages_json + "\n" + evidence.event_json).encode()).hexdigest()
                    conn.execute("INSERT INTO episode_sources(source_id,episode_id,revision,generation,task_id,messages_json,bytes,message_count,event_json,source_hash,created) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                                 (source_id, episode_id, revision, auth["generation"], evidence.task_id,
                                  evidence.messages_json, size, evidence.message_count, evidence.event_json, source_hash, now))
                    if not deferred:
                        event_values = [row[0] for row in conn.execute(
                            "SELECT event_json FROM episode_sources WHERE episode_id=? ORDER BY seq", (episode_id,))]
                        if was_challenged:
                            event_values.insert(max(0, len(event_values) - 1), packed(dict(version=1, correction=True, events=[])))
                        eligible, correction, correction_verified = self._episode_signal(event_values)
                        if eligible and not correction_verified and signatures:
                            placeholders = ",".join("?" for _ in signatures)
                            known = conn.execute(
                                f"SELECT count(*) FROM review_fingerprints WHERE fingerprint IN ({placeholders})",
                                tuple(signatures),
                            ).fetchone()[0]
                            if known == len(signatures):
                                eligible = False
                        state = ("READY" if correction_verified else "ELIGIBLE" if eligible
                                 else "CHALLENGED" if correction else "DRAFT")
                        eligible_at = now if eligible else None
                        if current and current["eligible_at"] is not None:
                            eligible_at = current["eligible_at"]
                        conn.execute("UPDATE episodes SET revision=?,state=?,eligible=?,correction=?,updated=?,eligible_at=?,ready_at=?,resources_json=?,signatures_json=? WHERE episode_id=?",
                                     (revision, state, int(eligible), int(correction), now, eligible_at,
                                      now if state == "READY" else None, packed(sorted(resources)),
                                      packed(sorted(signatures)), episode_id))
                        # Maximum age is checked only at this completed-turn boundary.
                        conn.execute("UPDATE episodes SET state='READY',ready_at=? WHERE generation=? AND state='ELIGIBLE' AND eligible_at<=?",
                                     (now, auth["generation"], now - ELIGIBLE_AGE_SECONDS))
            self._wake.set()
            return Admission("ELIGIBLE" if eligible else "SKIPPED", "")
        except (OSError, ValueError, sqlite3.Error, TimeoutError) as exc:
            return Admission("FAILED", error_detail("episode.admission", exc) + "; " + REJECTED)

    def flush_session(self, session_id=None, *, retire=False):
        """Dispatch eligible work; explicit normal exit also retires inactive state."""
        if self.closed or not self.enabled:
            return
        with gate(self.entry):
            auth = read_auth(self.roster, self.entry)
            if not self._authorized(auth):
                return
            with closing(connect(self.entry)) as conn, conn:
                where, params = "generation=?", [auth["generation"]]
                if session_id is not None:
                    where += " AND session_id=?"
                    params.append(str(session_id)[:128])
                now = time.time()
                conn.execute(f"UPDATE episodes SET state='READY',ready_at=? WHERE {where} AND state='ELIGIBLE'", [now, *params])
                draft_ids = [row[0] for row in conn.execute(f"SELECT episode_id FROM episodes WHERE {where} AND state='DRAFT'", params)]
                for episode_id in draft_ids:
                    conn.execute("DELETE FROM episode_sources WHERE episode_id=?", (episode_id,))
                    conn.execute("DELETE FROM episodes WHERE episode_id=?", (episode_id,))
                if retire and session_id is not None:
                    inactive = conn.execute(
                        "SELECT episode_id FROM episodes WHERE generation=? AND session_id=? "
                        "AND state IN ('ACKED','CHALLENGED') AND last_job_id IS NULL "
                        "AND NOT EXISTS (SELECT 1 FROM episode_sources s WHERE s.episode_id=episodes.episode_id AND s.job_id IS NOT NULL)",
                        (auth["generation"], str(session_id)[:128]),
                    ).fetchall()
                    for row in inactive:
                        conn.execute("DELETE FROM episode_sources WHERE episode_id=?", (row["episode_id"],))
                        conn.execute("DELETE FROM episodes WHERE episode_id=?", (row["episode_id"],))
        self._wake.set()

    @staticmethod
    def _original_anchor(source):
        try:
            messages = json.loads(source["messages_json"])
        except (TypeError, ValueError):
            return "", 0, 0
        first = next((message for message in messages
                      if isinstance(message, dict) and message.get("role") == "user"), None)
        if not first:
            return "", 0, 0
        value = packed(dict(context_only=True, kind="original_request", messages=[first]))
        observed = len(value.encode())
        if observed > CARRY_BYTES:
            value = packed(dict(context_only=True, kind="original_request", missing_context=True,
                                reason="carry_anchor_bytes", observed_bytes=observed,
                                limit_bytes=CARRY_BYTES))
        return value, len(value.encode()), 1

    def _ack(self, conn, job, outcome, result=None):
        try:
            host = json.loads(job["host_json"])
        except (TypeError, ValueError):
            host = {}
        conn.execute("UPDATE jobs SET status=?,detail=?,evidence_json='',catalog_json='',host_json='',result_json='' WHERE job_id=? AND status='RESULT'",
                     (outcome.status, outcome.detail, job["job_id"]))
        if host.get("kind") == "episode" and isinstance(host.get("episode_id"), str):
            episode = conn.execute("SELECT * FROM episodes WHERE episode_id=?", (host["episode_id"],)).fetchone()
            if episode and episode["revision"] == host.get("revision") and episode["last_job_id"] == job["job_id"]:
                if not episode["anchor_json"]:
                    first = conn.execute(
                        "SELECT * FROM episode_sources WHERE episode_id=? AND job_id=? ORDER BY seq LIMIT 1",
                        (host["episode_id"], job["job_id"]),
                    ).fetchone()
                    if first:
                        anchor_json, anchor_bytes, anchor_messages = self._original_anchor(first)
                        conn.execute(
                            "UPDATE episodes SET anchor_json=?,anchor_bytes=?,anchor_messages=?,anchor_task_id=? WHERE episode_id=?",
                            (anchor_json, anchor_bytes, anchor_messages, first["task_id"], host["episode_id"]))
                related = json.loads(episode["related_skills_json"] or "[]")
                proposal = result.get("proposal") if isinstance(result, dict) else None
                if outcome.status == "APPLIED" and isinstance(proposal, dict):
                    if proposal.get("action") == "CREATE" and isinstance(proposal.get("name"), str):
                        related.append(dict(action="CREATE", name=proposal["name"][:128]))
                    elif proposal.get("action") == "UPDATE" and isinstance(proposal.get("target_id"), str):
                        target = next((item for item in host.get("targets", [])
                                       if item.get("target_id") == proposal["target_id"]), None)
                        if target and isinstance(target.get("name"), str):
                            related.append(dict(action="UPDATE", name=target["name"][:128]))
                    conn.execute("UPDATE episodes SET related_skills_json=? WHERE episode_id=?",
                                 (packed(related[-8:]), host["episode_id"]))
                consumed = conn.execute(
                    "SELECT event_json FROM episode_sources WHERE episode_id=? AND job_id=? ORDER BY seq",
                    (host["episode_id"], job["job_id"]),
                ).fetchall()
                if (isinstance(result, dict) and result.get("status") == "PROPOSAL"
                        and outcome.status not in ("FAILED", "INVALID")):
                    self._record_fingerprints(conn, consumed)
                conn.execute("DELETE FROM episode_sources WHERE episode_id=? AND job_id=?",
                             (host["episode_id"], job["job_id"]))
                self._remaining_episode_state(conn, episode)
        else:
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
                now = time.time()
                conn.execute("UPDATE episodes SET state='READY',ready_at=? WHERE generation=? AND state='ELIGIBLE' AND updated<=?",
                             (now, auth["generation"], now - IDLE_SECONDS))
                for row in conn.execute("SELECT * FROM jobs WHERE status='RESULT' ORDER BY seq").fetchall():
                    job = dict(row)
                    with diagnostic_stage("owner.delivery", job_id=job["job_id"]):
                        if not valid_job(auth, self.entry, job) or not hmac.compare_digest(job["result_id"], result_seal(auth, job)):
                            continue
                        result = json.loads(job["result_json"])
                        from skill_catalog import Publication
                        host = json.loads(job["host_json"])
                        if host.get("kind") == "episode":
                            episode = conn.execute("SELECT revision,last_job_id FROM episodes WHERE episode_id=?", (host.get("episode_id"),)).fetchone()
                            if not episode or episode["revision"] != host.get("revision") or episode["last_job_id"] != job["job_id"]:
                                conn.execute("UPDATE jobs SET status='SUPERSEDED',detail='stage=owner.freshness reason=stale_revision',evidence_json='',catalog_json='',host_json='',result_json='' WHERE job_id=?",
                                             (job["job_id"],))
                                conn.execute("UPDATE episode_sources SET job_id=NULL WHERE job_id=?", (job["job_id"],))
                                conn.execute("UPDATE episodes SET last_job_id=NULL WHERE episode_id=? AND last_job_id=?",
                                             (host.get("episode_id"), job["job_id"]))
                                continue
                        if result["status"] == "SUPERSEDED":
                            conn.execute("UPDATE jobs SET status='SUPERSEDED',detail=?,evidence_json='',catalog_json='',host_json='',result_json='' WHERE job_id=?",
                                         (result.get("detail", "stage=owner.freshness reason=stale_revision"), job["job_id"]))
                            conn.execute("UPDATE episode_sources SET job_id=NULL WHERE job_id=?", (job["job_id"],))
                            if host.get("kind") == "episode":
                                conn.execute("UPDATE episodes SET last_job_id=NULL WHERE episode_id=? AND last_job_id=?",
                                             (host.get("episode_id"), job["job_id"]))
                            continue
                        if result["status"] in ("INVALID", "FAILED", "BUDGET_REFUSED"):
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
                        self._ack(conn, job, outcome, result)
                if conn.execute("SELECT 1 FROM jobs WHERE status IN ('PREPARED','RUNNING','RESULT') LIMIT 1").fetchone():
                    return
                episode = conn.execute("SELECT * FROM episodes WHERE generation=? AND state='READY' ORDER BY coalesce(ready_at,created),seq LIMIT 1",
                                       (auth["generation"],)).fetchone()
                selected, size = [], 0
                if not episode:
                    for row in conn.execute("SELECT * FROM evidence WHERE job_id IS NULL AND generation=? ORDER BY seq LIMIT ?", (auth["generation"], self.batch_count)):
                        if selected and size + row["bytes"] > BATCH_BYTES:
                            break
                        selected.append(dict(row))
                        size += row["bytes"]
                generation = auth["generation"]
        if not selected and not episode:
            return
        # The owner snapshots its private catalog; episode assembly and selection
        # remain service work outside the owner/foreground process.
        with diagnostic_stage("owner.preparation"):
            snapshot = self.store.prepare_review()
            if episode:
                with closing(connect(self.entry, readonly=True)) as source_conn:
                    refs = [dict(source_id=row["source_id"], revision=row["revision"], source_hash=row["source_hash"])
                            for row in source_conn.execute("SELECT source_id,revision,source_hash FROM episode_sources WHERE episode_id=? AND revision<=? ORDER BY seq",
                                                           (episode["episode_id"], episode["revision"]))]
                evidence_json = packed(dict(kind="episode", episode_id=episode["episode_id"], revision=episode["revision"], sources=refs))
                host_json = packed(dict(store_id=snapshot.store_id, targets=[asdict(t) for t in snapshot.targets],
                                        kind="episode", episode_id=episode["episode_id"], revision=episode["revision"]))
                created = episode["eligible_at"] or episode["created"]
            else:
                evidence_json = packed([dict(session_id=r["session_id"], task_id=r["task_id"], messages=json.loads(r["messages_json"])) for r in selected])
                host_json = packed(dict(store_id=snapshot.store_id, targets=[asdict(t) for t in snapshot.targets], kind="legacy"))
                created = selected[0]["created"]
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
                if episode and not conn.execute("SELECT 1 FROM episodes WHERE episode_id=? AND revision=? AND state='READY'",
                                                (episode["episode_id"], episode["revision"])).fetchone():
                    return
                conn.execute("INSERT INTO jobs(job_id,profile_id,store_id,generation,evidence_json,catalog_json,host_json,input_seal,created,status) VALUES(?,?,?,?,?,?,?,?,?,'PREPARED')",
                             tuple(job[k] for k in JOB_FIELDS) + (job_seal(auth, job), created))
                if episode:
                    conn.execute("UPDATE episode_sources SET job_id=? WHERE episode_id=? AND revision<=? AND generation=? AND job_id IS NULL",
                                 (job["job_id"], episode["episode_id"], episode["revision"], generation))
                    conn.execute("UPDATE episodes SET state='PREPARED',last_job_id=? WHERE episode_id=? AND revision=? AND state='READY'",
                                 (job["job_id"], episode["episode_id"], episode["revision"]))
                else:
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
        if self.enabled:
            try:
                self.flush_session()
                self.pump()
            except (OSError, ValueError, sqlite3.Error, TimeoutError):
                pass
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

    @staticmethod
    def _required_source(row, first=False):
        data = Owner._event_data(row["event_json"])
        events = [event for event in data["events"] if isinstance(event, dict)]
        evidentiary = any(event.get("outcome") in ("failure", "metadata_success")
                          or (event.get("outcome") == "business_success" and event.get("nonroutine"))
                          for event in events)
        return first or data.get("correction") is True or evidentiary

    def _episode_request(self, entry, job):
        manifest = json.loads(job["evidence_json"])
        if manifest.get("kind") != "episode" or not isinstance(manifest.get("sources"), list):
            raise ValueError("Invalid episode manifest")
        refs = manifest["sources"]
        if len(refs) > EPISODE_MESSAGES:
            raise BudgetRefusal("episode_sources", len(refs), EPISODE_MESSAGES)
        with closing(connect(entry, readonly=True)) as conn:
            episode = conn.execute("SELECT revision,state,last_job_id,anchor_json,anchor_bytes,anchor_messages,anchor_task_id,related_skills_json,retired_sources,retired_bytes,retired_messages FROM episodes WHERE episode_id=? AND generation=?",
                                   (manifest.get("episode_id"), job["generation"])).fetchone()
            if (not episode or episode["revision"] != manifest.get("revision")
                    or episode["state"] != "PREPARED" or episode["last_job_id"] != job["job_id"]):
                return None
            rows_by_id = {}
            for ref in refs:
                if not isinstance(ref, dict) or not isinstance(ref.get("source_id"), str):
                    raise ValueError("Invalid episode source reference")
                row = conn.execute("SELECT * FROM episode_sources WHERE source_id=? AND episode_id=? AND generation=? AND job_id=?",
                                   (ref["source_id"], manifest["episode_id"], job["generation"], job["job_id"])).fetchone()
                if not row:
                    raise ValueError("Missing episode source")
                row = dict(row)
                digest = hashlib.sha256((row["messages_json"] + "\n" + row["event_json"]).encode()).hexdigest()
                if (row["revision"] != ref.get("revision") or digest != ref.get("source_hash")):
                    raise ValueError("Episode source integrity failure")
                rows_by_id[ref["source_id"]] = row
        ordered = [rows_by_id[ref["source_id"]] for ref in refs]
        required = {index for index, row in enumerate(ordered) if self._required_source(row, first=index == 0)}
        include_anchor = bool(episode["anchor_json"] and not any(
            row["task_id"] == episode["anchor_task_id"] for row in ordered))
        if include_anchor and episode["anchor_bytes"] > CARRY_BYTES:
            raise BudgetRefusal("carry_anchor_bytes", episode["anchor_bytes"], CARRY_BYTES)
        anchor = json.loads(episode["anchor_json"]) if include_anchor else None
        if isinstance(anchor, dict) and anchor.get("missing_context") is True:
            raise BudgetRefusal("missing_context", anchor.get("observed_bytes", CARRY_BYTES + 1), CARRY_BYTES)

        def task_for(indices):
            sources = []
            message_count = episode["anchor_messages"] if include_anchor else 0
            for index in sorted(indices):
                row = ordered[index]
                messages = json.loads(row["messages_json"])
                message_count += len(messages)
                sources.append(dict(source_id=row["source_id"], task_id=row["task_id"],
                                    revision=row["revision"], messages=messages))
            task = dict(episode_id=manifest["episode_id"], revision=manifest["revision"],
                        association="observed-resource-or-session", sources=sources,
                        omitted_optional_sources=len(ordered) - len(indices) + episode["retired_sources"])
            if episode["retired_sources"]:
                task["retired_optional_context"] = dict(
                    reason="episode_capacity_optional_routine",
                    whole_sources=episode["retired_sources"],
                    messages=episode["retired_messages"],
                    bytes=episode["retired_bytes"],
                )
            if include_anchor:
                task["context_only_anchor"] = anchor
            related = json.loads(episode["related_skills_json"] or "[]")
            if related:
                task["related_skills"] = related
            view = packed(dict(tasks=[task]))
            return task, len(view.encode()), message_count

        chosen = set(required)
        task, view_bytes, message_count = task_for(chosen)
        if message_count > BATCH_MESSAGES:
            raise BudgetRefusal("selected_messages", message_count, BATCH_MESSAGES)
        if view_bytes > BATCH_BYTES:
            raise BudgetRefusal("selected_view_bytes", view_bytes, BATCH_BYTES)
        for index in range(len(ordered)):
            if index in chosen:
                continue
            candidate = chosen | {index}
            next_task, next_bytes, next_messages = task_for(candidate)
            if next_bytes <= BATCH_BYTES and next_messages <= BATCH_MESSAGES:
                chosen, task, view_bytes, message_count = candidate, next_task, next_bytes, next_messages
        request = packed(dict(catalog=json.loads(job["catalog_json"]), tasks=[task]))
        prepared_bytes = len(request.encode())
        if prepared_bytes > 128 * 1024:
            raise BudgetRefusal("prepared_bytes", prepared_bytes, 128 * 1024)
        return request

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
            host = json.loads(job["host_json"])
            request = self._episode_request(entry, job) if host.get("kind") == "episode" else request_json(self.roster, job)
        except BudgetRefusal as exc:
            return dict(status="BUDGET_REFUSED", detail=exc.detail())
        except (OSError, sqlite3.Error, TimeoutError, MailboxCapacityError) as exc:
            raise DiagnosticFailure("service.preparation", exc, job["job_id"], sqlite_busy_timeout_s=BUSY_SECONDS) from None
        except ValueError as exc:
            return failure("service.preparation_validation", "INVALID", exc)
        if request is None:
            return dict(status="SUPERSEDED", detail="stage=service.freshness reason=stale_revision; outcome=SUPERSEDED; zero provider requests; accepted source references retained.")
        try:
            proposal = provider(request)
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
