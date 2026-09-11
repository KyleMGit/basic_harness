"""Confined catalog authority and single-file generated skill publication."""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import uuid

from safety import screen_prompt_content
from review_diagnostics import ReviewDiagnosticError
from skill_lock import CatalogLock, path_identity


MAX_FILE_BYTES = 256 * 1024
MAX_OUTPUT_BYTES = 24 * 1024


@dataclass(frozen=True)
class ReviewTarget:
    target_id: str
    name: str
    relative_path: str
    revision: str


@dataclass(frozen=True)
class CatalogSnapshot:
    store_id: str
    public_json: str
    targets: tuple[ReviewTarget, ...]


@dataclass(frozen=True)
class Publication:
    status: str
    detail: str = ""


def validate_proposal(value):
    """A strict data contract; routing/revision/receipt authority is never model input."""
    if not isinstance(value, dict):
        raise ReviewDiagnosticError("proposal_not_object")
    action = value.get("action")
    if action == "NONE" and set(value) == {"action"}:
        return value
    fields = {"action", "description", "instructions", "complete"}
    fields.add("name" if action == "CREATE" else "target_id")
    if action not in ("CREATE", "UPDATE") or set(value) != fields:
        raise ReviewDiagnosticError("invalid_action_or_fields")
    if value["complete"] is not True:
        raise ReviewDiagnosticError("incomplete_complete_flag")
    for key in fields - {"complete"}:
        if not isinstance(value[key], str) or not value[key].strip():
            raise ReviewDiagnosticError("empty_fields")
    normalized_bytes = len(json.dumps(value).encode())
    if normalized_bytes > MAX_OUTPUT_BYTES:
        raise ReviewDiagnosticError(
            "output_oversized", observed_bytes=normalized_bytes, limit_bytes=MAX_OUTPUT_BYTES)
    if action == "CREATE" and (len(value["name"]) > 128 or re.search(r"[/\\:\n\r]", value["name"])):
        raise ReviewDiagnosticError("invalid_name")
    if re.search(r"[\r\n]", value["description"]) or len(value["description"]) > 1024:
        raise ReviewDiagnosticError("invalid_description")
    instructions = value["instructions"]
    if (instructions.count("```") % 2 or instructions.count("~~~") % 2
            or re.search(r"\[(?:TRUNCATED|INCOMPLETE)\]|<rest omitted>|TODO|\.\.\.\s*$", instructions, re.I)):
        raise ReviewDiagnosticError("incomplete_instructions")
    if not screen_prompt_content("\n".join(value[k] for k in fields - {"complete"}))[0]:
        raise ReviewDiagnosticError("safety_rejection")
    return value


class CanonicalCatalog:
    def __init__(self, storage_dir=None, *, bound=False):
        self._storage_dir = os.path.abspath(storage_dir or os.path.join(os.getcwd(), ".agent_skills"))
        self._bound = bound
        self._root_identity = os.path.realpath(self._storage_dir)
        self._write_allowed = lambda: True

    def set_write_guard(self, callback):
        """Host-only mode check, evaluated inside every mutation's catalog lock."""
        self._write_allowed = callback

    @property
    def storage_dir(self):
        return self._storage_dir

    @storage_dir.setter
    def storage_dir(self, value):
        if self._bound:
            raise ValueError("Bound catalog identity cannot be changed")
        self._storage_dir = os.path.abspath(value)
        self._root_identity = os.path.realpath(self._storage_dir)

    def bind(self):
        return type(self)(self.storage_dir, bound=True)

    @property
    def store_id(self):
        return path_identity(self._root_identity)

    def lock(self, timeout=5.0):
        return CatalogLock(self._root_identity, timeout)

    def _confined(self, path):
        root = Path(self._root_identity)
        if Path(self.storage_dir).resolve() != root:
            raise ValueError("Catalog root identity changed")
        resolved = Path(path).resolve()
        if not resolved.is_relative_to(root) or resolved == root:
            raise ValueError("Skill path escapes bound catalog")
        return resolved

    def _read(self, path):
        path = self._confined(path)  # Includes symlinks/junctions, BEFORE opening.
        with path.open("rb") as handle:
            raw = handle.read(MAX_FILE_BYTES + 1)
        if len(raw) > MAX_FILE_BYTES:
            raise ValueError("Skill file exceeds supported byte limit")
        return raw

    @staticmethod
    def _metadata(text):
        match = re.match(r"^---\s*\n(.*?)\n---", text, re.S)
        metadata = {}
        if match:
            for line in match[1].splitlines():
                if line.startswith("hermes_") and ":" in line:
                    key, value = line.split(":", 1)
                    metadata[key] = json.loads(value)
        return metadata

    def _entries(self):
        """One resolver for all readers/writers; unknown pairs remain ambiguous."""
        entries = []
        for directory, dirs, files in os.walk(self.storage_dir, followlinks=False):
            safe_dirs = []
            for name in sorted(dirs):
                try:
                    self._confined(Path(directory, name))
                    safe_dirs.append(name)
                except (OSError, ValueError):
                    pass
            dirs[:] = safe_dirs
            for filename in sorted(files):
                path = Path(directory, filename)
                if path.suffix.lower() not in (".md", ".json"):
                    continue
                try:
                    raw = self._read(path)
                    text = raw.decode("utf-8-sig")
                    if path.suffix.lower() == ".json":
                        data = json.loads(text)
                        if not isinstance(data, dict):
                            continue
                        # Marked generated caches are NEVER authoritative, even orphaned.
                        if data.get("hermes_cache") == 1:
                            continue
                        meta = {}
                    else:
                        data = self.parse_markdown_skill(text, path.parent.name if filename == "SKILL.md" else path.stem)
                        meta = self._metadata(text)
                    if not isinstance(data.get("name"), str) or not isinstance(data.get("instructions"), str):
                        continue
                    aliases = {self._safe_name(data["name"]), self._safe_name(path.parent.name if filename == "SKILL.md" else path.stem)}
                    entries.append(dict(path=path, raw=raw, data=data, meta=meta, aliases=aliases, cache=None))
                except (OSError, ValueError, UnicodeError):
                    # An unreadable/confined invalid file still reserves its filename.
                    entries.append(dict(path=path, raw=None, data={}, meta={}, aliases={self._safe_name(path.stem)}, cache=None))
        removed = set()
        for index, item in enumerate(entries):
            if item["path"].suffix.lower() != ".json" or item["raw"] is None:
                continue
            data = item["data"]
            for md in entries:
                if (md["path"] == item["path"].with_suffix(".md") and md["raw"] is not None
                        and self._recognized_legacy_cache(md["path"], md["raw"], item["raw"], data)):
                    md["cache"] = item["path"]
                    removed.add(index)
        return [item for i, item in enumerate(entries) if i not in removed]

    def _resolve(self, name, entries=None, include_deleted=False):
        matches = [item for item in (entries if entries is not None else self._entries())
                   if self._safe_name(name.strip()) in item["aliases"]]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous skill name '{name}': {len(matches)} logical targets")
        if matches and (include_deleted or not matches[0]["meta"].get("hermes_deleted")):
            return matches[0]
        return None

    def resolve_skill_file(self, name):
        with self.lock():
            item = self._resolve(name)
            return str(self._confined(item["path"])) if item and item["raw"] is not None else None

    def get_all_skills(self):
        with self.lock():
            entries = self._entries()
            result = []
            for item in entries:
                if item["raw"] is None or item["meta"].get("hermes_deleted"):
                    continue
                try:
                    if any(self._resolve(alias, entries) is not item for alias in item["aliases"]):
                        continue
                except ValueError:
                    continue
                data = item["data"]
                if screen_prompt_content(str(data.get("description", "")) + "\n" + data["instructions"])[0]:
                    result.append(dict(data, file_path=str(item["path"])))
            return result

    def load_skill(self, name):
        with self.lock():
            try:
                item = self._resolve(name)
                if not item:
                    return f"Skill '{name}' not found in '{self.storage_dir}'."
                if item["raw"] is None:
                    return "Error: invalid or unconfined skill"
                data = item["data"]
                safe, reason = screen_prompt_content(str(data.get("description", "")) + "\n" + data["instructions"])
                if not safe:
                    return reason
                return f"=== SKILL: {data['name']} ===\nFile: {item['path']}\nDescription: {data.get('description', '')}\n\nInstructions:\n{data['instructions']}"
            except (OSError, ValueError) as exc:
                return f"Error: {exc}"

    def _atomic(self, path, raw):
        path = self._confined(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = None
        try:
            with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=".skill-", suffix=".tmp", delete=False) as handle:
                pending = handle.name
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            self._confined(path)
            os.replace(pending, path)
        finally:
            if pending and os.path.exists(pending):
                os.unlink(pending)

    def _publish(self, path, name, description, instructions, tags=(), receipts=(), deleted=False):
        # Independently imported JSON at the cache slot is never overwritten.
        cache_path = self._confined(Path(path).with_suffix(".json"))
        cache_allowed = self._cache_slot_available(path)
        text = self.format_markdown_skill(name, description, instructions, tags)
        meta = dict(hermes_generated=1, hermes_receipts=list(receipts), hermes_deleted=deleted)
        if cache_allowed and cache_path.exists():
            cache_raw = self._read(cache_path)
            if json.loads(cache_raw).get("hermes_cache") != 1:
                # Commit the recognized old cache's identity WITH the new
                # authority, before replacing JSON. A failed/crashed cache write
                # must not turn those exact legacy bytes into a second authority.
                meta["hermes_legacy_cache"] = self._legacy_cache_identity(path, cache_raw)
        text = text.replace("---\n", "---\n" + "".join(k + ": " + json.dumps(v) + "\n" for k, v in meta.items()), 1)
        raw = text.encode("utf-8")
        if len(raw) > MAX_FILE_BYTES:
            raise ValueError("Authoritative skill exceeds supported byte limit")
        self._atomic(path, raw)  # Sole authoritative commit. No rollback after this.
        if not cache_allowed:
            return Publication("APPLIED", "Markdown committed; independent JSON preserved")
        try:
            cache = dict(name=name, description=description, instructions=instructions, tags=list(tags),
                         hermes_cache=1, revision=hashlib.sha256(raw).hexdigest(), file=Path(path).name)
            self._atomic(cache_path, json.dumps(cache, indent=2).encode())
        except (OSError, ValueError) as exc:
            return Publication("APPLIED", f"Markdown committed; cache unavailable: {type(exc).__name__}")
        return Publication("APPLIED", "Markdown committed; revision-bound cache written")

    @staticmethod
    def _legacy_cache_identity(path, cache_raw):
        return dict(file=Path(path).with_suffix(".json").name,
                    sha256=hashlib.sha256(cache_raw).hexdigest())

    def _recognized_legacy_cache(self, path, md_raw, cache_raw, data):
        # Narrow initial recognition, or exact recovery proof atomically bound
        # to canonical Markdown. A matching name alone grants no cache authority.
        if (set(data) != {"name", "description", "instructions", "tags", "format", "file"}
                or data["format"] != "markdown" or data["file"] != Path(path).name):
            return False
        text = md_raw.decode("utf-8-sig").replace("\r\n", "\n")
        metadata = self._metadata(text)
        if (metadata.get("hermes_generated") == 1
                and metadata.get("hermes_legacy_cache") == self._legacy_cache_identity(path, cache_raw)):
            return True
        return text == self.format_markdown_skill(
            data["name"], data["description"], data["instructions"], data["tags"]).replace("\r\n", "\n")

    def _cache_slot_available(self, path):
        cache_path = self._confined(Path(path).with_suffix(".json"))
        if not cache_path.exists():
            return True
        try:
            cache_raw = self._read(cache_path)
            data = json.loads(cache_raw)
            if data.get("hermes_cache") == 1:
                return True
            return self._recognized_legacy_cache(path, self._read(path), cache_raw, data)
        except (OSError, ValueError, TypeError, AttributeError):
            return False

    def save_skill(self, name, description, instructions, tags=None):
        """CREATE only. No update, overwrite, receipt, or model authorization argument."""
        name, description, instructions = name.strip(), description.strip(), instructions.strip()
        safe, reason = screen_prompt_content("\n".join((name, description, instructions, json.dumps(tags or []))))
        if not safe:
            return reason
        if (not self._safe_name(name) or len(name) > 128 or re.search(r"[/\\:\n\r]", name)
                or re.search(r"[\r\n]", description) or not isinstance(tags or [], (list, tuple))
                or len(tags or []) > 16 or any(not isinstance(t, str) or len(t) > 64 or re.search(r"[\r\n:\[\]]", t) for t in tags or [])):
            return "Refused: invalid skill name/description"
        with self.lock():
            if not self._write_allowed():
                return "Refused: profile writes are disabled"
            try:
                existing = self._resolve(name, include_deleted=True)
                if existing and not existing["meta"].get("hermes_deleted"):
                    return f"Refused to create skill '{name}': conflicts with existing skill '{existing['data'].get('name', name)}' after name normalization. Load and reuse the existing skill; use post-task reflection for a curated update."
                path = existing["path"] if existing else Path(self.storage_dir, self._safe_name(name) + ".md")
                receipts = existing["meta"].get("hermes_receipts", []) if existing else []
                result = self._publish(path, name, description, instructions, tags or [], receipts)
                return f"Skill '{name}' successfully saved as '{path}'. {result.detail}"
            except (OSError, ValueError) as exc:
                return f"Refused to save skill '{name}': {exc}"

    def prepare_review(self, max_targets=8, max_target_bytes=12 * 1024, max_bytes=48 * 1024):
        with self.lock():
            entries = self._entries()
            catalog, public_targets, targets = [], [], []
            for item in entries:
                if item["raw"] is None or item["meta"].get("hermes_deleted"):
                    continue
                try:
                    if any(self._resolve(alias, entries) is not item for alias in item["aliases"]):
                        continue
                except ValueError:
                    continue
                data = item["data"]
                if not screen_prompt_content(json.dumps(data))[0]:
                    continue
                if len(catalog) < 128:
                    summary = dict(name=data["name"][:128], description=str(data.get("description", ""))[:256])
                    if len(json.dumps(catalog + [summary]).encode()) <= 16 * 1024:
                        catalog.append(summary)
                if (item["path"].suffix.lower() != ".md" or len(targets) >= max_targets or len(item["raw"]) > max_target_bytes
                        or not self._cache_slot_available(item["path"])):
                    continue
                target_id = uuid.uuid4().hex
                public = dict(target_id=target_id, name=data["name"], description=data.get("description", ""), instructions=data["instructions"])
                if len(json.dumps(dict(catalog=catalog, targets=public_targets + [public])).encode()) > max_bytes - 16 * 1024:
                    continue
                public_targets.append(public)
                targets.append(ReviewTarget(target_id, data["name"], str(item["path"].relative_to(self.storage_dir)), hashlib.sha256(item["raw"]).hexdigest()))
            return CatalogSnapshot(self.store_id, json.dumps(dict(catalog=catalog, targets=public_targets)), tuple(targets))

    def apply_review(self, proposal, snapshot, receipt):
        try:
            validate_proposal(proposal)
            if snapshot.store_id != self.store_id or not isinstance(receipt, str) or not receipt or len(receipt) > 128:
                raise ValueError("Invalid host publication identity")
            if proposal["action"] == "NONE":
                return Publication("NONE")
            with self.lock():
                if not self._write_allowed():
                    return Publication("CANCELLED", "Profile writes are disabled")
                if proposal["action"] == "UPDATE":
                    target = next((t for t in snapshot.targets if t.target_id == proposal["target_id"]), None)
                    public = json.loads(snapshot.public_json)["targets"]
                    if target is None or not any(t["target_id"] == target.target_id for t in public):
                        raise ValueError("Unknown or ineligible target")
                    path = self._confined(Path(self.storage_dir, target.relative_path))
                    item = self._resolve(target.name, include_deleted=True)
                    if (not item or self._confined(item["path"]) != path or path.suffix.lower() != ".md"
                            or not self._cache_slot_available(path)):
                        raise ValueError("Target identity changed")
                    name = target.name
                else:
                    name = proposal["name"].strip()
                    if not self._safe_name(name):
                        raise ValueError("Empty normalized skill name")
                    item = self._resolve(name, include_deleted=True)
                    path = item["path"] if item else Path(self.storage_dir, self._safe_name(name) + ".md")
                receipts = item["meta"].get("hermes_receipts", []) if item else []
                if receipt in receipts:
                    return Publication("DUPLICATE", "Operation already committed, including if subsequently deleted")
                if proposal["action"] == "UPDATE":
                    if item["raw"] is None or item["meta"].get("hermes_deleted") or hashlib.sha256(item["raw"]).hexdigest() != target.revision:
                        return Publication("STALE", "Target revision changed")
                elif item and not item["meta"].get("hermes_deleted"):
                    return Publication("COLLISION", "Refused: normalized name is already present")
                return self._publish(path, name, proposal["description"], proposal["instructions"],
                                     item["data"].get("tags", []) if item else [], receipts + [receipt])
        except OSError as exc:
            return Publication("FAILED", type(exc).__name__)
        except (ValueError, TypeError, KeyError) as exc:
            return Publication("INVALID", str(exc))

    def delete_skill(self, name):
        with self.lock():
            if not self._write_allowed():
                return False
            item = self._resolve(name)
            if not item or item["raw"] is None:
                return False
            if item["path"].suffix.lower() == ".json":
                self._confined(item["path"]).unlink()
            else:
                self._publish(item["path"], item["data"]["name"], "Deleted skill", "", receipts=item["meta"].get("hermes_receipts", []), deleted=True)
            return True
