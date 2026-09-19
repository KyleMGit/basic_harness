"""
Hermes-inspired Persistent Memory System:
1. USER.md: Operator profile, technical background, and communication preferences.
2. MEMORY.md: Project architecture, environment configuration, and codebase facts.
3. AutoMemoryExtractor: Autonomous reflection engine that extracts durable preferences
   and project facts from conversation turns.
"""

import json
import os
import re
import tempfile
from typing import Any, Dict, List, Optional, Tuple
from safety import screen_prompt_content


class _MarkdownMemoryManager:
    """Shared, bounded, atomic operations for one Markdown memory store."""

    FILE_LABEL = "memory"
    ITEM_LABEL = "Item"
    MAX_OPERATIONS = 8

    @staticmethod
    def _result(
        status: str,
        changed: bool,
        message: str,
        errors: Optional[List[str]] = None,
        error_code: Optional[str] = None,
        **diagnostics: Any,
    ) -> Dict[str, Any]:
        result = {
            "status": status,
            "changed": changed,
            "message": message,
            "errors": list(errors or []),
        }
        if error_code is not None:
            result["error_code"] = error_code
        result.update(diagnostics)
        return result

    @staticmethod
    def _normalize(value: str) -> str:
        value = value.strip()
        if value.startswith("- "):
            value = value[2:]
        return " ".join(value.split()).casefold()

    @staticmethod
    def _normalize_heading(value: str) -> str:
        return " ".join(value.split()).casefold()

    @staticmethod
    def _clean_value(value: str) -> str:
        value = value.strip()
        if value.startswith("- "):
            value = value[2:].strip()
        return value

    @staticmethod
    def _is_single_line(value: str) -> bool:
        return not any(separator in value for separator in "\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029")

    @staticmethod
    def _line_parts(line: str) -> Tuple[str, str]:
        if line.endswith("\r\n"):
            return line[:-2], "\r\n"
        if line.endswith("\n"):
            return line[:-1], "\n"
        if line.endswith("\r"):
            return line[:-1], "\r"
        return line, ""

    @staticmethod
    def _dominant_newline(content: str) -> str:
        crlf = content.count("\r\n")
        lf = content.count("\n") - crlf
        if crlf > lf:
            return "\r\n"
        return "\n"

    @classmethod
    def _heading_name(cls, line: str) -> Optional[str]:
        body, _ = cls._line_parts(line)
        match = re.fullmatch(r"##[ \t]+(.+?)[ \t]*", body)
        return match.group(1) if match else None

    @classmethod
    def _bullet_value(cls, line: str) -> Optional[str]:
        body, _ = cls._line_parts(line)
        stripped = body.lstrip(" \t")
        return stripped[2:] if stripped.startswith("- ") else None

    def _resolve_existing_file(self) -> bool:
        if os.path.isfile(self.file_path):
            return True
        root_path = os.path.join(os.getcwd(), os.path.basename(self.file_path))
        if self.allow_root_fallback and os.path.isfile(root_path):
            self.file_path = root_path
            return True
        return False

    def _read_for_update(self) -> Tuple[Optional[str], bool, Optional[str]]:
        exists = self._resolve_existing_file()
        if not exists:
            return None, False, None
        try:
            with open(self.file_path, "r", encoding="utf-8", newline="") as handle:
                content = handle.read()
        except Exception as exc:
            return None, True, f"Error loading {self.FILE_LABEL}; existing content was not modified: {exc}"
        safe, status = screen_prompt_content(content)
        if not safe:
            return None, True, status
        return content, True, None

    def capacity_snapshot(self) -> Dict[str, Any]:
        """Return the exact safe editable document and its actual instance budget."""
        content, exists, error = self._read_for_update()
        if error:
            return {
                "available": False,
                "error_code": "snapshot_unavailable",
                "status": "The memory store could not be read safely.",
            }
        authoritative = content if exists else self.DEFAULT_TEMPLATE.strip() + "\n"
        if authoritative is None:
            authoritative = ""
        current_chars = len(authoritative)
        limit = self.MAX_CHAR_BUDGET
        percent_used = (current_chars * 100 / limit) if limit > 0 else 100.0
        return {
            "available": True,
            "exists": exists,
            "content": authoritative,
            "current_chars": current_chars,
            "limit": limit,
            "remaining_chars": max(0, limit - current_chars),
            "percent_used": percent_used,
        }

    def _atomic_publish(self, content: str) -> Optional[str]:
        directory = os.path.dirname(self.file_path) or "."
        temp_path = None
        try:
            os.makedirs(directory, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                dir=directory,
                prefix=f".{os.path.basename(self.file_path)}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temp_path = handle.name
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.file_path)
            temp_path = None
            return None
        except Exception as exc:
            return str(exc)
        finally:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    def _validate_operations(self, operations: Any) -> Tuple[Optional[List[Dict[str, str]]], List[str]]:
        if not isinstance(operations, list) or not operations:
            return None, ["operations must be a non-empty array."]
        if len(operations) > self.MAX_OPERATIONS:
            return None, [f"operations is limited to {self.MAX_OPERATIONS} items per store."]
        normalized: List[Dict[str, str]] = []
        errors: List[str] = []
        allowed = {"action", "category", "value", "old_text"}
        for index, operation in enumerate(operations):
            prefix = f"operation {index + 1}"
            if not isinstance(operation, dict):
                errors.append(f"{prefix} must be an object.")
                continue
            unknown = set(operation) - allowed
            if unknown:
                errors.append(f"{prefix} has unknown fields: {', '.join(sorted(unknown))}.")
                continue
            if "action" not in operation:
                errors.append(f"{prefix} requires an action.")
                continue
            action_value = operation.get("action")
            if not isinstance(action_value, str):
                errors.append(f"{prefix} action must be a string.")
                continue
            action = action_value.strip().upper()
            if action not in {"ADD", "REPLACE", "REMOVE"}:
                errors.append(f"{prefix} action must be ADD, REPLACE, or REMOVE.")
                continue
            category = operation.get("category")
            value = operation.get("value")
            old_text = operation.get("old_text")
            fields = (("category", category), ("value", value), ("old_text", old_text))
            if any(field is not None and not isinstance(field, str) for _, field in fields):
                errors.append(f"{prefix} fields must be strings when provided.")
                continue
            if not category or not category.strip() or not self._is_single_line(category):
                errors.append(f"{prefix} requires a nonempty single-line category.")
                continue
            if action in {"ADD", "REPLACE"}:
                if not value or not self._clean_value(value) or not self._is_single_line(value):
                    errors.append(f"{prefix} {action} requires a nonempty single-line value.")
                    continue
            elif "value" in operation:
                errors.append(f"{prefix} REMOVE must omit value.")
                continue
            if action in {"REPLACE", "REMOVE"}:
                if not old_text or not self._clean_value(old_text) or not self._is_single_line(old_text):
                    errors.append(f"{prefix} {action} requires nonempty single-line old_text.")
                    continue
            elif "old_text" in operation:
                errors.append(f"{prefix} ADD must omit old_text.")
                continue
            item = {"action": action, "category": category.strip()}
            if value is not None:
                item["value"] = self._clean_value(value)
            if old_text is not None:
                item["old_text"] = self._clean_value(old_text)
            safe, status = screen_prompt_content("\n".join(item.values()))
            if not safe:
                errors.append(f"{prefix}: {status}")
                continue
            normalized.append(item)
        return (normalized if not errors else None), errors

    def _document_parts(self, content: str) -> Tuple[List[str], List[Tuple[int, str]]]:
        lines = content.splitlines(keepends=True)
        headings = []
        for index, line in enumerate(lines):
            heading = self._heading_name(line)
            if heading is not None:
                headings.append((index, self._normalize_heading(heading)))
        return lines, headings

    def _has_continuation(self, lines: List[str], target: int, section_end: int) -> bool:
        for line in lines[target + 1:section_end]:
            body, _ = self._line_parts(line)
            if not body.strip():
                continue
            if self._bullet_value(line) is not None or re.match(r"^\s*#{1,6}(?:\s|$)", body):
                return False
            return True
        return False

    def _apply_one(self, content: str, operation: Dict[str, str], newline: str, preserve_eof: bool) -> Tuple[str, str, Optional[str]]:
        lines, headings = self._document_parts(content)
        category_key = self._normalize_heading(operation["category"])
        matches = [position for position, name in headings if name == category_key]
        if len(matches) > 1:
            return content, "error", f"Category '{operation['category']}' is ambiguous because its heading appears more than once."

        all_bullets = [
            (index, self._normalize(value))
            for index, line in enumerate(lines)
            for value in [self._bullet_value(line)]
            if value is not None
        ]
        action = operation["action"]
        if action == "ADD":
            value_key = self._normalize(operation["value"])
            if any(existing == value_key for _, existing in all_bullets):
                return content, "no_op", None
            if not matches:
                separator = ""
                if content:
                    if content.endswith(("\r\n", "\n", "\r")):
                        separator = newline if not content.endswith(newline + newline) else ""
                    else:
                        separator = newline + newline
                candidate = content + separator + f"## {operation['category']}{newline}- {operation['value']}{newline}"
            else:
                start = matches[0]
                later_headings = [position for position, _ in headings if position > start]
                end = min(later_headings) if later_headings else len(lines)
                insert_at = end
                while insert_at > start + 1 and not self._line_parts(lines[insert_at - 1])[0].strip():
                    insert_at -= 1
                if insert_at > 0:
                    body, ending = self._line_parts(lines[insert_at - 1])
                    if not ending:
                        lines[insert_at - 1] = body + newline
                lines.insert(insert_at, f"- {operation['value']}{newline}")
                candidate = "".join(lines)
        else:
            if not matches:
                return content, "error", f"Category '{operation['category']}' was not found."
            start = matches[0]
            later_headings = [position for position, _ in headings if position > start]
            end = min(later_headings) if later_headings else len(lines)
            old_key = self._normalize(operation["old_text"])
            targets = [
                index for index in range(start + 1, end)
                if self._bullet_value(lines[index]) is not None
                and self._normalize(self._bullet_value(lines[index]) or "") == old_key
            ]
            if len(targets) != 1:
                reason = "was not found" if not targets else "is ambiguous"
                return content, "error", f"Target '{operation['old_text']}' {reason} in category '{operation['category']}'."
            target = targets[0]
            if self._has_continuation(lines, target, end):
                return content, "error", f"Target '{operation['old_text']}' has continuation text and cannot be edited safely."
            if action == "REPLACE":
                value_key = self._normalize(operation["value"])
                if value_key == old_key:
                    return content, "no_op", None
                if any(index != target and existing == value_key for index, existing in all_bullets):
                    return content, "error", "REPLACE would duplicate another existing entry."
                _, ending = self._line_parts(lines[target])
                lines[target] = f"- {operation['value']}{ending}"
            else:
                del lines[target]
            candidate = "".join(lines)

        if preserve_eof:
            if candidate and not candidate.endswith(("\r\n", "\n", "\r")):
                candidate += newline
        elif candidate.endswith("\r\n"):
            candidate = candidate[:-2]
        elif candidate.endswith(("\n", "\r")):
            candidate = candidate[:-1]
        return candidate, "applied", None

    def apply_operations(self, operations: Any) -> Dict[str, Any]:
        validated, errors = self._validate_operations(operations)
        if errors:
            error_code = (
                "unsafe_candidate"
                if any("Rejected/quarantined" in error for error in errors)
                else "invalid_operations"
            )
            return self._result(
                "error", False, f"Error updating {self.FILE_LABEL}.", errors,
                error_code=error_code,
            )

        current, existed, load_error = self._read_for_update()
        if load_error:
            return self._result(
                "error", False, load_error, [load_error], error_code="snapshot_unavailable"
            )
        assert validated is not None
        if not existed and any(operation["action"] != "ADD" for operation in validated):
            error = f"Cannot edit {self.FILE_LABEL} because no existing store was found."
            return self._result("error", False, error, [error], error_code="missing_store")

        if existed:
            working = current or ""
            preserve_eof = working.endswith(("\r\n", "\n", "\r"))
            newline = self._dominant_newline(working)
        else:
            working = self.DEFAULT_TEMPLATE.strip() + "\n"
            preserve_eof = True
            newline = "\n"

        changed = False
        for operation in validated:
            working, status, error = self._apply_one(working, operation, newline, preserve_eof)
            if status == "error":
                return self._result(
                    "error", False, f"Error updating {self.FILE_LABEL}.",
                    [error or "Unknown operation error."], error_code="operation_rejected",
                )
            changed = changed or status == "applied"

        if not changed:
            return self._result("no_op", False, f"No changes applied to {self.FILE_LABEL}.")
        safe, status = screen_prompt_content(working)
        if not safe:
            return self._result(
                "error", False, f"Error updating {self.FILE_LABEL}.", [status],
                error_code="unsafe_candidate",
            )
        if len(working) > self.MAX_CHAR_BUDGET:
            current_content = current if existed else self.DEFAULT_TEMPLATE.strip() + "\n"
            current_content = current_content or ""
            capacity = {
                "current_chars": len(current_content),
                "candidate_chars": len(working),
                "limit": self.MAX_CHAR_BUDGET,
                "remaining_chars": max(0, self.MAX_CHAR_BUDGET - len(current_content)),
                "percent_used": (
                    len(current_content) * 100 / self.MAX_CHAR_BUDGET
                    if self.MAX_CHAR_BUDGET > 0 else 100.0
                ),
                "content": current_content,
                "guidance": (
                    "Shorten or losslessly consolidate affected entries so the whole final document fits; "
                    "do not delete unrelated valid facts merely to free space."
                ),
            }
            error = (
                f"{self.FILE_LABEL} capacity overflow: candidate is {len(working)} characters, "
                f"limit is {self.MAX_CHAR_BUDGET}, current document is {len(current_content)}; "
                "shorten or losslessly consolidate affected entries. Existing content was not modified."
            )
            return self._result(
                "error", False, f"Error: {error}", [error],
                error_code="capacity_overflow", capacity=capacity, operations=validated,
            )
        publish_error = self._atomic_publish(working)
        if publish_error:
            error = f"Error saving {self.FILE_LABEL}: {publish_error}"
            return self._result("error", False, error, [error], error_code="publish_failed")
        return self._result("applied", True, f"Successfully updated {self.FILE_LABEL} ({len(working)} chars).")

    def _legacy_update(
        self,
        category: Optional[str],
        value: Optional[str],
        action: str = "ADD",
        old_text: Optional[str] = None,
        operations: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        if operations is not None:
            if category is not None or value is not None or old_text is not None or str(action).upper() != "ADD":
                return "Error: operations cannot be combined with single-operation fields."
            result = self.apply_operations(operations)
        else:
            operation: Dict[str, Any] = {"action": action, "category": category}
            if value is not None:
                operation["value"] = value
            if old_text is not None:
                operation["old_text"] = old_text
            result = self.apply_operations([operation])
        if result["status"] == "no_op" and operations is None and str(action).upper() == "ADD":
            return f"{self.ITEM_LABEL} already recorded in {self.FILE_LABEL}."
        if result["status"] == "error":
            return "Error: " + "; ".join(result["errors"])
        return result["message"]

    def _direct_save(self, content: str) -> str:
        if not isinstance(content, str):
            return f"Error saving {self.FILE_LABEL}: content must be a string."
        content = content.strip()
        safe, status = screen_prompt_content(content)
        if not safe:
            return status
        if len(content) > self.MAX_CHAR_BUDGET:
            return (
                f"Error: {self.FILE_LABEL} update exceeds the {self.MAX_CHAR_BUDGET} character "
                "budget; existing content was not modified."
            )
        candidate = content + "\n"
        publish_error = self._atomic_publish(candidate)
        if publish_error:
            return f"Error saving {self.FILE_LABEL}: {publish_error}"
        return f"Successfully updated {self.FILE_LABEL} ({len(content)} chars)."


class UserProfileManager(_MarkdownMemoryManager):
    """
    Manages loading, saving, formatting, and bounded editing of USER.md.
    Injected into the system prompt as a high-signal operator profile snapshot.
    """

    MAX_CHAR_BUDGET = 2000  # Hermes standard character limit for USER.md (~400-500 tokens)
    FILE_LABEL = "USER.md"
    ITEM_LABEL = "Preference"

    DEFAULT_TEMPLATE = """# User Profile & Preferences (USER.md)

## Role & Background
- Software Engineer / AI Developer working with local LLMs and autonomous agents.

## Communication Preferences
- Direct, concise, technical, and high-signal responses.
- Show terminal command outcomes and code diffs clearly.

## Technical Preferences & Conventions
- Environment: Windows (PowerShell / Command Prompt)
- Python Version: Modern Python 3.10+
- Models: Local Qwen-32B via vLLM / Ollama (OpenAI compatible endpoint)

## Operational Constraints & Safety
- Interactively confirm all terminal commands before execution.
- Maintain test coverage and verify changes before marking tasks complete.
"""

    def __init__(self, storage_dir: Optional[str] = None, allow_root_fallback: bool = True):
        self.storage_dir = os.path.abspath(storage_dir or os.path.join(os.getcwd(), ".agent_memories"))
        self.file_path = os.path.join(self.storage_dir, "USER.md")
        self.allow_root_fallback = allow_root_fallback

    def _ensure_file_exists(self):
        """Create default USER.md if it doesn't already exist."""
        if self._resolve_existing_file():
            return
        self._atomic_publish(self.DEFAULT_TEMPLATE.strip() + "\n")

    def load_profile(self) -> str:
        """Load raw USER.md content."""
        if not os.path.exists(self.file_path):
            root_user_md = os.path.join(os.getcwd(), "USER.md")
            if self.allow_root_fallback and os.path.isfile(root_user_md):
                self.file_path = root_user_md
            else:
                return self.DEFAULT_TEMPLATE.strip()

        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            safe, status = screen_prompt_content(content)
            if not safe:
                return status
            return content if content else self.DEFAULT_TEMPLATE.strip()
        except Exception as e:
            return f"Error loading USER.md: {str(e)}"

    def _load_profile_for_update(self) -> Tuple[Optional[str], Optional[str]]:
        """Read existing profile content without substituting display/status text."""
        content, exists, error = self._read_for_update()
        if error:
            return None, error
        return (content if exists else self.DEFAULT_TEMPLATE.strip() + "\n"), None

    def save_profile(self, content: str) -> str:
        """Save updated content to USER.md with character budget enforcement."""
        return self._direct_save(content)

    def update_preference(
        self,
        category: Optional[str] = None,
        note: Optional[str] = None,
        action: str = "ADD",
        old_text: Optional[str] = None,
        operations: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Apply one legacy-compatible preference edit or an atomic per-store batch."""
        return self._legacy_update(category, note, action, old_text, operations)

    def format_system_prompt_block(self) -> str:
        """Format USER.md into the standard Hermes frozen snapshot XML block."""
        profile = self.load_profile()
        return f"<user_profile>\n{profile}\n</user_profile>"


class ProjectMemoryManager(_MarkdownMemoryManager):
    """
    Manages loading, saving, formatting, and bounded editing of MEMORY.md.
    Stores durable facts about the codebase architecture, environment, and technical decisions.
    """

    MAX_CHAR_BUDGET = 2500  # Character budget for project facts (~500-600 tokens)
    FILE_LABEL = "MEMORY.md"
    ITEM_LABEL = "Fact"

    DEFAULT_TEMPLATE = """# Project Memory & Architecture Facts (MEMORY.md)

## Codebase Architecture & Tech Stack
- Primary Language: Python
- Key Modules: Terminal session engine, Context compaction summarizer, Hermes skill repository, SQLite trajectory logger.

## Environment & Configuration
- Workspace: Local project repository
- LLM Protocol: OpenAI JSON Tool Calling & Hermes XML ChatML

## Key Patterns & Conventions
- Unit Tests: Standard Python unittest suite in test_tools.py
- Skill Storage: Native Markdown (.md) with YAML frontmatter in .agent_skills/

## Known Gotchas & Resolved Issues
- Local LLM Token Limits: Always use context compaction threshold to stay safely within context limits.
- Process State: Working directory (cwd) persists across tool calls via stateful terminal session.
"""

    def __init__(self, storage_dir: Optional[str] = None, allow_root_fallback: bool = True):
        self.storage_dir = os.path.abspath(storage_dir or os.path.join(os.getcwd(), ".agent_memories"))
        self.file_path = os.path.join(self.storage_dir, "MEMORY.md")
        self.allow_root_fallback = allow_root_fallback

    def _ensure_file_exists(self):
        """Create default MEMORY.md if it doesn't already exist."""
        if self._resolve_existing_file():
            return
        self._atomic_publish(self.DEFAULT_TEMPLATE.strip() + "\n")

    def load_memory(self) -> str:
        """Load raw MEMORY.md content."""
        if not os.path.exists(self.file_path):
            root_mem_md = os.path.join(os.getcwd(), "MEMORY.md")
            if self.allow_root_fallback and os.path.isfile(root_mem_md):
                self.file_path = root_mem_md
            else:
                return self.DEFAULT_TEMPLATE.strip()

        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            safe, status = screen_prompt_content(content)
            if not safe:
                return status
            return content if content else self.DEFAULT_TEMPLATE.strip()
        except Exception as e:
            return f"Error loading MEMORY.md: {str(e)}"

    def _load_memory_for_update(self) -> Tuple[Optional[str], Optional[str]]:
        """Read existing memory content without substituting display/status text."""
        content, exists, error = self._read_for_update()
        if error:
            return None, error
        return (content if exists else self.DEFAULT_TEMPLATE.strip() + "\n"), None

    def save_memory(self, content: str) -> str:
        """Save updated content to MEMORY.md with character budget enforcement."""
        return self._direct_save(content)

    def update_fact(
        self,
        category: Optional[str] = None,
        fact: Optional[str] = None,
        action: str = "ADD",
        old_text: Optional[str] = None,
        operations: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Apply one legacy-compatible fact edit or an atomic per-store batch."""
        return self._legacy_update(category, fact, action, old_text, operations)

    def format_system_prompt_block(self) -> str:
        """Format MEMORY.md into the standard Hermes frozen snapshot XML block."""
        mem = self.load_memory()
        return f"<project_memory>\n{mem}\n</project_memory>"


class AutoMemoryExtractor:
    """
    Autonomous Memory Reflection & Evolution Engine.
    Evaluates conversation turns to automatically extract:
    1. Operator preferences, workflow constraints, or corrections -> updates USER.md
    2. Project architecture, environment facts, or resolved bugs -> updates MEMORY.md
    """

    REFLECTION_PROMPT = """You are an autonomous Memory Curator for an AI coding assistant.
Review the conversation turns to determine if any durable user preferences, workflow rules, corrections, or project architecture facts were communicated.

=== CURRENT USER PROFILE (USER.md) ===
{current_user_profile}
======================================

=== CURRENT PROJECT MEMORY (MEMORY.md) ===
{current_project_memory}
==========================================

### Reconciliation Rules:
1. **OPERATOR PREFERENCES & CORRECTIONS (USER.md)**:
   - Extract durable facts about the operator: their role, stated workflow preferences (e.g. "prefers pytest", "use powershell", "keep responses concise", "always format diffs"), tool choices, or direct corrections (e.g. "don't use pip, use uv", "never edit test files directly").
   - Do NOT save transient/one-off task requests (e.g. "fix bug on line 12", "create file foo.py").
   - Category options: "Communication Preferences", "Technical Preferences & Conventions", "Operational Constraints & Safety", "Role & Background".

2. **PROJECT ARCHITECTURE & FACTS (MEMORY.md)**:
   - Extract durable facts about the codebase, environment, tech stack, database ports, server configs, or permanent bug resolutions established in this session.
   - Do NOT save business/query result rows, temporary results, one-off filters, or speculation.
   - Category options: "Codebase Architecture & Tech Stack", "Environment & Configuration", "Key Patterns & Conventions", "Known Gotchas & Resolved Issues".

3. **COMPARE MEANING AGAINST ALL EXISTING ENTRIES**:
   - Only user-stated corrections/preferences or verified facts in the visible conversation authorize an edit.
   - A paraphrase of an existing entry is a no-op: return null for that store.
   - A visible, explicit correction or refinement is REPLACE and must name the exact existing bullet as old_text.
   - A visible, explicit retraction is REMOVE and must name the exact existing bullet as old_text.
   - A genuinely new durable preference or verified project fact is ADD.
   - Never remove an entry merely because it is unmentioned, looks stale, or would make the file tidier.
   - Consolidate only entries directly affected by an explicit visible refinement or supersession. Use one REPLACE plus bounded REMOVE operations, and preserve every relevant fact in the canonical replacement. Distinct related rules remain distinct.
   - If nothing safe and durable changed, return null. Do not request a retry or another semantic review.

4. **CAPACITY AND EDIT SAFETY**:
   - Each store snapshot includes exact current characters, limit, remaining characters, and percent used.
   - When proposing a save to a store above 80%, prefer compact lossless wording and consolidate only genuine duplicates or explicitly superseded facts.
   - Never delete unrelated valid facts merely to create space. The host cannot guarantee semantic preservation and will reject unsafe or oversized final documents.
   - MAX_OPERATIONS is 8 per store. Each category, value, and old_text must be one logical line. REPLACE/REMOVE old_text must be one unique normalized whole bullet in the exact existing level-two (##) category. ADD may create a category. Do not create duplicates or edit bullets with continuation text. Use only fields valid for the action.

Respond ONLY with a JSON object in this format:
{
  "user_profile_update": [
    {"action": "ADD" | "REPLACE" | "REMOVE", "category": "exact USER.md category", "value": "new bullet for ADD/REPLACE", "old_text": "exact old bullet for REPLACE/REMOVE"}
  ] | null,
  "project_memory_update": [
    {"action": "ADD" | "REPLACE" | "REMOVE", "category": "exact MEMORY.md category", "value": "new bullet for ADD/REPLACE", "old_text": "exact old bullet for REPLACE/REMOVE"}
  ] | null
}

Omit value for REMOVE and omit old_text for ADD. A per-store list is one atomic batch intended only for a bounded correction/consolidation. Legacy single objects using category+preference or category+fact are also accepted by the host as ADDs.
"""

    def __init__(
        self,
        user_manager: Optional[UserProfileManager] = None,
        project_manager: Optional[ProjectMemoryManager] = None
    ):
        self.user_manager = user_manager or user_profile_manager
        self.project_manager = project_manager or project_memory_manager

    def extract_and_update(
        self,
        client: Any,
        model: str,
        messages: List[Dict[str, Any]],
        task_summary: str = ""
    ) -> Dict[str, Any]:
        """
        Runs an evaluation pass over conversation turns to extract durable preferences and facts.
        Returns: Dict containing any applied updates.
        """
        empty_result = {
            "user_updated": None,
            "project_updated": None,
            "errors": [],
            "attempts": 0,
            "store_attempts": {"user": 0, "project": 0},
            "states": {"user": "no_op", "project": "no_op"},
            "exhausted": False,
        }
        if len(messages) < 2:
            return empty_result

        # Build transcript excerpt
        transcript_parts = [f"Session Task / Context: {task_summary}\n"]
        for msg in messages[-8:]:  # Focus on the most recent turns and user messages
            role = str(msg.get("role", "")).upper()
            content = str(msg.get("content") or "")
            if len(content) > 600:
                content = content[:300] + "\n...[TRUNCATED]...\n" + content[-200:]
            transcript_parts.append(f"[{role}]:\n{content}")

        transcript_text = "\n\n".join(transcript_parts)
        stores = {
            "user": {
                "response_key": "user_profile_update",
                "legacy_value_key": "preference",
                "manager": self.user_manager,
                "result_key": "user_updated",
                "label": "USER.md",
            },
            "project": {
                "response_key": "project_memory_update",
                "legacy_value_key": "fact",
                "manager": self.project_manager,
                "result_key": "project_updated",
                "label": "MEMORY.md",
            },
        }
        result = {
            **empty_result,
            "store_attempts": dict(empty_result["store_attempts"]),
            "states": {"user": "pending", "project": "pending"},
        }
        snapshots: Dict[str, Dict[str, Any]] = {}
        overflows: Dict[str, Dict[str, Any]] = {}
        terminal_codes: Dict[str, str] = {}

        def terminal(store_name: str, code: str, message: str) -> None:
            if result["states"][store_name] == "terminal_error":
                return
            result["states"][store_name] = "terminal_error"
            terminal_codes[store_name] = code
            result["errors"].append(f"{stores[store_name]['label']}: {message}")
            overflows.pop(store_name, None)

        def snapshot_text(store_name: str) -> str:
            snapshot = snapshots.get(store_name, {})
            label = stores[store_name]["label"]
            if not snapshot.get("available"):
                code = terminal_codes.get(store_name, snapshot.get("error_code", "snapshot_unavailable"))
                return f"[{label} unavailable: {code}]"
            counters = {
                "current_chars": snapshot["current_chars"],
                "limit": snapshot["limit"],
                "remaining_chars": snapshot["remaining_chars"],
                "percent_used": snapshot["percent_used"],
            }
            return f"CAPACITY {json.dumps(counters, sort_keys=True)}\n{snapshot['content']}"

        for store_name, store in stores.items():
            try:
                snapshot = store["manager"].capacity_snapshot()
                required = {"content", "current_chars", "limit", "remaining_chars", "percent_used"}
                if not isinstance(snapshot, dict) or (
                    snapshot.get("available") and not required.issubset(snapshot)
                ):
                    raise ValueError("malformed capacity snapshot")
            except Exception as exc:
                snapshot = {
                    "available": False,
                    "error_code": "snapshot_unavailable",
                }
                terminal(
                    store_name,
                    "snapshot_unavailable",
                    f"snapshot_unavailable: {exc}; learning was not saved.",
                )
            snapshots[store_name] = snapshot
            if not snapshot.get("available"):
                terminal(
                    store_name,
                    "snapshot_unavailable",
                    "snapshot_unavailable; store could not be read safely, so reflection was not attempted.",
                )

        if all(result["states"][name] == "terminal_error" for name in stores):
            return result

        constraints = (
            "MAX_OPERATIONS=8 per store. Each category/value/old_text is one logical line. "
            "REPLACE/REMOVE requires an exact unique normalized whole bullet in the real ## category; "
            "ADD may create a category; use action-valid fields only; do not create duplicates or edit "
            "a bullet with continuation text. Shorten or merge losslessly when needed, but never remove "
            "unrelated valid facts merely for space."
        )

        for attempt_index in range(4):
            is_recovery = attempt_index > 0
            if is_recovery:
                pending_names = [
                    name for name in stores
                    if result["states"][name] == "pending" and name in overflows
                ]
                if not pending_names:
                    break
                # Re-read every store for a genuinely current retry prompt. A finalized
                # store remains finalized even if its defensive read later fails.
                for store_name in stores:
                    try:
                        snapshot = stores[store_name]["manager"].capacity_snapshot()
                        required = {"content", "current_chars", "limit", "remaining_chars", "percent_used"}
                        if not isinstance(snapshot, dict) or (
                            snapshot.get("available") and not required.issubset(snapshot)
                        ):
                            raise ValueError("malformed capacity snapshot")
                    except Exception as exc:
                        snapshot = {
                            "available": False,
                            "error_code": "snapshot_unavailable",
                        }
                        if store_name in pending_names:
                            terminal(
                                store_name,
                                "snapshot_unavailable",
                                f"snapshot_unavailable during capacity recovery: {exc}; learning was not saved.",
                            )
                    snapshots[store_name] = snapshot
                    if store_name in pending_names and not snapshot.get("available"):
                        terminal(
                            store_name,
                            "snapshot_unavailable",
                            "snapshot_unavailable during capacity recovery; learning was not saved.",
                        )
                pending_names = [
                    name for name in pending_names
                    if result["states"][name] == "pending"
                ]
                if not pending_names:
                    break

            prompt = (
                self.REFLECTION_PROMPT
                .replace("{current_user_profile}", snapshot_text("user"))
                .replace("{current_project_memory}", snapshot_text("project"))
            )
            user_input_parts = [
                f"<CONVERSATION_TURNS>\n{transcript_text}\n</CONVERSATION_TURNS>"
            ]
            if is_recovery:
                recovery_items = []
                for store_name in stores:
                    if result["states"][store_name] == "pending" and store_name in overflows:
                        overflow = overflows[store_name]
                        capacity = overflow.get("capacity", {})
                        recovery_items.append({
                            "store": stores[store_name]["label"],
                            "response_key": stores[store_name]["response_key"],
                            "error_code": "capacity_overflow",
                            "current_chars": capacity.get("current_chars"),
                            "candidate_chars": capacity.get("candidate_chars"),
                            "limit": capacity.get("limit"),
                            "remaining_chars": capacity.get("remaining_chars"),
                            "operations": overflow.get("operations", []),
                        })
                sanitized_terminal = [
                    {"store": stores[name]["label"], "error_code": code}
                    for name, code in terminal_codes.items()
                ]
                user_input_parts.append(
                    "<CAPACITY_RECOVERY>\n"
                    "Only the overflow-pending response keys below may be changed. Return null for every "
                    "other store. The earlier candidate was not saved. Produce a complete replacement plan "
                    "that fits the current snapshot; do not repeat an oversized plan.\n"
                    f"{constraints}\n"
                    f"overflow_pending={json.dumps(recovery_items, sort_keys=True)}\n"
                    f"terminal_siblings={json.dumps(sanitized_terminal, sort_keys=True)}\n"
                    "</CAPACITY_RECOVERY>"
                )

            result["attempts"] += 1
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": "\n\n".join(user_input_parts)},
                    ],
                    temperature=0.1,
                )
            except Exception as exc:
                active = [
                    name for name in stores if result["states"][name] == "pending"
                ]
                for store_name in active:
                    suffix = "; learning was not saved." if store_name in overflows else "."
                    terminal(
                        store_name,
                        "provider_error",
                        f"memory reflection provider error: {exc}{suffix}",
                    )
                break

            try:
                content = response.choices[0].message.content
            except Exception:
                content = None
            if not isinstance(content, str):
                for store_name in stores:
                    if result["states"][store_name] == "pending":
                        terminal(
                            store_name,
                            "invalid_response",
                            "reflection returned a malformed provider envelope; learning was not saved.",
                        )
                break

            try:
                json_match = re.search(r"\{.*\}", content, re.DOTALL)
            except Exception as exc:
                for store_name in stores:
                    if result["states"][store_name] == "pending":
                        terminal(
                            store_name,
                            "invalid_response",
                            f"reflection response could not be parsed: {exc}; learning was not saved.",
                        )
                break
            if not json_match:
                for store_name in stores:
                    if result["states"][store_name] == "pending":
                        suffix = "; learning was not saved."
                        terminal(store_name, "invalid_response", f"reflection returned no JSON object{suffix}")
                break
            try:
                data = json.loads(json_match.group(0))
            except Exception:
                for store_name in stores:
                    if result["states"][store_name] == "pending":
                        suffix = "; learning was not saved."
                        terminal(store_name, "invalid_response", f"reflection returned invalid JSON{suffix}")
                break
            if not isinstance(data, dict):
                for store_name in stores:
                    if result["states"][store_name] == "pending":
                        terminal(store_name, "invalid_response", "reflection JSON must be an object.")
                break

            for store_name, store in stores.items():
                if result["states"][store_name] != "pending":
                    continue
                if is_recovery and store_name not in overflows:
                    continue
                response_key = store["response_key"]
                if response_key not in data:
                    suffix = " Original learning was not saved." if store_name in overflows else ""
                    terminal(
                        store_name,
                        "omitted_key",
                        f"reflection result omitted {response_key}.{suffix}",
                    )
                    continue
                update = data[response_key]
                if update is None:
                    if store_name in overflows:
                        terminal(
                            store_name,
                            "recovery_abandoned",
                            "capacity recovery returned null; original learning was not saved.",
                        )
                    else:
                        result["states"][store_name] = "no_op"
                    continue
                legacy_value_key = store["legacy_value_key"]
                if isinstance(update, dict):
                    allowed = {"category", legacy_value_key}
                    if set(update) - allowed or not update.get("category") or not update.get(legacy_value_key):
                        terminal(store_name, "malformed_update", "malformed legacy update object; learning was not saved.")
                        continue
                    operations = [{
                        "action": "ADD",
                        "category": update["category"],
                        "value": update[legacy_value_key],
                    }]
                elif isinstance(update, list):
                    operations = update
                else:
                    terminal(
                        store_name,
                        "malformed_update",
                        "update must be null, a legacy object, or an operations array; learning was not saved.",
                    )
                    continue

                result["store_attempts"][store_name] += 1
                try:
                    write_result = store["manager"].apply_operations(operations)
                    if not isinstance(write_result, dict) or not isinstance(write_result.get("status"), str):
                        raise ValueError("malformed writer result")
                except Exception as exc:
                    terminal(
                        store_name,
                        "write_failed",
                        f"write_failed: writer error: {exc}; proposed update was not saved.",
                    )
                    continue
                if write_result["status"] == "applied":
                    result[store["result_key"]] = write_result.get("message") or "Memory update applied."
                    result["states"][store_name] = "applied"
                    overflows.pop(store_name, None)
                elif write_result["status"] == "no_op":
                    if store_name in overflows:
                        terminal(
                            store_name,
                            "recovery_no_change",
                            "capacity recovery made no change; original learning was not saved.",
                        )
                    else:
                        result["states"][store_name] = "no_op"
                elif write_result.get("error_code") == "capacity_overflow":
                    overflows[store_name] = write_result
                else:
                    code = write_result.get("error_code", "write_failed")
                    terminal(
                        store_name,
                        code,
                        f"{code}; proposed update was not saved.",
                    )

            pending_overflows = [
                name for name in stores
                if result["states"][name] == "pending" and name in overflows
            ]
            if not pending_overflows:
                break
            if result["attempts"] >= 4:
                result["exhausted"] = True
                for store_name in pending_overflows:
                    terminal(
                        store_name,
                        "capacity_recovery_exhausted",
                        "capacity recovery exhausted after 4 provider calls; learning was not saved.",
                    )
                break

        return result


# Shared singleton instances
user_profile_manager = UserProfileManager()
project_memory_manager = ProjectMemoryManager()
auto_memory_extractor = AutoMemoryExtractor(
    user_manager=user_profile_manager,
    project_manager=project_memory_manager
)
