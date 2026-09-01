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


def _normalized_bullet(text: str) -> str:
    """Normalize one Markdown bullet for exact, case-insensitive comparison."""
    value = str(text or "").strip()
    value = re.sub(r"^[-*+]\s*", "", value)
    return re.sub(r"\s+", " ", value).strip().casefold()


def _single_line_input_error(
    category: str,
    new_text: str,
    action: str,
    old_text: Optional[str],
    target_name: str,
) -> Optional[str]:
    """Reject operation fields that could alter Markdown structure."""
    operation = str(action or "ADD").strip().upper()
    fields = [("category", category)]
    if operation in {"ADD", "REPLACE"}:
        fields.append((target_name, new_text))
    if operation in {"REPLACE", "REMOVE"}:
        fields.append(("old_text", old_text))
    for field_name, value in fields:
        if "\r" in str(value or "") or "\n" in str(value or ""):
            return (
                f"Error: {field_name} must be single-line (no CR or LF); "
                "existing content was not modified."
            )
    return None


def _apply_bullet_operation(
    current: str,
    category: str,
    new_text: str,
    action: str,
    old_text: Optional[str],
    target_name: str,
) -> Tuple[Optional[str], Optional[str], bool]:
    """Build an updated document without writing it."""
    operation = str(action or "ADD").strip().upper()
    if operation not in {"ADD", "REPLACE", "REMOVE"}:
        return None, "Error: action must be ADD, REPLACE, or REMOVE; existing content was not modified.", False

    category = str(category or "").strip()
    new_text = str(new_text or "").strip()
    old_text = None if old_text is None else str(old_text).strip()
    if not category:
        return None, "Error: category must be non-empty; existing content was not modified.", False
    if operation in {"ADD", "REPLACE"} and not _normalized_bullet(new_text):
        return None, f"Error: {operation} requires a non-empty new {target_name}; existing content was not modified.", False
    if operation in {"REPLACE", "REMOVE"} and not _normalized_bullet(old_text or ""):
        return None, f"Error: {operation} requires old_text; existing content was not modified.", False

    category_pattern = re.compile(
        rf"(?im)^##\s+{re.escape(category)}\s*$.*?(?=^##\s+|\Z)", re.DOTALL
    )
    sections = list(category_pattern.finditer(current))
    clean_new = f"- {_normalized_display_bullet(new_text)}"

    if operation == "ADD":
        if len(sections) > 1:
            return None, f"Error: multiple category headers match '{category}'; existing content was not modified.", False
        if sections:
            section = sections[0]
            existing = [
                _normalized_bullet(match.group(0))
                for match in re.finditer(r"(?m)^[ \t]*[-*+]\s+.*$", section.group(0))
            ]
            if _normalized_bullet(new_text) in existing:
                return current, None, False
            section_text = section.group(0)
            trailing_newlines = re.search(r"\n+$", section_text)
            separator = trailing_newlines.group(0) if trailing_newlines else ""
            body = section_text[:-len(separator)] if separator else section_text
            if section.end() < len(current) and len(separator) < 2:
                separator = "\n\n"
            elif section.end() == len(current):
                separator = "\n"
            updated_section = body.rstrip() + "\n" + clean_new + separator
            return current[:section.start()] + updated_section + current[section.end():], None, True
        return f"{current.rstrip()}\n\n## {category}\n{clean_new}", None, True

    wanted = _normalized_bullet(old_text or "")
    matches = []
    for section in sections:
        for bullet in re.finditer(r"(?m)^[ \t]*[-*+]\s+.*(?:\n|$)", section.group(0)):
            if _normalized_bullet(bullet.group(0)) == wanted:
                matches.append((section.start() + bullet.start(), section.start() + bullet.end()))
    if not matches:
        return None, f"Error: no matching bullet for old_text in category '{category}'; existing content was not modified.", False
    if len(matches) != 1:
        return None, f"Error: multiple matching bullets for old_text in category '{category}'; existing content was not modified.", False

    start, end = matches[0]
    if operation == "REPLACE":
        normalized_new = _normalized_bullet(new_text)
        for section in sections:
            for bullet in re.finditer(r"(?m)^[ \t]*[-*+]\s+.*(?:\n|$)", section.group(0)):
                bullet_start = section.start() + bullet.start()
                bullet_end = section.start() + bullet.end()
                if (bullet_start, bullet_end) != (start, end) and _normalized_bullet(bullet.group(0)) == normalized_new:
                    return None, (
                        f"Error: the replacement {target_name} already exists elsewhere in "
                        f"category '{category}'; existing content was not modified."
                    ), False
    replacement = clean_new + "\n" if operation == "REPLACE" else ""
    return current[:start] + replacement + current[end:], None, True


def _normalized_display_bullet(text: str) -> str:
    value = str(text or "").strip()
    return re.sub(r"^[-*+]\s*", "", value).strip()


def _atomic_save(file_path: str, content: str, label: str) -> Optional[str]:
    """Publish UTF-8 text atomically from a same-directory temporary file."""
    directory = os.path.dirname(file_path) or "."
    temp_path = None
    try:
        newline = "\n"
        if os.path.exists(file_path):
            with open(file_path, "rb") as existing:
                if b"\r\n" in existing.read():
                    newline = "\r\n"
        content = content.replace("\r\n", "\n").replace("\r", "\n").replace("\n", newline)
        os.makedirs(directory, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=directory,
            prefix=f".{os.path.basename(file_path)}.", suffix=".tmp", delete=False,
        ) as handle:
            temp_path = handle.name
            handle.write(content + newline)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, file_path)
        temp_path = None
        return None
    except Exception as exc:
        return f"Error saving {label}; existing content was not modified: {exc}"
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


class UserProfileManager:
    """
    Manages loading, saving, formatting, and bounded editing of USER.md.
    Injected into the system prompt as a high-signal operator profile snapshot.
    """

    MAX_CHAR_BUDGET = 2000  # Hermes standard character limit for USER.md (~400-500 tokens)

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

    def __init__(self, storage_dir: Optional[str] = None):
        self.storage_dir = os.path.abspath(storage_dir or os.path.join(os.getcwd(), ".agent_memories"))
        self.file_path = os.path.join(self.storage_dir, "USER.md")

    def _ensure_file_exists(self):
        """Create default USER.md if it doesn't already exist."""
        root_user_md = os.path.join(os.getcwd(), "USER.md")
        if os.path.isfile(root_user_md) and not os.path.isfile(self.file_path):
            self.file_path = root_user_md
            return

        if not os.path.exists(self.file_path):
            try:
                os.makedirs(self.storage_dir, exist_ok=True)
                with open(self.file_path, "w", encoding="utf-8") as f:
                    f.write(self.DEFAULT_TEMPLATE.strip() + "\n")
            except Exception:
                pass

    def load_profile(self) -> str:
        """Load raw USER.md content."""
        if not os.path.exists(self.file_path):
            root_user_md = os.path.join(os.getcwd(), "USER.md")
            if os.path.isfile(root_user_md):
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
        if not os.path.exists(self.file_path):
            root_user_md = os.path.join(os.getcwd(), "USER.md")
            if os.path.isfile(root_user_md):
                self.file_path = root_user_md
            else:
                return self.DEFAULT_TEMPLATE.strip(), None
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
        except Exception as e:
            return None, f"Error loading USER.md; existing content was not modified: {str(e)}"
        safe, status = screen_prompt_content(content)
        if not safe:
            return None, status
        return content if content else self.DEFAULT_TEMPLATE.strip(), None

    def save_profile(self, content: str) -> str:
        """Save updated content to USER.md with character budget enforcement."""
        content = content.strip()
        safe, status = screen_prompt_content(content)
        if not safe:
            return status

        if len(content) > self.MAX_CHAR_BUDGET:
            return (
                f"Error: USER.md update exceeds the {self.MAX_CHAR_BUDGET} character "
                "budget; existing content was not modified."
            )

        error = _atomic_save(self.file_path, content, "USER.md")
        return error or f"Successfully updated USER.md ({len(content)} chars)."

    def update_preference(
        self, category: str, note: str, action: str = "ADD", old_text: Optional[str] = None
    ) -> str:
        """Apply an ADD, REPLACE, or REMOVE preference operation to USER.md."""
        line_error = _single_line_input_error(category, note, action, old_text, "preference")
        if line_error:
            return line_error
        safe, status = screen_prompt_content(f"{category}\n{note}\n{action}\n{old_text or ''}")
        if not safe:
            return status
        current, load_error = self._load_profile_for_update()
        if load_error:
            return load_error
        new_content, error, changed = _apply_bullet_operation(
            current, category, note, action, old_text, "preference"
        )
        if error:
            return error
        if not changed:
            return "Preference already recorded in USER.md."
        return self.save_profile(new_content or "")

    def format_system_prompt_block(self) -> str:
        """Format USER.md into the standard Hermes frozen snapshot XML block."""
        profile = self.load_profile()
        return f"<user_profile>\n{profile}\n</user_profile>"


class ProjectMemoryManager:
    """
    Manages loading, saving, formatting, and bounded editing of MEMORY.md.
    Stores durable facts about the codebase architecture, environment, and technical decisions.
    """

    MAX_CHAR_BUDGET = 2500  # Character budget for project facts (~500-600 tokens)

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

    def __init__(self, storage_dir: Optional[str] = None):
        self.storage_dir = os.path.abspath(storage_dir or os.path.join(os.getcwd(), ".agent_memories"))
        self.file_path = os.path.join(self.storage_dir, "MEMORY.md")

    def _ensure_file_exists(self):
        """Create default MEMORY.md if it doesn't already exist."""
        root_mem_md = os.path.join(os.getcwd(), "MEMORY.md")
        if os.path.isfile(root_mem_md) and not os.path.isfile(self.file_path):
            self.file_path = root_mem_md
            return

        if not os.path.exists(self.file_path):
            try:
                os.makedirs(self.storage_dir, exist_ok=True)
                with open(self.file_path, "w", encoding="utf-8") as f:
                    f.write(self.DEFAULT_TEMPLATE.strip() + "\n")
            except Exception:
                pass

    def load_memory(self) -> str:
        """Load raw MEMORY.md content."""
        if not os.path.exists(self.file_path):
            root_mem_md = os.path.join(os.getcwd(), "MEMORY.md")
            if os.path.isfile(root_mem_md):
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
        if not os.path.exists(self.file_path):
            root_mem_md = os.path.join(os.getcwd(), "MEMORY.md")
            if os.path.isfile(root_mem_md):
                self.file_path = root_mem_md
            else:
                return self.DEFAULT_TEMPLATE.strip(), None
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
        except Exception as e:
            return None, f"Error loading MEMORY.md; existing content was not modified: {str(e)}"
        safe, status = screen_prompt_content(content)
        if not safe:
            return None, status
        return content if content else self.DEFAULT_TEMPLATE.strip(), None

    def save_memory(self, content: str) -> str:
        """Save updated content to MEMORY.md with character budget enforcement."""
        content = content.strip()
        safe, status = screen_prompt_content(content)
        if not safe:
            return status

        if len(content) > self.MAX_CHAR_BUDGET:
            return (
                f"Error: MEMORY.md update exceeds the {self.MAX_CHAR_BUDGET} character "
                "budget; existing content was not modified."
            )

        error = _atomic_save(self.file_path, content, "MEMORY.md")
        return error or f"Successfully updated MEMORY.md ({len(content)} chars)."

    def update_fact(
        self, category: str, fact: str, action: str = "ADD", old_text: Optional[str] = None
    ) -> str:
        """Apply an ADD, REPLACE, or REMOVE fact operation to MEMORY.md."""
        line_error = _single_line_input_error(category, fact, action, old_text, "fact")
        if line_error:
            return line_error
        safe, status = screen_prompt_content(f"{category}\n{fact}\n{action}\n{old_text or ''}")
        if not safe:
            return status
        current, load_error = self._load_memory_for_update()
        if load_error:
            return load_error
        new_content, error, changed = _apply_bullet_operation(
            current, category, fact, action, old_text, "fact"
        )
        if error:
            return error
        if not changed:
            return "Fact already recorded in MEMORY.md."
        return self.save_memory(new_content or "")

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

### Extraction Rules:
1. **OPERATOR PREFERENCES & CORRECTIONS (USER.md)**:
   - Extract durable facts about the operator: their role, stated workflow preferences (e.g. "prefers pytest", "use powershell", "keep responses concise", "always format diffs"), tool choices, or direct corrections (e.g. "don't use pip, use uv", "never edit test files directly").
   - Do NOT save transient/one-off task requests (e.g. "fix bug on line 12", "create file foo.py").
   - Category options: "Communication Preferences", "Technical Preferences & Conventions", "Operational Constraints & Safety", "Role & Background".

2. **PROJECT ARCHITECTURE & FACTS (MEMORY.md)**:
   - Extract durable facts about the codebase, environment, tech stack, database ports, server configs, or permanent bug resolutions established in this session.
   - Category options: "Codebase Architecture & Tech Stack", "Environment & Configuration", "Key Patterns & Conventions", "Known Gotchas & Resolved Issues".

3. **DEDUPLICATION**:
   - If a preference or fact is ALREADY present in the current profile or memory, do NOT duplicate it. Set to null.

4. **OPERATIONS**:
   - Use ADD for a new durable bullet.
   - Use REPLACE for a direct correction of one existing bullet and include old_text exactly identifying it.
   - Use REMOVE for a retraction of one existing bullet and include old_text; omit or leave the new value empty.

Respond ONLY with a JSON object in this format:
{
  "user_profile_update": {
    "action": "ADD" | "REPLACE" | "REMOVE",
    "category": "Communication Preferences" | "Technical Preferences & Conventions" | "Operational Constraints & Safety" | "Role & Background",
    "preference": "New concise bullet; required for ADD/REPLACE and empty for REMOVE.",
    "old_text": "Exact existing bullet required for REPLACE/REMOVE; omit for ADD."
  } | null,
  "project_memory_update": {
    "action": "ADD" | "REPLACE" | "REMOVE",
    "category": "Codebase Architecture & Tech Stack" | "Environment & Configuration" | "Key Patterns & Conventions" | "Known Gotchas & Resolved Issues",
    "fact": "New concise bullet; required for ADD/REPLACE and empty for REMOVE.",
    "old_text": "Exact existing bullet required for REPLACE/REMOVE; omit for ADD."
  } | null
}
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
        if len(messages) < 2:
            return {"user_updated": None, "project_updated": None}

        # Build transcript excerpt
        transcript_parts = [f"Session Task / Context: {task_summary}\n"]
        for msg in messages[-8:]:  # Focus on the most recent turns and user messages
            role = str(msg.get("role", "")).upper()
            content = str(msg.get("content") or "")
            if len(content) > 600:
                content = content[:300] + "\n...[TRUNCATED]...\n" + content[-200:]
            transcript_parts.append(f"[{role}]:\n{content}")

        transcript_text = "\n\n".join(transcript_parts)
        prompt = (
            self.REFLECTION_PROMPT
            .replace("{current_user_profile}", self.user_manager.load_profile())
            .replace("{current_project_memory}", self.project_manager.load_memory())
        )

        user_input_block = f"<CONVERSATION_TURNS>\n{transcript_text}\n</CONVERSATION_TURNS>"

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_input_block}
                ],
                temperature=0.1,
            )
            raw_text = response.choices[0].message.content or ""
            
            # Extract JSON block
            json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if not json_match:
                return {"user_updated": None, "project_updated": None, "user_error": None, "project_error": None}

            data = json.loads(json_match.group(0))
            applied = {
                "user_updated": None, "project_updated": None,
                "user_error": None, "project_error": None,
            }

            # 1. Process USER.md update
            user_update = data.get("user_profile_update")
            if user_update and isinstance(user_update, dict):
                cat = user_update.get("category")
                pref = user_update.get("preference", "")
                action = user_update.get("action") or "ADD"
                old_text = user_update.get("old_text")
                if cat:
                    res = self.user_manager.update_preference(cat, pref, action=action, old_text=old_text)
                    if "Successfully updated" in res:
                        applied["user_updated"] = f"[{action.upper()}] [{cat}] {pref or old_text}"
                    elif "already recorded" not in res:
                        applied["user_error"] = res

            # 2. Process MEMORY.md update
            proj_update = data.get("project_memory_update")
            if proj_update and isinstance(proj_update, dict):
                cat = proj_update.get("category")
                fact = proj_update.get("fact", "")
                action = proj_update.get("action") or "ADD"
                old_text = proj_update.get("old_text")
                if cat:
                    res = self.project_manager.update_fact(cat, fact, action=action, old_text=old_text)
                    if "Successfully updated" in res:
                        applied["project_updated"] = f"[{action.upper()}] [{cat}] {fact or old_text}"
                    elif "already recorded" not in res:
                        applied["project_error"] = res

            return applied

        except Exception as exc:
            return {
                "user_updated": None, "project_updated": None,
                "user_error": f"Memory reflection failed: {exc}", "project_error": None,
            }


# Shared singleton instances
user_profile_manager = UserProfileManager()
project_memory_manager = ProjectMemoryManager()
auto_memory_extractor = AutoMemoryExtractor(
    user_manager=user_profile_manager,
    project_manager=project_memory_manager
)
