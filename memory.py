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
from typing import Any, Dict, List, Optional, Tuple


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
            return content if content else self.DEFAULT_TEMPLATE.strip()
        except Exception as e:
            return f"Error loading USER.md: {str(e)}"

    def save_profile(self, content: str) -> str:
        """Save updated content to USER.md with character budget enforcement."""
        content = content.strip()

        if len(content) > self.MAX_CHAR_BUDGET:
            return (
                f"Error: USER.md update exceeds the {self.MAX_CHAR_BUDGET} character "
                "budget; existing content was not modified."
            )

        try:
            os.makedirs(os.path.dirname(self.file_path) or ".", exist_ok=True)
            with open(self.file_path, "w", encoding="utf-8") as f:
                f.write(content + "\n")
            return f"Successfully updated USER.md ({len(content)} chars)."
        except Exception as e:
            return f"Error saving USER.md: {str(e)}"

    def update_preference(self, category: str, note: str) -> str:
        """Append or update a specific preference under a category section in USER.md."""
        current = self.load_profile()
        category_header = f"## {category}"

        clean_note = note.strip()
        if not clean_note.startswith("-"):
            clean_note = f"- {clean_note}"

        if category_header.lower() in current.lower():
            pattern = re.compile(rf"({re.escape(category_header)}.*?)(\n##|\Z)", re.IGNORECASE | re.DOTALL)
            match = pattern.search(current)
            if match:
                section_body = match.group(1).rstrip()
                if clean_note.lower() not in section_body.lower():
                    updated_section = f"{section_body}\n{clean_note}\n"
                    new_content = current[:match.start()] + updated_section + current[match.end(1):]
                    return self.save_profile(new_content)
                return "Preference already recorded in USER.md."
        
        new_content = f"{current}\n\n## {category}\n{clean_note}"
        return self.save_profile(new_content)

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
            return content if content else self.DEFAULT_TEMPLATE.strip()
        except Exception as e:
            return f"Error loading MEMORY.md: {str(e)}"

    def save_memory(self, content: str) -> str:
        """Save updated content to MEMORY.md with character budget enforcement."""
        content = content.strip()

        if len(content) > self.MAX_CHAR_BUDGET:
            return (
                f"Error: MEMORY.md update exceeds the {self.MAX_CHAR_BUDGET} character "
                "budget; existing content was not modified."
            )

        try:
            os.makedirs(os.path.dirname(self.file_path) or ".", exist_ok=True)
            with open(self.file_path, "w", encoding="utf-8") as f:
                f.write(content + "\n")
            return f"Successfully updated MEMORY.md ({len(content)} chars)."
        except Exception as e:
            return f"Error saving MEMORY.md: {str(e)}"

    def update_fact(self, category: str, fact: str) -> str:
        """Append or update a specific fact under a category section in MEMORY.md."""
        current = self.load_memory()
        category_header = f"## {category}"

        clean_fact = fact.strip()
        if not clean_fact.startswith("-"):
            clean_fact = f"- {clean_fact}"

        if category_header.lower() in current.lower():
            pattern = re.compile(rf"({re.escape(category_header)}.*?)(\n##|\Z)", re.IGNORECASE | re.DOTALL)
            match = pattern.search(current)
            if match:
                section_body = match.group(1).rstrip()
                if clean_fact.lower() not in section_body.lower():
                    updated_section = f"{section_body}\n{clean_fact}\n"
                    new_content = current[:match.start()] + updated_section + current[match.end(1):]
                    return self.save_memory(new_content)
                return "Fact already recorded in MEMORY.md."
        
        new_content = f"{current}\n\n## {category}\n{clean_fact}"
        return self.save_memory(new_content)

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

Respond ONLY with a JSON object in this format:
{
  "user_profile_update": {
    "category": "Communication Preferences" | "Technical Preferences & Conventions" | "Operational Constraints & Safety" | "Role & Background",
    "preference": "Specific concise bullet point to record."
  } | null,
  "project_memory_update": {
    "category": "Codebase Architecture & Tech Stack" | "Environment & Configuration" | "Key Patterns & Conventions" | "Known Gotchas & Resolved Issues",
    "fact": "Specific concise bullet point to record."
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
                return {"user_updated": None, "project_updated": None}

            data = json.loads(json_match.group(0))
            applied = {"user_updated": None, "project_updated": None}

            # 1. Process USER.md update
            user_update = data.get("user_profile_update")
            if user_update and isinstance(user_update, dict):
                cat = user_update.get("category")
                pref = user_update.get("preference")
                if cat and pref:
                    res = self.user_manager.update_preference(cat, pref)
                    if "Successfully updated" in res:
                        applied["user_updated"] = f"[{cat}] {pref}"

            # 2. Process MEMORY.md update
            proj_update = data.get("project_memory_update")
            if proj_update and isinstance(proj_update, dict):
                cat = proj_update.get("category")
                fact = proj_update.get("fact")
                if cat and fact:
                    res = self.project_manager.update_fact(cat, fact)
                    if "Successfully updated" in res:
                        applied["project_updated"] = f"[{cat}] {fact}"

            return applied

        except Exception:
            return {"user_updated": None, "project_updated": None}


# Shared singleton instances
user_profile_manager = UserProfileManager()
project_memory_manager = ProjectMemoryManager()
auto_memory_extractor = AutoMemoryExtractor(
    user_manager=user_profile_manager,
    project_manager=project_memory_manager
)
