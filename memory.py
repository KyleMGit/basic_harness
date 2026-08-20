"""
Hermes-inspired Persistent Memory System:
1. USER.md: Operator profile, technical background, and communication preferences.
2. MEMORY.md: Project architecture, environment configuration, and codebase facts.
"""

import os
import re
from typing import Dict, List, Optional


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
        os.makedirs(self.storage_dir, exist_ok=True)
        self.file_path = os.path.join(self.storage_dir, "USER.md")
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """Create default USER.md if it doesn't already exist."""
        root_user_md = os.path.join(os.getcwd(), "USER.md")
        if os.path.isfile(root_user_md) and not os.path.isfile(self.file_path):
            self.file_path = root_user_md
            return

        if not os.path.exists(self.file_path):
            try:
                with open(self.file_path, "w", encoding="utf-8") as f:
                    f.write(self.DEFAULT_TEMPLATE.strip() + "\n")
            except Exception:
                pass

    def load_profile(self) -> str:
        """Load raw USER.md content."""
        if not os.path.exists(self.file_path):
            self._ensure_file_exists()

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
            content = content[:self.MAX_CHAR_BUDGET]
            warning = f"\n[!] Note: Content truncated to fit within the {self.MAX_CHAR_BUDGET} character budget."
        else:
            warning = ""

        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                f.write(content + "\n")
            return f"Successfully updated USER.md ({len(content)} chars).{warning}"
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
                if clean_note not in section_body:
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
        os.makedirs(self.storage_dir, exist_ok=True)
        self.file_path = os.path.join(self.storage_dir, "MEMORY.md")
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """Create default MEMORY.md if it doesn't already exist."""
        root_mem_md = os.path.join(os.getcwd(), "MEMORY.md")
        if os.path.isfile(root_mem_md) and not os.path.isfile(self.file_path):
            self.file_path = root_mem_md
            return

        if not os.path.exists(self.file_path):
            try:
                with open(self.file_path, "w", encoding="utf-8") as f:
                    f.write(self.DEFAULT_TEMPLATE.strip() + "\n")
            except Exception:
                pass

    def load_memory(self) -> str:
        """Load raw MEMORY.md content."""
        if not os.path.exists(self.file_path):
            self._ensure_file_exists()

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
            content = content[:self.MAX_CHAR_BUDGET]
            warning = f"\n[!] Note: Content truncated to fit within the {self.MAX_CHAR_BUDGET} character budget."
        else:
            warning = ""

        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                f.write(content + "\n")
            return f"Successfully updated MEMORY.md ({len(content)} chars).{warning}"
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
                if clean_fact not in section_body:
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


# Shared singleton instances
user_profile_manager = UserProfileManager()
project_memory_manager = ProjectMemoryManager()
