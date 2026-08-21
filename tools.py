"""
Coding Agent Tools and Registry.
Includes file manipulation, stateful terminal execution, codebase search,
persistent skills, user profile (USER.md), and project facts (MEMORY.md).
"""

import fnmatch
import os
import re
from typing import Any, Callable, Dict, List, Optional
from terminal import TerminalSession
from skills import SkillStore
from memory import user_profile_manager, project_memory_manager


class ToolRegistry:
    """Tool registry and dispatcher with schema support."""

    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._schemas: List[Dict[str, Any]] = []
        self._write_tools = {
            "run_terminal_command",
            "write_file",
            "patch_file",
            "save_skill",
            "update_user_profile",
            "update_project_memory",
        }

    def register(self, name: str, description: str, parameters: Dict[str, Any]):
        def decorator(func: Callable):
            self._tools[name] = func
            self._schemas.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                }
            })
            return func
        return decorator

    @property
    def schemas(self) -> List[Dict[str, Any]]:
        return self._schemas

    def is_write_tool(self, name: str) -> bool:
        """Return whether a registered tool can mutate external state."""
        return name in self._write_tools

    def execute(self, name: str, arguments: Dict[str, Any], read_only: bool = False) -> str:
        if read_only and self.is_write_tool(name):
            return f"[Testing Mode] Read-only active: Tool '{name}' was not executed."
        if name not in self._tools:
            # Check if the model called a skill directly by its name
            skill_file = skill_store.resolve_skill_file(name)
            if skill_file:
                return skill_store.load_skill(name)
            return f"Error: Tool '{name}' not found."
        try:
            result = self._tools[name](**arguments)
            return str(result)
        except TypeError as te:
            return f"Error: Invalid arguments for '{name}': {str(te)}"
        except Exception as e:
            return f"Error executing tool '{name}': {str(e)}"


# Instantiate shared singletons
registry = ToolRegistry()
terminal_session = TerminalSession()
skill_store = SkillStore()


# ==============================================================================
# TOOL DEFINITIONS: TERMINAL & PROCESS EXECUTION
# ==============================================================================

@registry.register(
    name="run_terminal_command",
    description="Execute a bash/cmd/PowerShell command. Only the working directory (cwd) persists between calls; environment variables, shell functions, and activated virtualenvs do not.",
    parameters={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The command string to execute."
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default: 60).",
                "default": 60
            }
        },
        "required": ["command"]
    }
)
def run_terminal_command(command: str, timeout: int = 60) -> str:
    return terminal_session.execute(command, timeout=timeout)


# ==============================================================================
# TOOL DEFINITIONS: FILE MANIPULATION & CODEBASE SEARCH
# ==============================================================================

@registry.register(
    name="read_file",
    description="Read file contents with optional line range slicing (1-indexed).",
    parameters={
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to the file to read."},
            "start_line": {"type": "integer", "description": "Optional starting line (1-indexed)."},
            "end_line": {"type": "integer", "description": "Optional ending line (1-indexed)."}
        },
        "required": ["file_path"]
    }
)
def read_file(file_path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> str:
    try:
        resolved_path = os.path.abspath(os.path.join(terminal_session.cwd, file_path))
        if not os.path.exists(resolved_path):
            return f"Error: File '{file_path}' does not exist."

        with open(resolved_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if start_line is not None or end_line is not None:
            start = max(0, (start_line or 1) - 1)
            end = end_line or len(lines)
            selected_lines = lines[start:end]
            numbered = [f"{i + start + 1:4d} | {line}" for i, line in enumerate(selected_lines)]
            return "".join(numbered)

        return "".join(lines)
    except Exception as e:
        return f"Error reading file '{file_path}': {str(e)}"


@registry.register(
    name="write_file",
    description="Write full content to a file. Creates missing directories automatically.",
    parameters={
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to the file to write."},
            "content": {"type": "string", "description": "Complete text content to write."}
        },
        "required": ["file_path", "content"]
    }
)
def write_file(file_path: str, content: str) -> str:
    try:
        resolved_path = os.path.abspath(os.path.join(terminal_session.cwd, file_path))
        os.makedirs(os.path.dirname(resolved_path), exist_ok=True)
        with open(resolved_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} bytes to '{file_path}'."
    except Exception as e:
        return f"Error writing file '{file_path}': {str(e)}"


@registry.register(
    name="patch_file",
    description="Perform targeted search and replace on an existing file.",
    parameters={
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to file to patch."},
            "search_content": {"type": "string", "description": "Exact text substring to replace."},
            "replace_content": {"type": "string", "description": "New text to substitute in."}
        },
        "required": ["file_path", "search_content", "replace_content"]
    }
)
def patch_file(file_path: str, search_content: str, replace_content: str) -> str:
    try:
        resolved_path = os.path.abspath(os.path.join(terminal_session.cwd, file_path))
        if not os.path.exists(resolved_path):
            return f"Error: File '{file_path}' not found."

        with open(resolved_path, "r", encoding="utf-8") as f:
            content = f.read()

        if search_content not in content:
            return f"Error: search_content string not found in '{file_path}'."

        occurrences = content.count(search_content)
        if occurrences > 1:
            return f"Error: search_content matched {occurrences} times. Provide more surrounding context."

        new_content = content.replace(search_content, replace_content, 1)
        with open(resolved_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        return f"Successfully patched '{file_path}'."
    except Exception as e:
        return f"Error patching file '{file_path}': {str(e)}"


@registry.register(
    name="list_directory",
    description="List contents of a directory (files, folders, sizes).",
    parameters={
        "type": "object",
        "properties": {
            "directory_path": {
                "type": "string",
                "description": "Path of the directory to list (default: current working directory).",
                "default": "."
            }
        }
    }
)
def list_directory(directory_path: str = ".") -> str:
    try:
        resolved_path = os.path.abspath(os.path.join(terminal_session.cwd, directory_path))
        if not os.path.isdir(resolved_path):
            return f"Error: Directory '{directory_path}' not found."

        entries = os.listdir(resolved_path)
        items = []
        for e in entries:
            full = os.path.join(resolved_path, e)
            kind = "[DIR]" if os.path.isdir(full) else "[FILE]"
            size = f"{os.path.getsize(full)} B" if os.path.isfile(full) else ""
            items.append(f"{kind:<7} {e:<30} {size}")

        return f"Directory: {resolved_path}\n" + "\n".join(items) if items else "[Empty Directory]"
    except Exception as e:
        return f"Error listing directory: {str(e)}"


@registry.register(
    name="grep_search",
    description="Fast regex or literal text search across files in the codebase (ripgrep-style). Returns file paths, line numbers, and matching snippets.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Text substring or regex pattern to search for."},
            "search_path": {"type": "string", "description": "Root directory or file to search within (default: current working directory).", "default": "."},
            "is_regex": {"type": "boolean", "description": "True to treat query as regex, False for literal text match.", "default": False},
            "file_pattern": {"type": "string", "description": "Optional file glob filter (e.g. '*.py', '*.js', '*.md')."},
            "max_results": {"type": "integer", "description": "Maximum matching lines to return (default: 30).", "default": 30}
        },
        "required": ["query"]
    }
)
def grep_search(
    query: str,
    search_path: str = ".",
    is_regex: bool = False,
    file_pattern: Optional[str] = None,
    max_results: int = 30
) -> str:
    resolved_root = os.path.abspath(os.path.join(terminal_session.cwd, search_path))
    if not os.path.exists(resolved_root):
        return f"Error: Path '{search_path}' not found."

    ignored_dirs = {".git", "node_modules", "__pycache__", "venv", ".venv", ".agent_history.db", ".pytest_cache"}
    
    try:
        matcher = re.compile(query, re.IGNORECASE) if is_regex else None
    except Exception as e:
        return f"Error in regex query: {str(e)}"

    matches = []
    
    def search_file(fpath: str):
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                for line_idx, line in enumerate(f, 1):
                    is_match = bool(matcher.search(line)) if is_regex else (query.lower() in line.lower())
                    if is_match:
                        rel_path = os.path.relpath(fpath, terminal_session.cwd)
                        matches.append(f"{rel_path}:{line_idx}: {line.strip()}")
                        if len(matches) >= max_results:
                            return True
        except Exception:
            pass
        return False

    if os.path.isfile(resolved_root):
        search_file(resolved_root)
    else:
        for root, dirs, files in os.walk(resolved_root):
            dirs[:] = [d for d in dirs if d not in ignored_dirs and not d.startswith(".")]
            for filename in files:
                if file_pattern and not fnmatch.fnmatch(filename, file_pattern):
                    continue
                full_path = os.path.join(root, filename)
                if search_file(full_path):
                    break
            if len(matches) >= max_results:
                break

    if not matches:
        return f"No matches found for '{query}' in '{search_path}'."

    output = [f"Found {len(matches)} match(es):"] + matches
    if len(matches) >= max_results:
        output.append(f"... [Capped at {max_results} results]")
    return "\n".join(output)


@registry.register(
    name="find_files_by_pattern",
    description="Find files and directories matching a glob pattern across the workspace (e.g. '*.py', '*test*', 'src/**/*.ts').",
    parameters={
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern (e.g. '*.py', '*agent*', '*.json')."},
            "search_path": {"type": "string", "description": "Root directory to start searching from (default: .).", "default": "."},
            "max_results": {"type": "integer", "description": "Maximum files to return (default: 40).", "default": 40}
        },
        "required": ["pattern"]
    }
)
def find_files_by_pattern(pattern: str, search_path: str = ".", max_results: int = 40) -> str:
    resolved_root = os.path.abspath(os.path.join(terminal_session.cwd, search_path))
    if not os.path.exists(resolved_root):
        return f"Error: Directory '{search_path}' not found."

    ignored_dirs = {".git", "node_modules", "__pycache__", "venv", ".venv", ".pytest_cache"}
    matched_files = []

    for root, dirs, files in os.walk(resolved_root):
        dirs[:] = [d for d in dirs if d not in ignored_dirs and not d.startswith(".")]
        for f in files:
            full_path = os.path.join(root, f)
            relative_to_root = os.path.relpath(full_path, resolved_root).replace(os.sep, "/")
            normalized_pattern = pattern.replace("\\", "/")
            patterns = {normalized_pattern}
            while "**/" in normalized_pattern:
                normalized_pattern = normalized_pattern.replace("**/", "", 1)
                patterns.add(normalized_pattern)
            if any(fnmatch.fnmatchcase(relative_to_root, candidate) for candidate in patterns):
                rel_path = os.path.relpath(full_path, terminal_session.cwd)
                matched_files.append(rel_path)
                if len(matched_files) >= max_results:
                    break
        if len(matched_files) >= max_results:
            break

    if not matched_files:
        return f"No files matching '{pattern}' found in '{search_path}'."

    output = [f"Found {len(matched_files)} file(s) matching '{pattern}':"] + [f"- {p}" for p in matched_files]
    if len(matched_files) >= max_results:
        output.append(f"... [Capped at {max_results} files]")
    return "\n".join(output)


# ==============================================================================
# TOOL DEFINITIONS: SKILLS & PERSISTENT MEMORY (USER.MD / MEMORY.MD)
# ==============================================================================

@registry.register(
    name="save_skill",
    description="Save a newly learned procedure, command workflow, or script to the agent's persistent skill library.",
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Unique concise name for the skill."},
            "description": {"type": "string", "description": "What this skill accomplishes and when to use it."},
            "instructions": {"type": "string", "description": "Step-by-step instructions, code snippets, or best practices."}
        },
        "required": ["name", "description", "instructions"]
    }
)
def save_skill(name: str, description: str, instructions: str) -> str:
    return skill_store.save_skill(name=name, description=description, instructions=instructions)


@registry.register(
    name="load_skill",
    description="Load instructions and workflow details for a specific learned skill from the repository.",
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "The name of the skill to load (e.g. 'git_squash' or 'git_squash.md')."}
        },
        "required": ["name"]
    }
)
def load_skill(name: str) -> str:
    return skill_store.load_skill(name=name)


@registry.register(
    name="list_skills",
    description="List all available learned skills in the skill repository.",
    parameters={"type": "object", "properties": {}}
)
def list_skills() -> str:
    return skill_store.list_skills()


@registry.register(
    name="read_user_profile",
    description="Read the current user profile (USER.md) containing operator background, preferred tools, and conventions.",
    parameters={"type": "object", "properties": {}}
)
def read_user_profile() -> str:
    return user_profile_manager.load_profile()


@registry.register(
    name="update_user_profile",
    description="Update or append persistent preferences, technical background, or operational constraints in USER.md.",
    parameters={
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "description": "Category heading in USER.md (e.g. 'Communication Preferences', 'Technical Preferences & Conventions', 'Operational Constraints & Safety')."
            },
            "preference": {
                "type": "string",
                "description": "The specific user preference, background detail, or rule to record."
            }
        },
        "required": ["category", "preference"]
    }
)
def update_user_profile(category: str, preference: str) -> str:
    return user_profile_manager.update_preference(category=category, note=preference)


@registry.register(
    name="read_project_memory",
    description="Read the current project memory (MEMORY.md) containing architecture facts, environment notes, and codebase conventions.",
    parameters={"type": "object", "properties": {}}
)
def read_project_memory() -> str:
    return project_memory_manager.load_memory()


@registry.register(
    name="update_project_memory",
    description="Update or append persistent architectural facts, environment notes, or resolved bug patterns in MEMORY.md.",
    parameters={
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "description": "Category heading in MEMORY.md (e.g. 'Codebase Architecture & Tech Stack', 'Environment & Configuration', 'Key Patterns & Conventions', 'Known Gotchas & Resolved Issues')."
            },
            "fact": {
                "type": "string",
                "description": "The specific technical fact, architectural note, or convention to record."
            }
        },
        "required": ["category", "fact"]
    }
)
def update_project_memory(category: str, fact: str) -> str:
    return project_memory_manager.update_fact(category=category, fact=fact)
