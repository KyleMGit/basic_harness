"""
Coding Agent Tools and Registry.
Includes file manipulation, stateful terminal execution, and persistent skills.
"""

import os
from typing import Any, Callable, Dict, List, Optional
from terminal import TerminalSession
from skills import SkillStore


class ToolRegistry:
    """Tool registry and dispatcher with JSON schema support."""

    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._schemas: List[Dict[str, Any]] = []

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

    def execute(self, name: str, arguments: Dict[str, Any]) -> str:
        if name not in self._tools:
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
# TOOL DEFINITIONS
# ==============================================================================

@registry.register(
    name="run_terminal_command",
    description="Execute a bash/cmd/PowerShell command in the persistent shell session. Maintains working directory across calls.",
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
    return terminal_session.execute(command=command, timeout=timeout)


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
            return f"Error: Target search string not found in '{file_path}'."

        count = content.count(search_content)
        if count > 1:
            return f"Error: Target search string occurs {count} times. Please provide more surrounding context."

        new_content = content.replace(search_content, replace_content, 1)
        with open(resolved_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        return f"Successfully patched '{file_path}' (1 substitution made)."
    except Exception as e:
        return f"Error patching file '{file_path}': {str(e)}"


@registry.register(
    name="list_directory",
    description="List files and directories in a target directory.",
    parameters={
        "type": "object",
        "properties": {
            "directory_path": {
                "type": "string",
                "description": "Path to directory (default: current working directory).",
                "default": "."
            }
        },
        "required": []
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
    description="Load instructions and workflow details for a specific learned skill.",
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "The name of the skill to load."}
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
