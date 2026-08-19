"""
Terminal execution engine for the coding agent harness.
Supports persistent working directory state, environment variables,
output truncation for context preservation, and safety checks.
"""

import os
import re
import shlex
import subprocess
from typing import Dict, Optional, Tuple


class TerminalSession:
    """
    Manages stateful terminal command execution.
    Maintains active working directory and environment variables across tool invocations.
    """

    SAFE_COMMAND_PREFIXES = {
        "ls", "dir", "cat", "type", "echo", "pwd", "cd", "git status",
        "git log", "git diff", "git branch", "pytest", "python -m unittest",
        "node -v", "python --version", "cargo --version", "grep", "find"
    }

    DESTRUCTIVE_PATTERNS = [
        r"\brm\s+(-rf|-fr|-r)\b",
        r"\bdel\s+/[sfq]\b",
        r"\brmdir\s+/[sq]\b",
        r"\bformat\b",
        r"\bdrop\s+database\b",
        r"\bgit\s+reset\s+--hard\b",
        r"\bgit\s+clean\s+-fdx?\b",
        r"\bkill\s+-9\b",
        r"\bpkill\b"
    ]

    def __init__(self, cwd: Optional[str] = None, max_output_chars: int = 8000):
        self.cwd = os.path.abspath(cwd or os.getcwd())
        self.env = os.environ.copy()
        self.max_output_chars = max_output_chars

    def is_destructive(self, command: str) -> bool:
        """Check if a command matches potentially destructive patterns."""
        for pattern in self.DESTRUCTIVE_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return True
        return False

    def is_safe(self, command: str) -> bool:
        """Check if a command starts with a known safe read-only prefix."""
        cmd_strip = command.strip().lower()
        return any(cmd_strip.startswith(prefix) for prefix in self.SAFE_COMMAND_PREFIXES)

    def _handle_cd(self, command: str) -> Optional[str]:
        """
        Handle 'cd' command separately to persist working directory across steps.
        Returns a status string if it was purely a cd command, or None to run via shell.
        """
        cmd_parts = command.strip().split()
        if len(cmd_parts) >= 1 and cmd_parts[0].lower() == "cd":
            if len(cmd_parts) == 1:
                target_dir = os.path.expanduser("~")
            else:
                target_dir = command.strip()[3:].strip().strip("\"'")
                target_dir = os.path.expanduser(target_dir)

            new_path = os.path.abspath(os.path.join(self.cwd, target_dir))
            if os.path.isdir(new_path):
                self.cwd = new_path
                return f"[Directory changed to]: {self.cwd}"
            else:
                return f"Error: Directory '{new_path}' does not exist."
        return None

    def execute(self, command: str, timeout: int = 60) -> str:
        """
        Execute command in the current persistent cwd.
        Captures stdout, stderr, exit code, and formats output.
        """
        # 1. Check for standalone 'cd'
        cd_result = self._handle_cd(command)
        if cd_result is not None:
            return cd_result

        # 2. Run command in current cwd
        try:
            process = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.cwd,
                env=self.env
            )

            stdout = process.stdout or ""
            stderr = process.stderr or ""
            exit_code = process.returncode

            # 3. Format and clip output to prevent context blowout
            formatted_output = self._format_output(stdout, stderr, exit_code)
            return formatted_output

        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after {timeout} seconds."
        except Exception as e:
            return f"Error executing command '{command}': {str(e)}"

    def _format_output(self, stdout: str, stderr: str, exit_code: int) -> str:
        """Format and smartly truncate large outputs preserving head and tail."""
        def truncate(text: str, max_len: int) -> str:
            text = text.strip()
            if len(text) <= max_len:
                return text
            head_len = max_len // 2
            tail_len = max_len // 2
            return f"{text[:head_len]}\n\n... [TRUNCATED {len(text) - max_len} CHARACTERS] ...\n\n{text[-tail_len:]}"

        parts = []
        if stdout.strip():
            parts.append(f"[STDOUT]:\n{truncate(stdout, self.max_output_chars // 2)}")
        if stderr.strip():
            parts.append(f"[STDERR]:\n{truncate(stderr, self.max_output_chars // 2)}")
        
        parts.append(f"[EXIT CODE]: {exit_code}")
        parts.append(f"[CWD]: {self.cwd}")

        return "\n\n".join(parts)
