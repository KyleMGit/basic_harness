"""
Terminal execution engine for the coding agent harness.
Supports persistent working directory state, output truncation for context
preservation, and safety checks.
"""

import os
import re
import shlex
import subprocess
from typing import Dict, Iterable, Optional, Tuple


class TerminalSession:
    """
    Manages stateful terminal command execution.
    Maintains the active working directory across tool invocations.
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

    _GREP_SHORT_FLAGS = frozenset("rRniHhlLcwxoFEvsaIqb")
    _GREP_LONG_FLAGS = {
        "--recursive", "--dereference-recursive", "--line-number", "--ignore-case",
        "--with-filename", "--no-filename", "--files-with-matches",
        "--files-without-match", "--count", "--word-regexp", "--line-regexp",
        "--invert-match", "--silent", "--quiet", "--text", "--binary",
        "--extended-regexp", "--fixed-strings", "--basic-regexp",
    }
    _SIMPLE_READ_COMMANDS = {
        "cat", "type", "ls", "dir", "wc", "stat", "file", "du",
        "sha256sum", "md5sum", "cmp", "diff", "readlink", "realpath",
    }
    _SIMPLE_SHORT_OPTIONS = {
        "cat": frozenset("AbeEnstTuv"),
        "type": frozenset(),
        "ls": frozenset("ABCDFGHINQRSTUWZabcdefghiklmnopqrstuvwx1"),
        "dir": frozenset("ABCDFGHINQRSTUWZabcdefghiklmnopqrstuvwx1"),
        "wc": frozenset("cmlLw"),
        "stat": frozenset("Lft"),
        "file": frozenset("bEhiklLNnprsSvzZ0"),
        "du": frozenset("abchkmstx"),
        "sha256sum": frozenset("btwz"),
        "md5sum": frozenset("btwz"),
        "cmp": frozenset("bls"),
        "diff": frozenset("abBdEHiNpqrsTtuwWZ"),
        "readlink": frozenset("efmnqsvz"),
        "realpath": frozenset("eLmszq"),
    }
    _SIMPLE_LONG_OPTIONS = {
        "cat": {"--show-all", "--number-nonblank", "--show-ends", "--number",
                "--squeeze-blank", "--show-tabs", "--show-nonprinting"},
        "type": set(),
        "ls": {"--all", "--almost-all", "--author", "--escape", "--directory",
               "--dired", "--classify", "--file-type", "--no-group",
               "--human-readable", "--si", "--numeric-uid-gid", "--literal",
               "--hide-control-chars", "--show-control-chars", "--reverse",
               "--recursive", "--size", "--inode"},
        "dir": set(),
        "wc": {"--bytes", "--chars", "--lines", "--max-line-length", "--words"},
        "stat": {"--dereference", "--file-system", "--terse"},
        "file": {"--brief", "--checking-printout", "--exclude-quiet",
                 "--extension", "--dereference", "--no-dereference", "--mime",
                 "--keep-going", "--list", "--preserve-date", "--raw",
                 "--special-files", "--uncompress", "--uncompress-noreport",
                 "--print0"},
        "du": {"--all", "--apparent-size", "--count-links", "--human-readable",
               "--inodes", "--kilobytes", "--megabytes", "--null",
               "--separate-dirs", "--summarize", "--total", "--one-file-system"},
        "sha256sum": {"--binary", "--tag", "--text", "--warn", "--zero"},
        "md5sum": {"--binary", "--tag", "--text", "--warn", "--zero"},
        "cmp": {"--print-bytes", "--verbose", "--silent", "--quiet"},
        "diff": {"--brief", "--context", "--ignore-all-space", "--ignore-blank-lines",
                 "--ignore-case", "--ignore-space-change", "--minimal",
                 "--recursive", "--report-identical-files", "--speed-large-files",
                 "--strip-trailing-cr", "--text", "--unified"},
        "readlink": {"--canonicalize", "--canonicalize-existing",
                     "--canonicalize-missing", "--no-newline", "--quiet",
                     "--silent", "--verbose", "--zero"},
        "realpath": {"--canonicalize-existing", "--canonicalize-missing",
                     "--logical", "--physical", "--quiet", "--strip",
                     "--zero"},
    }
    _FIND_WRITE_ACTIONS = {
        "-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint",
        "-fprint0", "-fprintf", "-fls",
    }

    def __init__(self, cwd: Optional[str] = None, max_output_chars: int = 8000):
        self._cwd = os.path.abspath(cwd or os.getcwd())
        self.workspace_root = os.path.realpath(self._cwd)
        self.read_only_roots: Tuple[str, ...] = ()
        self.env = os.environ.copy()
        self.max_output_chars = max_output_chars

    @property
    def cwd(self) -> str:
        return self._cwd

    @cwd.setter
    def cwd(self, value: str) -> None:
        """Explicit assignment configures a new workspace (used by embedders/tests)."""
        self._cwd = os.path.abspath(value)
        self.workspace_root = os.path.realpath(self._cwd)

    def set_read_only_roots(self, paths: Iterable[str]) -> None:
        """Bind canonical, deduplicated roots that file tools may only read."""
        roots = []
        for path in paths:
            resolved = os.path.realpath(os.path.abspath(os.path.expanduser(str(path))))
            if resolved not in roots:
                roots.append(resolved)
        self.read_only_roots = tuple(roots)

    @staticmethod
    def _strip_matching_quotes(value: str) -> str:
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            return value[1:-1]
        return value

    @staticmethod
    def _is_sensitive_reference(path: str) -> bool:
        normalized = os.path.realpath(path).replace("\\", "/").lower()
        parts = tuple(part for part in normalized.split("/") if part)
        name = parts[-1] if parts else ""
        if name == ".env.example":
            return False
        if name == ".env" or name.startswith(".env."):
            return True
        sensitive_names = {
            "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", ".netrc",
            "credentials", "credentials.json", "service-account.json",
            ".git-credentials", ".envrc", ".npmrc", ".pypirc", "auth.json",
            ".anthropic_oauth.json",
        }
        protected_dirs = {".ssh", ".aws", ".azure", ".kube", ".docker"}
        return (
            name in sensitive_names
            or name.endswith((".pem", ".key", ".p12", ".pfx"))
            or bool(protected_dirs.intersection(parts))
            or "/.config/gcloud/" in f"/{normalized}/"
            or "/.config/gh/" in f"/{normalized}/"
        )

    def _is_existing_allowed_path(self, value: str) -> bool:
        if not value or any(char in value for char in "*?["):
            return False
        candidate = os.path.realpath(os.path.abspath(os.path.join(self.cwd, value)))
        if not os.path.exists(candidate) or self._is_sensitive_reference(candidate):
            return False
        roots = (self.workspace_root, *self.read_only_roots)
        for root in roots:
            try:
                if os.path.commonpath((os.path.realpath(root), candidate)) == os.path.realpath(root):
                    return True
            except ValueError:
                continue
        return False

    def _contains_sensitive_descendant(self, value: str, limit: int = 10000) -> bool:
        """Fail closed when a recursive read could encounter credentials."""
        candidate = os.path.realpath(os.path.abspath(os.path.join(self.cwd, value)))
        if not os.path.isdir(candidate):
            return False
        examined = 0
        for current, directories, files in os.walk(candidate, followlinks=False):
            for name in (*directories, *files):
                examined += 1
                if examined > limit or self._is_sensitive_reference(os.path.join(current, name)):
                    return True
        return False

    @staticmethod
    def _short_option_present(arguments: list[str], letters: str) -> bool:
        for value in arguments:
            if value == "--":
                break
            if not value.startswith("-") or value == "-":
                break
            if not value.startswith("--") and any(letter in value[1:] for letter in letters):
                return True
        return False

    @classmethod
    def _head_paths(cls, arguments: list[str]) -> Optional[list[str]]:
        index = 0
        while index < len(arguments):
            value = arguments[index]
            if value == "--":
                return arguments[index + 1:] or None
            if not value.startswith("-") or value == "-":
                return arguments[index:]
            if value in ("-n", "--lines", "-c", "--bytes"):
                if index + 1 >= len(arguments) or not re.fullmatch(r"[+-]?\d+[A-Za-z]*", arguments[index + 1]):
                    return None
                index += 2
                continue
            if re.fullmatch(r"-(?:[nc])?[+-]?\d+[A-Za-z]*", value) or re.fullmatch(
                r"--(?:lines|bytes)=[+-]?\d+[A-Za-z]*", value
            ):
                index += 1
                continue
            if value in ("-q", "--quiet", "-v", "--verbose", "-z", "--zero-terminated"):
                index += 1
                continue
            return None
        return None

    @classmethod
    def _grep_paths(cls, arguments: list[str]) -> Optional[list[str]]:
        index = 0
        while index < len(arguments):
            value = arguments[index]
            if value == "--":
                index += 1
                break
            if not value.startswith("-") or value == "-":
                break
            if value.startswith("--"):
                if value not in cls._GREP_LONG_FLAGS and not value.startswith("--color="):
                    return None
            elif not value[1:] or any(flag not in cls._GREP_SHORT_FLAGS for flag in value[1:]):
                return None
            index += 1
        remaining = arguments[index:]
        if len(remaining) < 2:
            return None
        return remaining[1:]

    @classmethod
    def _simple_read_paths(cls, executable: str, arguments: list[str]) -> Optional[list[str]]:
        index = 0
        while index < len(arguments):
            value = arguments[index]
            if value == "--":
                index += 1
                break
            if value.startswith("--"):
                if value not in cls._SIMPLE_LONG_OPTIONS[executable]:
                    return None
                index += 1
                continue
            if executable == "dir" and os.name == "nt" and re.fullmatch(r"/[A-Za-z][^\\/]*", value):
                index += 1
                continue
            if value.startswith("-") and value != "-":
                if not value[1:] or any(
                    flag not in cls._SIMPLE_SHORT_OPTIONS[executable]
                    for flag in value[1:]
                ):
                    return None
                index += 1
                continue
            break
        paths = arguments[index:]
        if any(path.startswith("-") for path in paths):
            return None
        if paths:
            return paths
        if executable in ("ls", "dir", "du"):
            return ["."]
        return None

    @classmethod
    def _find_paths(cls, arguments: list[str]) -> Optional[list[str]]:
        index = 0
        if index < len(arguments) and arguments[index] == "-P":
            index += 1
        elif index < len(arguments) and arguments[index].startswith("-"):
            return None
        roots = []
        while index < len(arguments):
            value = arguments[index]
            if value.startswith("-") or value in ("!", "(", ")", ","):
                break
            roots.append(value)
            index += 1
        if not roots:
            roots = ["."]
        expression = arguments[index:]
        lowered = {value.lower() for value in expression}
        if lowered.intersection(cls._FIND_WRITE_ACTIONS):
            return None
        if any(value.startswith("-newer") or value == "-samefile" for value in lowered):
            return None
        return roots

    def _has_local_command_shadow(self, executable: str) -> bool:
        """Reject auto-approval when shell lookup could run workspace code."""
        if os.name == "nt":
            extensions = [""] + [
                extension.lower()
                for extension in self.env.get("PATHEXT", ".COM;.EXE;.BAT;.CMD").split(";")
                if extension
            ]
            return any(
                os.path.isfile(os.path.join(self.cwd, executable + extension))
                for extension in extensions
            )
        path_entries = self.env.get("PATH", "").split(os.pathsep)
        if "" not in path_entries and "." not in path_entries:
            return False
        candidate = os.path.join(self.cwd, executable)
        return os.path.isfile(candidate) and os.access(candidate, os.X_OK)

    def is_auto_approved_read_only_command(self, command: str) -> bool:
        """Allow recognized non-shelling reads confined to configured filesystem roots."""
        if not isinstance(command, str):
            return False
        if re.search(r"[\r\n\x00;&|<>`$^()%!~{}#]", command):
            return False
        try:
            tokens = shlex.split(command, posix=os.name != "nt")
        except ValueError:
            return False
        if os.name == "nt":
            tokens = [self._strip_matching_quotes(token) for token in tokens]
        if not tokens:
            return False
        if "/" in tokens[0] or "\\" in tokens[0]:
            return False
        executable = os.path.basename(tokens[0]).lower()
        if executable.endswith(".exe"):
            executable = executable[:-4]
        if self._has_local_command_shadow(executable):
            return False
        if executable == "head":
            paths = self._head_paths(tokens[1:])
        elif executable == "tail":
            paths = self._head_paths(tokens[1:])
        elif executable == "grep":
            paths = self._grep_paths(tokens[1:])
            dereference_recursive = (
                self._short_option_present(tokens[1:], "R")
                or "--dereference-recursive" in tokens[1:]
            )
            recursive = (
                dereference_recursive
                or self._short_option_present(tokens[1:], "r")
                or "--recursive" in tokens[1:]
            )
            if dereference_recursive or (
                paths
                and recursive
                and any(self._contains_sensitive_descendant(path) for path in paths)
            ):
                paths = None
        elif executable == "find":
            paths = self._find_paths(tokens[1:])
        elif executable == "pwd":
            paths = ["."] if len(tokens) == 1 else None
        elif executable == "file" and any(
            arg in {"-C", "--compile"} or arg.startswith("--compile=")
            for arg in tokens[1:]
        ):
            paths = None
        elif executable == "du" and (
            self._short_option_present(tokens[1:], "DHL")
            or any(
                arg in {"--dereference", "--dereference-args"}
                or arg.startswith("--dereference-args=")
                for arg in tokens[1:]
            )
        ):
            paths = None
        elif executable == "ls" and (
            self._short_option_present(tokens[1:], "L")
            or "--dereference" in tokens[1:]
        ):
            paths = None
        elif executable in self._SIMPLE_READ_COMMANDS:
            paths = self._simple_read_paths(executable, tokens[1:])
        else:
            return False
        return bool(paths) and all(self._is_existing_allowed_path(path) for path in paths)

    def is_destructive(self, command: str) -> bool:
        """Check if a command matches potentially destructive patterns."""
        for pattern in self.DESTRUCTIVE_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return True
        return False

    def is_safe(self, command: str) -> bool:
        """Check if a command starts with a known safe read-only prefix."""
        cmd_strip = command.strip().lower()
        if re.search(r"(?:&&|\|\||[|;&])", cmd_strip):
            return False
        return any(cmd_strip == prefix or cmd_strip.startswith(prefix + " ") for prefix in self.SAFE_COMMAND_PREFIXES)

    def _handle_cd(self, command: str) -> Optional[str]:
        """
        Handle 'cd' command separately to persist working directory across steps.
        Returns a status string if it was purely a cd command, or None to run via shell.
        """
        if re.search(r"(?:&&|\|\||[|;&])", command):
            return None
        cmd_parts = command.strip().split()
        if len(cmd_parts) >= 1 and cmd_parts[0].lower() == "cd":
            if len(cmd_parts) == 1:
                target_dir = os.path.expanduser("~")
            else:
                target_dir = command.strip()[3:].strip().strip("\"'")
                target_dir = os.path.expanduser(target_dir)

            new_path = os.path.abspath(os.path.join(self.cwd, target_dir))
            if os.path.isdir(new_path):
                self._cwd = new_path
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

            # A successful leading cd in a compound shell command updates the
            # session cwd without attempting to interpret the whole chain as a path.
            if exit_code == 0:
                leading_cd = re.match(r'^\s*cd\s+(?:/d\s+)?("[^"]+"|\'[^\']+\'|[^&|;]+?)\s*(?:&&|;)', command, re.IGNORECASE)
                if leading_cd:
                    target = os.path.expanduser(leading_cd.group(1).strip().strip("\"'"))
                    changed = os.path.abspath(os.path.join(self.cwd, target))
                    if os.path.isdir(changed):
                        self._cwd = changed

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
