"""Host-selected profile paths and shared defaults; never model-facing authority."""
import os
from pathlib import Path
import re
import stat

from skill_lock import path_identity


DEFAULT_PROFILES_DIR = Path(__file__).resolve().parent / ".agent_profiles"
PROFILE_ID = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")


def default_model():
    return os.environ.get("AGENT_MODEL", "Qwen-32b")


def default_base_url():
    return os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")


def validate_profile_id(value):
    if not isinstance(value, str) or not PROFILE_ID.fullmatch(value):
        raise ValueError("--profile must be 1-64 letters, digits, hyphens, or underscores")


def absolute_path(path):
    # Do not resolve links before inspecting the path chosen by the host.
    return Path(os.path.abspath(Path(path).expanduser()))


def plain_stat(path, *, directory=False):
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
        raise ValueError("Profile paths cannot use symlinks, junctions or reparse points")
    if directory and not stat.S_ISDIR(info.st_mode):
        raise ValueError("Profile path is not a directory")
    if not directory and (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1):
        raise ValueError("Profile file must be a regular file without aliases")
    return info


def incarnation(info):
    if not info.st_ino:
        raise ValueError("Filesystem cannot provide a stable directory identity")
    return (info.st_dev, info.st_ino, getattr(info, "st_birthtime_ns", 0))


class PinnedDirectory:
    """Pin every existing ancestor; allow a missing suffix to be created once."""
    def __init__(self, path):
        self.path = absolute_path(path)
        self._pins = {}
        self._failed = False
        self.validate(required=False)

    def validate(self, *, required=True):
        if self._failed:
            raise ValueError("Host-bound directory identity changed")
        for path in (*reversed(self.path.parents), self.path):
            try:
                stamp = incarnation(plain_stat(path, directory=True))
            except FileNotFoundError:
                if path in self._pins:
                    self._failed = True
                    raise ValueError("Host-bound directory identity changed")
                if required:
                    raise ValueError(f"Named profile path does not exist: {self.path}")
                return False
            except (OSError, ValueError):
                self._failed = True
                raise
            if path in self._pins and stamp != self._pins[path]:
                self._failed = True
                raise ValueError("Host-bound directory identity changed")
            self._pins[path] = stamp
        if os.path.normcase(str(self.path.resolve())) != os.path.normcase(str(self.path)):
            self._failed = True
            raise ValueError("Host-bound directory alias changed")
        return True

    @property
    def stamp(self):
        self.validate()
        return list(self._pins[self.path])


def control_directory(root):
    root = absolute_path(root)
    if root == root.parent:
        raise ValueError("Profile root cannot be a filesystem root")
    # Keep external auth paths below common Windows path-length limits. The
    # policy also verifies the full root, so even a hash collision fails closed.
    return root.parent / (".skill-review-" + path_identity(root)[:24])


class ProfilePaths:
    def __init__(self, root, profile_id):
        validate_profile_id(profile_id)
        self.root = root if isinstance(root, PinnedDirectory) else PinnedDirectory(root)
        self.root.validate(required=False)
        self.profile_id = profile_id
        self.directory = PinnedDirectory(self.root.path / profile_id)
        self.validate(required=False)

    def validate(self, *, required=True):
        self.root.validate(required=required)
        exists = self.directory.validate(required=required)
        if exists:
            if self.directory.path.resolve().name != self.profile_id:
                raise ValueError("Profile directory alias is not an exact profile ID")
            # Fixed metadata only: no catalog enumeration or file content reads.
            for name in ("skills", "memories"):
                path = self.directory.path / name
                try:
                    plain_stat(path, directory=True)
                except FileNotFoundError:
                    pass
            for name in ("skill_review.db", "skill_review.db-journal", "skill_review.db-wal", "skill_review.db-shm",
                         "history.db", "history.db-journal", "history.db-wal", "history.db-shm",
                         "memories/USER.md", "memories/MEMORY.md"):
                try:
                    plain_stat(self.directory.path / name)
                except FileNotFoundError:
                    pass
        return exists

    @property
    def identity(self):
        self.validate()
        return dict(root=self.root.stamp, profile=self.directory.stamp)
