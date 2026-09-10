"""Process and thread exclusion, without leases or profile lock-file writes.

Windows uses kernel named mutexes (including abandoned-owner recovery). POSIX
uses flock on coordination files in the OS temp directory, outside profiles.
Locks coordinate cooperating processes; they are not an ACL/security boundary.
"""
import hashlib
import os
from pathlib import Path
import tempfile
import threading
import time


_process_held = set()
_process_guard = threading.Lock()


def path_identity(path):
    return hashlib.sha256(os.path.normcase(os.path.realpath(path)).encode("utf-8")).hexdigest()


class ProcessLock:
    def __init__(self, identity, timeout=5.0):
        self.identity = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        self.timeout = timeout
        self._local = threading.local()

    def __enter__(self):
        # Kernel mutexes are reentrant for the owning Windows thread. Service
        # and owner lifetime exclusion must also reject that second instance.
        with _process_guard:
            if self.identity in _process_held:
                raise TimeoutError("Skill coordination lock is busy")
            _process_held.add(self.identity)
        try:
            return self._acquire()
        except BaseException:
            with _process_guard:
                _process_held.discard(self.identity)
            raise

    def _acquire(self):
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.CreateMutexW.argtypes = (ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR)
            kernel.CreateMutexW.restype = wintypes.HANDLE
            kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
            kernel.WaitForSingleObject.restype = wintypes.DWORD
            kernel.ReleaseMutex.argtypes = (wintypes.HANDLE,)
            kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
            handle = kernel.CreateMutexW(None, False, "Global\\HermesSkill-" + self.identity)
            if not handle:
                raise ctypes.WinError(ctypes.get_last_error())
            result = kernel.WaitForSingleObject(handle, int(self.timeout * 1000))
            if result not in (0, 0x80):  # WAIT_OBJECT_0 / WAIT_ABANDONED
                kernel.CloseHandle(handle)
                raise TimeoutError("Skill coordination lock is busy")
            release = lambda: (kernel.ReleaseMutex(handle), kernel.CloseHandle(handle))
        else:
            import fcntl
            path = Path(tempfile.gettempdir()) / ("hermes-skill-" + self.identity + ".lock")
            fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
            deadline = time.monotonic() + self.timeout
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        os.close(fd)
                        raise TimeoutError("Skill coordination lock is busy")
                    time.sleep(0.005)
            release = lambda: (fcntl.flock(fd, fcntl.LOCK_UN), os.close(fd))
        stack = getattr(self._local, "stack", [])
        self._local.stack = stack + [release]
        return self

    def __exit__(self, *args):
        self._local.stack.pop()()
        with _process_guard:
            _process_held.discard(self.identity)


_locks = {}
_guard = threading.Lock()


class CatalogLock:
    """One reentrant lock per resolved catalog, across stores and processes."""
    def __init__(self, path=None, timeout=5.0, *, identity=None):
        self.key = "catalog:" + (identity or path_identity(path))
        self.timeout = timeout
        with _guard:
            self.thread_lock, self.local = _locks.setdefault(self.key, (threading.RLock(), threading.local()))

    def __enter__(self):
        if not self.thread_lock.acquire(timeout=self.timeout):
            raise TimeoutError("Catalog mutation is busy")
        try:
            if not getattr(self.local, "depth", 0):
                self.local.process = ProcessLock(self.key, self.timeout)
                self.local.process.__enter__()
            self.local.depth = getattr(self.local, "depth", 0) + 1
        except BaseException:
            self.thread_lock.release()
            raise
        return self

    def __exit__(self, *args):
        self.local.depth -= 1
        if self.local.depth == 0:
            self.local.process.__exit__(*args)
        self.thread_lock.release()
