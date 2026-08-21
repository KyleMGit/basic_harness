"""
Storage, trajectory logger, and session resumption engine for the coding agent.
Records all tool executions, messages, and iterations to SQLite and JSONL,
and supports multi-session continuity and replay.
"""

from datetime import datetime
from contextlib import closing
import json
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple


class TrajectoryLogger:
    """Logs conversation trajectories into SQLite and JSONL, and manages session resumption."""

    def __init__(self, db_path: Optional[str] = None, write_enabled: bool = True):
        self.db_path = db_path or os.path.join(os.getcwd(), ".agent_history.db")
        self.write_enabled = write_enabled
        self._initialized = False

    def set_write_enabled(self, enabled: bool):
        """Enable or disable trajectory mutations for this logger instance."""
        self.write_enabled = enabled

    def _ensure_db(self):
        """Create the database lazily so read-only agents have no init writes."""
        if not self._initialized:
            self._init_db()
            self._initialized = True

    def _init_db(self):
        with closing(sqlite3.connect(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    start_time REAL,
                    task TEXT,
                    status TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS steps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    step_index INTEGER,
                    role TEXT,
                    content TEXT,
                    tool_calls TEXT,
                    tool_call_id TEXT,
                    timestamp REAL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS session_state (
                    session_id TEXT PRIMARY KEY,
                    messages_json TEXT NOT NULL,
                    step_counter INTEGER NOT NULL,
                    context_state_json TEXT NOT NULL,
                    updated_at REAL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)
            columns = {row[1] for row in cursor.execute("PRAGMA table_info(steps)")}
            if "tool_call_id" not in columns:
                cursor.execute("ALTER TABLE steps ADD COLUMN tool_call_id TEXT")
            conn.commit()

    def start_session(self, session_id: str, task: str):
        if not self.write_enabled:
            return
        self._ensure_db()
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.cursor().execute(
                "INSERT OR REPLACE INTO sessions (session_id, start_time, task, status) VALUES (?, ?, ?, ?)",
                (session_id, time.time(), task, "IN_PROGRESS")
            )
            conn.commit()

    def log_step(
        self,
        session_id: str,
        step_index: int,
        role: str,
        content: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        tool_call_id: Optional[str] = None,
    ):
        if not self.write_enabled:
            return
        self._ensure_db()
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.cursor().execute(
                "INSERT INTO steps (session_id, step_index, role, content, tool_calls, tool_call_id, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    step_index,
                    role,
                    content or "",
                    json.dumps(tool_calls) if tool_calls else None,
                    tool_call_id,
                    time.time()
                )
            )
            conn.commit()

    def update_step_tool_calls(
        self, session_id: str, step_index: int, tool_calls: List[Dict[str, Any]]
    ):
        """Replace a recorded assistant call after command approval/editing."""
        if not self.write_enabled:
            return
        self._ensure_db()
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                "UPDATE steps SET tool_calls = ? WHERE session_id = ? AND step_index = ? AND role = 'assistant'",
                (json.dumps(tool_calls), session_id, step_index),
            )
            conn.commit()

    def end_session(self, session_id: str, status: str = "COMPLETED"):
        if not self.write_enabled:
            return
        self._ensure_db()
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.cursor().execute(
                "UPDATE sessions SET status = ? WHERE session_id = ?",
                (status, session_id)
            )
            conn.commit()

    def save_session_state(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        step_counter: int,
        context_state: Dict[str, Any],
    ):
        """Persist the active, possibly compacted transcript transactionally."""
        if not self.write_enabled:
            return
        self._ensure_db()
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                """
                INSERT INTO session_state (
                    session_id, messages_json, step_counter, context_state_json, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    messages_json = excluded.messages_json,
                    step_counter = excluded.step_counter,
                    context_state_json = excluded.context_state_json,
                    updated_at = excluded.updated_at
                """,
                (
                    session_id,
                    json.dumps(messages),
                    step_counter,
                    json.dumps(context_state),
                    time.time(),
                ),
            )
            conn.commit()

    def load_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load the latest active transcript snapshot, if this session has one."""
        if not os.path.exists(self.db_path):
            return None
        try:
            with closing(sqlite3.connect(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT messages_json, step_counter, context_state_json "
                    "FROM session_state WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
        except sqlite3.OperationalError:
            return None
        if not row:
            return None
        try:
            return {
                "messages": json.loads(row["messages_json"]),
                "step_counter": int(row["step_counter"]),
                "context_state": json.loads(row["context_state_json"]),
            }
        except (TypeError, ValueError, json.JSONDecodeError):
            return None

    def list_sessions(self, limit: int = 15) -> List[Dict[str, Any]]:
        """List past sessions with status, date, task, and step count."""
        if not os.path.exists(self.db_path):
            return []
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    s.session_id, 
                    s.start_time, 
                    s.task, 
                    s.status,
                    COUNT(st.id) as step_count
                FROM sessions s
                LEFT JOIN steps st ON s.session_id = st.session_id
                GROUP BY s.session_id
                ORDER BY s.start_time DESC
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()

        results = []
        for r in rows:
            dt_str = datetime.fromtimestamp(r["start_time"]).strftime("%Y-%m-%d %H:%M:%S") if r["start_time"] else "Unknown"
            results.append({
                "session_id": r["session_id"],
                "date": dt_str,
                "task": r["task"] or "No task recorded",
                "status": r["status"] or "UNKNOWN",
                "step_count": r["step_count"]
            })
        return results

    def load_session_messages(self, session_id: str) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Reconstruct the message history and original task for a past session.
        Returns: (task: Optional[str], messages: List[Dict[str, Any]])
        """
        if not os.path.exists(self.db_path):
            return None, []

        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # 1. Fetch session record
            cursor.execute("SELECT task FROM sessions WHERE session_id = ?", (session_id,))
            session_row = cursor.fetchone()
            if not session_row:
                return None, []

            task = session_row["task"]

            # 2. Fetch steps
            cursor.execute(
                "SELECT * FROM steps WHERE session_id = ? ORDER BY step_index ASC, id ASC",
                (session_id,)
            )
            steps = cursor.fetchall()

        reconstructed_messages = []
        for st in steps:
            role = st["role"]
            content = st["content"]
            tool_calls_json = st["tool_calls"]

            # Skip internal system logs from being re-injected as raw conversation turns
            if role in ("system_compaction", "skill_synthesis"):
                continue

            msg: Dict[str, Any] = {"role": role, "content": content}
            if tool_calls_json:
                try:
                    calls = json.loads(tool_calls_json)
                    msg["tool_calls"] = calls
                except Exception:
                    pass
            if role == "tool" and "tool_call_id" in st.keys() and st["tool_call_id"]:
                msg["tool_call_id"] = st["tool_call_id"]

            reconstructed_messages.append(msg)

        return task, reconstructed_messages

    def export_jsonl(self, session_id: str, output_file: str):
        """Export session trajectory to JSONL format."""
        if not self.write_enabled or not os.path.exists(self.db_path):
            return False
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM steps WHERE session_id = ? ORDER BY step_index ASC",
                (session_id,)
            )
            rows = cursor.fetchall()

        with open(output_file, "w", encoding="utf-8") as f:
            for row in rows:
                step_dict = {
                    "step_index": row["step_index"],
                    "role": row["role"],
                    "content": row["content"],
                    "tool_calls": json.loads(row["tool_calls"]) if row["tool_calls"] else None,
                    "tool_call_id": row["tool_call_id"] if "tool_call_id" in row.keys() else None,
                    "timestamp": row["timestamp"]
                }
                f.write(json.dumps(step_dict) + "\n")
        return True
