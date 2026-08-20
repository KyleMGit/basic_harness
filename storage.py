"""
Storage, trajectory logger, and session resumption engine for the coding agent.
Records all tool executions, messages, and iterations to SQLite and JSONL,
and supports multi-session continuity and replay.
"""

from datetime import datetime
import json
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple


class TrajectoryLogger:
    """Logs conversation trajectories into SQLite and JSONL, and manages session resumption."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.path.join(os.getcwd(), ".agent_history.db")
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
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
                    timestamp REAL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)
            conn.commit()

    def start_session(self, session_id: str, task: str):
        with sqlite3.connect(self.db_path) as conn:
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
        tool_calls: Optional[List[Dict[str, Any]]] = None
    ):
        with sqlite3.connect(self.db_path) as conn:
            conn.cursor().execute(
                "INSERT INTO steps (session_id, step_index, role, content, tool_calls, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    step_index,
                    role,
                    content or "",
                    json.dumps(tool_calls) if tool_calls else None,
                    time.time()
                )
            )
            conn.commit()

    def end_session(self, session_id: str, status: str = "COMPLETED"):
        with sqlite3.connect(self.db_path) as conn:
            conn.cursor().execute(
                "UPDATE sessions SET status = ? WHERE session_id = ?",
                (status, session_id)
            )
            conn.commit()

    def list_sessions(self, limit: int = 15) -> List[Dict[str, Any]]:
        """List past sessions with status, date, task, and step count."""
        with sqlite3.connect(self.db_path) as conn:
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
        with sqlite3.connect(self.db_path) as conn:
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

            reconstructed_messages.append(msg)

        return task, reconstructed_messages

    def export_jsonl(self, session_id: str, output_file: str):
        """Export session trajectory to JSONL format."""
        with sqlite3.connect(self.db_path) as conn:
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
                    "timestamp": row["timestamp"]
                }
                f.write(json.dumps(step_dict) + "\n")
