"""
Storage and trajectory logger for the coding agent.
Records all tool executions, messages, and iterations to SQLite and JSONL
for evaluation, debugging, and model fine-tuning dataset export.
"""

import json
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional


class TrajectoryLogger:
    """Logs conversation trajectories and tool executions into SQLite and JSONL."""

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
