"""
memory.py – SQLite conversation memory with emotion arc tracking.

Extends basic storage with:
  - Emotion arc: tracks the emotional trajectory across a session
  - Conversation summary: last-N messages with emotion labels
  - Session mood: dominant emotion in recent turns (like Sesame's contextual awareness)
"""

import sqlite3
import os
from datetime import datetime
from collections import Counter

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "database", "db.sqlite")


def _get_conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL DEFAULT 'default',
                role       TEXT NOT NULL,
                content    TEXT NOT NULL,
                emotion    TEXT DEFAULT 'calm',
                timestamp  TEXT NOT NULL
            )
        """)
        conn.commit()


def save_message(
    role: str,
    content: str,
    emotion: str = "calm",
    session_id: str = "default",
) -> None:
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO conversations (session_id, role, content, emotion, timestamp) VALUES (?,?,?,?,?)",
            (session_id, role, content, emotion, datetime.utcnow().isoformat()),
        )
        conn.commit()


def get_recent_messages(n: int = 10, session_id: str = "default") -> list[dict]:
    """Return last n messages as chat-completion dicts (oldest first)."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT role, content FROM conversations WHERE session_id=? ORDER BY id DESC LIMIT ?",
            (session_id, n),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def get_emotion_arc(n: int = 6, session_id: str = "default") -> list[str]:
    """
    Return the last n assistant emotions in chronological order.

    This is Sesame's 'contextual awareness' concept:
    knowing the emotional trajectory lets the LLM continue
    the arc naturally rather than resetting each turn.

    Example:  ['calm', 'calm', 'sad', 'sad', 'calm', 'happy']
    → The assistant is warming up → provide encouragement.
    """
    with _get_conn() as conn:
        rows = conn.execute(
            """SELECT emotion FROM conversations
               WHERE session_id=? AND role='assistant'
               ORDER BY id DESC LIMIT ?""",
            (session_id, n),
        ).fetchall()
    return [r["emotion"] for r in reversed(rows)]


def get_session_mood(n: int = 6, session_id: str = "default") -> str:
    """
    Dominant emotion in the last n assistant turns.
    Returns the most common emotion string (e.g. 'sad').
    Falls back to 'calm' if no history.
    """
    arc = get_emotion_arc(n=n, session_id=session_id)
    if not arc:
        return "calm"
    return Counter(arc).most_common(1)[0][0]


def get_all_sessions() -> list[str]:
    with _get_conn() as conn:
        rows = conn.execute("SELECT DISTINCT session_id FROM conversations").fetchall()
    return [r["session_id"] for r in rows]
