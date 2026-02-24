"""SQLite prompt fetch/write for Dataset 4 pipeline."""
import sqlite3


def setup_db(conn: sqlite3.Connection) -> None:
    """Initialize the prompts table: id, user_intent, tool_context, processed."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prompts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_intent TEXT NOT NULL,
            tool_context TEXT NOT NULL,
            processed INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.commit()


def fetch_unprocessed_prompts(conn: sqlite3.Connection) -> list[dict]:
    """Return all rows from prompts where processed = 0. Rows as dicts."""
    conn.row_factory = sqlite3.Row
    cur = conn.execute("SELECT id, user_intent, tool_context FROM prompts WHERE processed = 0")
    return [dict(row) for row in cur.fetchall()]


def mark_prompts_processed(conn: sqlite3.Connection, prompt_ids: list[int]) -> int:
    """Mark the given prompt IDs as processed (processed = 1).
    
    Args:
        conn: SQLite connection
        prompt_ids: List of prompt IDs to mark as processed
        
    Returns:
        Number of rows updated
    """
    if not prompt_ids:
        return 0
    placeholders = ",".join("?" * len(prompt_ids))
    cur = conn.execute(
        f"UPDATE prompts SET processed = 1 WHERE id IN ({placeholders})",
        prompt_ids
    )
    conn.commit()
    return cur.rowcount
