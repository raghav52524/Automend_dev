"""Phase 3: SQLite prompt fetcher tests. Use in-memory DB in fixtures."""
import sys
from pathlib import Path
import sqlite3
import pytest

DS4_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DS4_ROOT / "src"))

from data import db_ops


def test_setup_creates_prompts_table():
    conn = sqlite3.connect(":memory:")
    db_ops.setup_db(conn)
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='prompts'"
    )
    row = cur.fetchone()
    conn.close()
    assert row is not None
    assert row[0] == "prompts"


def test_setup_table_has_required_columns():
    conn = sqlite3.connect(":memory:")
    db_ops.setup_db(conn)
    cur = conn.execute("PRAGMA table_info(prompts)")
    columns = {row[1] for row in cur.fetchall()}
    conn.close()
    assert "id" in columns
    assert "user_intent" in columns
    assert "tool_context" in columns


def test_fetch_unprocessed_returns_inserted_prompts():
    conn = sqlite3.connect(":memory:")
    db_ops.setup_db(conn)
    conn.execute(
        "INSERT INTO prompts (user_intent, tool_context) VALUES (?, ?)",
        ("Fix latency", "scale_service, restart_pod"),
    )
    conn.execute(
        "INSERT INTO prompts (user_intent, tool_context) VALUES (?, ?)",
        ("Restart model", "restart_pod"),
    )
    conn.commit()
    rows = db_ops.fetch_unprocessed_prompts(conn)
    conn.close()
    assert len(rows) == 2
    assert rows[0]["user_intent"] == "Fix latency"
    assert rows[0]["tool_context"] == "scale_service, restart_pod"
    assert rows[1]["user_intent"] == "Restart model"


def test_fetch_unprocessed_excludes_processed():
    conn = sqlite3.connect(":memory:")
    db_ops.setup_db(conn)
    conn.execute(
        "INSERT INTO prompts (user_intent, tool_context, processed) VALUES (?, ?, ?)",
        ("Done intent", "tools", 1),
    )
    conn.execute(
        "INSERT INTO prompts (user_intent, tool_context, processed) VALUES (?, ?, ?)",
        ("Pending intent", "tools", 0),
    )
    conn.commit()
    rows = db_ops.fetch_unprocessed_prompts(conn)
    conn.close()
    assert len(rows) == 1
    assert rows[0]["user_intent"] == "Pending intent"


def test_mark_prompts_processed_updates_rows():
    """Test that mark_prompts_processed sets processed=1 for given IDs."""
    conn = sqlite3.connect(":memory:")
    db_ops.setup_db(conn)
    # Insert 3 prompts
    conn.execute(
        "INSERT INTO prompts (user_intent, tool_context) VALUES (?, ?)",
        ("Intent 1", "tools1"),
    )
    conn.execute(
        "INSERT INTO prompts (user_intent, tool_context) VALUES (?, ?)",
        ("Intent 2", "tools2"),
    )
    conn.execute(
        "INSERT INTO prompts (user_intent, tool_context) VALUES (?, ?)",
        ("Intent 3", "tools3"),
    )
    conn.commit()
    
    # Mark first two as processed
    updated = db_ops.mark_prompts_processed(conn, [1, 2])
    assert updated == 2
    
    # Verify only ID 3 remains unprocessed
    rows = db_ops.fetch_unprocessed_prompts(conn)
    conn.close()
    assert len(rows) == 1
    assert rows[0]["id"] == 3
    assert rows[0]["user_intent"] == "Intent 3"


def test_mark_prompts_processed_empty_list():
    """Test that mark_prompts_processed handles empty list gracefully."""
    conn = sqlite3.connect(":memory:")
    db_ops.setup_db(conn)
    conn.execute(
        "INSERT INTO prompts (user_intent, tool_context) VALUES (?, ?)",
        ("Intent 1", "tools1"),
    )
    conn.commit()
    
    # Mark with empty list
    updated = db_ops.mark_prompts_processed(conn, [])
    assert updated == 0
    
    # Verify nothing changed
    rows = db_ops.fetch_unprocessed_prompts(conn)
    conn.close()
    assert len(rows) == 1
