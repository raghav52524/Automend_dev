"""Flush/empty the prompts table in raw/ds4_synthetic/prompts.db. Table structure is kept."""
import sqlite3
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DS4_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = DS4_ROOT.parent.parent
sys.path.insert(0, str(DS4_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import db_ops
from src.config.paths import get_ds4_prompts_db

DB_PATH = get_ds4_prompts_db()


def main():
    if not DB_PATH.exists():
        print(f"No DB at {DB_PATH}; nothing to flush.")
        return
    conn = sqlite3.connect(DB_PATH)
    try:
        db_ops.setup_db(conn)  # ensure table exists
        cur = conn.execute("DELETE FROM prompts")
        conn.commit()
        print(f"Flushed prompts.db: deleted {cur.rowcount} row(s).")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
