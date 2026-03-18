"""Reset all DS4 prompts to unprocessed so the DAG can re-run."""
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.config.paths import get_ds4_prompts_db
    DB_PATH = get_ds4_prompts_db()
except ImportError:
    DB_PATH = PROJECT_ROOT / "data" / "raw" / "ds4_synthetic" / "prompts.db"


def main():
    if not DB_PATH.exists():
        print(f"DB not found: {DB_PATH}")
        return
    conn = sqlite3.connect(DB_PATH)
    try:
        updated = conn.execute("UPDATE prompts SET processed = 0").rowcount
        conn.commit()
        print(f"Reset {updated} prompts to unprocessed in {DB_PATH}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
