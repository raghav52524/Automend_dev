"""Normalize Linux_2k.log_structured.csv into the unified schema.

Actual columns: LineId, Month, Date, Time, Level, Component, PID, Content, EventId, EventTemplate
Level is always 'combo' (non-standard) → severity inferred from message keywords.
Timestamp: "Jun 14 15:16:01"
"""
import json
import sys
from pathlib import Path

DS2_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DS2_SRC))
from utils.paths import get_ds2_raw_dir, get_ds2_processed_dir, get_legacy_raw_dir

from utils.io import read_csv, write_parquet
from utils.logger import get_logger

logger = get_logger(__name__)

# Use centralized paths, fallback to legacy
RAW_DIR = get_ds2_raw_dir() / "loghub"
if not RAW_DIR.exists():
    RAW_DIR = get_legacy_raw_dir() / "loghub"

PROCESSED_DIR = get_ds2_processed_dir()

INPUT  = RAW_DIR / "Linux" / "Linux_2k.log_structured.csv"
OUTPUT = PROCESSED_DIR / "normalized" / "linux.parquet"

# Applied in order — first match wins
ERROR_KEYWORDS = [
    "authentication failure", "failed password", "permission denied",
    "not permitted", "denied", "invalid", "refused", "error",
    "failure", "segfault", "panic",
]
WARN_KEYWORDS = [
    "warn", "timeout", "timed out", "retry", "unable",
    "deprecated", "slow", "backoff",
]


def normalize_severity(message: str) -> str:
    msg = str(message).lower()
    for kw in ERROR_KEYWORDS:
        if kw in msg:
            return "ERROR"
    for kw in WARN_KEYWORDS:
        if kw in msg:
            return "WARN"
    return "INFO"


def normalize_linux(input_path: Path = INPUT, output_path: Path = OUTPUT):
    logger.info("Reading %s", input_path)
    df = read_csv(str(input_path))

    # Timestamp: "Jun 14 15:16:01"
    df["timestamp"] = df["Month"] + " " + df["Date"] + " " + df["Time"]

    df["system"]         = "linux"
    df["raw_id"]         = df["LineId"]
    df["source"]         = df["Component"]
    df["message"]        = df["Content"]
    df["event_id"]       = df["EventId"]
    df["event_template"] = df["EventTemplate"]
    df["severity"]       = df["message"].apply(normalize_severity)
    df["extras"]         = df.apply(
        lambda r: json.dumps({"pid": r["PID"], "level_raw": r["Level"]}), axis=1
    )
    df["event_type"] = ""

    unified_cols = [
        "system", "timestamp", "severity", "source",
        "event_id", "event_template", "message", "raw_id", "extras", "event_type",
    ]
    out = df[unified_cols].copy()
    write_parquet(out, str(output_path))
    logger.info("Done — %d rows.", len(out))
    return out


if __name__ == "__main__":
    normalize_linux()
