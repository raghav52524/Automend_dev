"""Normalize Hadoop_2k.log_structured.csv into the unified schema.

Actual columns: LineId, Date, Time, Level, Process, Component, Content, EventId, EventTemplate
Level is standard (INFO, WARN, ERROR, FATAL) — direct mapping.
Timestamp: "2015-10-18 18:01:47,978" (ISO date with comma-milliseconds) — kept as string.
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

RAW_DIR = get_ds2_raw_dir() / "loghub"
if not RAW_DIR.exists():
    RAW_DIR = get_legacy_raw_dir() / "loghub"

PROCESSED_DIR = get_ds2_processed_dir()

INPUT  = RAW_DIR / "Hadoop" / "Hadoop_2k.log_structured.csv"
OUTPUT = PROCESSED_DIR / "normalized" / "hadoop.parquet"

LEVEL_MAP = {
    "INFO":    "INFO",
    "WARN":    "WARN",
    "WARNING": "WARN",
    "ERROR":   "ERROR",
    "FATAL":   "ERROR",
    "DEBUG":   "INFO",
    "TRACE":   "INFO",
}
ERROR_KW = ["exception", "failed", "failure", "killed", "oom", "error", "denied"]
WARN_KW  = ["warn", "timeout", "retry", "slow", "fallback"]


def normalize_severity(level: str, message: str) -> str:
    mapped = LEVEL_MAP.get(str(level).strip().upper())
    if mapped:
        return mapped
    m = str(message).lower()
    for kw in ERROR_KW:
        if kw in m:
            return "ERROR"
    for kw in WARN_KW:
        if kw in m:
            return "WARN"
    return "INFO"


def normalize_hadoop(input_path: Path = INPUT, output_path: Path = OUTPUT):
    logger.info("Reading %s", input_path)
    df = read_csv(str(input_path))

    # Timestamp: "2015-10-18 18:01:47,978" — combine Date + Time as string
    df["timestamp"] = df["Date"] + " " + df["Time"]

    df["system"]         = "hadoop"
    df["raw_id"]         = df["LineId"]
    df["source"]         = df["Component"]
    df["message"]        = df["Content"]
    df["event_id"]       = df["EventId"]
    df["event_template"] = df["EventTemplate"]
    df["severity"]       = df.apply(
        lambda r: normalize_severity(r["Level"], r["Content"]), axis=1
    )
    df["extras"] = df.apply(
        lambda r: json.dumps({"process": r["Process"], "level_raw": r["Level"]}), axis=1
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
    normalize_hadoop()
