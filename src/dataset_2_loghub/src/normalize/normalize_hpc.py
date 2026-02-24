"""Normalize HPC_2k.log_structured.csv into the unified schema.

Actual columns: LineId, LogId, Node, Component, State, Time, Flag, Content, EventId, EventTemplate
No Level column — severity from State + Flag (Flag='1' means alert/error).
Timestamp: Unix epoch integer stored as string (e.g., "1077804742").
State example: "state_change.unavailable"
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

INPUT  = RAW_DIR / "HPC" / "HPC_2k.log_structured.csv"
OUTPUT = PROCESSED_DIR / "normalized" / "hpc.parquet"


def normalize_severity(state: str, flag: str, message: str) -> str:
    s = str(state).lower()
    m = str(message).lower()
    f = str(flag).strip()

    if ("unavailable" in s or f == "1"
            or any(kw in m for kw in ("unavailable", "panic", "fatal", "error"))):
        return "ERROR"
    if (any(kw in s for kw in ("degraded", "fail", "error"))
            or any(kw in m for kw in ("timeout", "retry", "slow"))):
        return "WARN"
    return "INFO"


def normalize_hpc(input_path: Path = INPUT, output_path: Path = OUTPUT):
    logger.info("Reading %s", input_path)
    df = read_csv(str(input_path))

    # Prefer LogId as raw_id; fall back to LineId if blank
    df["raw_id"] = df.apply(
        lambda r: r["LogId"] if str(r["LogId"]).strip() else r["LineId"], axis=1
    )

    # Timestamp: Unix epoch string — keep as-is to avoid float precision loss
    df["timestamp"] = df["Time"]

    # Source: "node-246|unix.hw"
    df["source"] = df["Node"] + "|" + df["Component"]

    df["system"]         = "hpc"
    df["message"]        = df["Content"]
    df["event_id"]       = df["EventId"]
    df["event_template"] = df["EventTemplate"]
    df["severity"]       = df.apply(
        lambda r: normalize_severity(r["State"], r["Flag"], r["Content"]), axis=1
    )
    df["extras"] = df.apply(
        lambda r: json.dumps({
            "node":      r["Node"],
            "component": r["Component"],
            "state":     r["State"],
            "flag":      r["Flag"],
        }), axis=1
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
    normalize_hpc()

