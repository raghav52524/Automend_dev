"""Convert mlops_events.parquet to Format A for log anomaly detection.

Reads:  data/processed/ds2_loghub/mlops_processed/mlops_events.parquet
Output: data/processed/ds2_loghub/data_ready/event_sequences.parquet

Format A schema (one row per 5-minute window):
    {"sequence_ids": [List[int]], "label": int}

Labels:
    0 = Normal
    1 = Resource_Exhaustion
    2 = System_Crash
    3 = Network_Failure       (keyword: Timeout, Unreachable)
    4 = Data_Drift/Corruption (keyword: Checksum, Verification Failed)
    5 = Auth_Failure
    6 = Permission_Denied

Sequence rules:
    - Max length: 512 tokens
    - Truncation: keep most recent logs (drop from start)
    - Padding: pad shorter sequences with 0s
"""
import sys
from pathlib import Path

DS2_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DS2_SRC))
from utils.paths import get_ds2_processed_dir

import pandas as pd
from utils.io import read_parquet, write_parquet
from utils.logger import get_logger

logger = get_logger(__name__)

PROCESSED_DIR = get_ds2_processed_dir()
EVENTS_PATH = PROCESSED_DIR / "mlops_processed" / "mlops_events.parquet"
OUTPUT      = PROCESSED_DIR / "data_ready" / "event_sequences.parquet"

MAX_LEN = 512
WINDOW  = "1min"

# Step 1: Map existing event_type → class (0-6)
LABEL_MAP = {
    "normal_ops":            0,  # Normal
    "unknown":               0,  # Normal (no clear failure signal)
    "compute_oom":           1,  # Resource_Exhaustion
    "storage_unavailable":   2,  # System_Crash
    "executor_failure":      2,  # System_Crash
    "system_crash":          2,  # System_Crash (HPC boot/halt/cluster ops)
    "job_failed":            2,  # System_Crash (generic failure)
    "network_issue":         3,  # Network_Failure
    "data_ingestion_failed": 4,  # Data_Drift/Corruption
    "auth_failure":          5,  # Auth_Failure
    "permission_denied":     6,  # Permission_Denied
}

# Step 2: Document-specified keyword rules applied on event_template
# These OVERRIDE the label map if matched (as per LogHub preprocessing spec)
KEYWORD_RULES = [
    (3, ["timeout", "unreachable"]),            # Network_Failure
    (4, ["checksum", "verification failed"]),   # Data_Drift/Corruption
]


def keyword_label(template: str):
    """Return event label if template matches document keyword rules, else None."""
    t = str(template).lower()
    for label, keywords in KEYWORD_RULES:
        if any(kw in t for kw in keywords):
            return label
    return None


def pad_or_truncate(seq: list, max_len: int = MAX_LEN) -> list:
    """Truncate from start (keep most recent) or pad end with zeros."""
    if len(seq) > max_len:
        return seq[-max_len:]
    return seq + [0] * (max_len - len(seq))


def format_response(events_path: Path = EVENTS_PATH, output_path: Path = OUTPUT):
    logger.info("Loading %s", events_path)
    df = read_parquet(str(events_path))

    # 1. Convert EventId string → integer (E55 → 55)
    df["event_int"] = df["event_id"].str.extract(r"E(\d+)").astype(int)
    logger.info("EventId range: E%d – E%d", df["event_int"].min(), df["event_int"].max())

    # 2. Assign event label from existing event_type labels
    df["event_label"] = df["event_type"].map(LABEL_MAP).fillna(0).astype(int)

    # 3. Override with document keyword rules on event_template
    kw_labels = df["event_template"].apply(keyword_label)
    mask = kw_labels.notna()
    df.loc[mask, "event_label"] = kw_labels[mask].astype(int)
    logger.info("Keyword rule override applied to %d rows", mask.sum())

    # 4. Parse timestamps — handle multiple formats across systems:
    #    HPC:   Unix epoch integer strings ("1077804742") → pd.to_datetime(..., unit="s")
    #    Linux: Month-day-time strings ("Jun 14 15:16:01", no year) → format="%b %d %H:%M:%S"
    #    Others (HDFS, Hadoop, Spark): standard ISO/datetime strings → default parsing
    orig_ts = df["timestamp"].copy()  # keep original strings before any conversion
    numeric_ts = pd.to_numeric(orig_ts, errors="coerce")
    is_numeric = numeric_ts.notna()
    linux_ts = pd.to_datetime(orig_ts, format="%b %d %H:%M:%S", errors="coerce")
    is_linux_fmt = linux_ts.notna() & ~is_numeric
    df["timestamp"] = pd.to_datetime(orig_ts, errors="coerce")          # standard formats
    df.loc[is_numeric, "timestamp"] = pd.to_datetime(numeric_ts[is_numeric], unit="s")
    df.loc[is_linux_fmt, "timestamp"] = linux_ts[is_linux_fmt]          # Linux month-day-time
    logger.info("Timestamp parse results — standard: %d, epoch: %d, linux-fmt: %d, NaT: %d",
                (~is_numeric & ~is_linux_fmt & df["timestamp"].notna()).sum(),
                is_numeric.sum(), is_linux_fmt.sum(), df["timestamp"].isna().sum())
    df = df.sort_values(["system", "timestamp"])
    df["window"] = df["timestamp"].dt.floor(WINDOW)

    sequences = (
        df.groupby(["system", "window"])
        .agg(
            sequence_ids=("event_int", list),
            label=("event_label", lambda x: int(x.max()))  # minority label: if ANY failure exists, use it
        )
        .reset_index()
    )

    # 5. Pad / truncate every sequence to MAX_LEN
    sequences["sequence_ids"] = sequences["sequence_ids"].apply(pad_or_truncate)

    logger.info("Total sequences: %d", len(sequences))
    logger.info("Label distribution:\n%s", sequences["label"].value_counts().to_string())

    # 6. Save Format A: only sequence_ids and label columns
    output = sequences[["sequence_ids", "label"]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(output, str(output_path))
    logger.info("Written %d sequences → %s", len(output), output_path)
    return output


if __name__ == "__main__":
    format_response()
