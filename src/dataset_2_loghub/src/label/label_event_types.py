"""Add event_type column to sampled events using deterministic rule-based labeling.

Categories (applied in order — first match wins):
  auth_failure, permission_denied, storage_unavailable, compute_oom,
  network_issue, data_ingestion_failed, executor_failure, system_crash,
  job_failed, normal_ops, unknown

Input/Output: data/processed/ds2_loghub/mlops_processed/mlops_events.parquet (updated in-place)
"""
import sys
from pathlib import Path

DS2_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DS2_SRC))
from utils.paths import get_ds2_processed_dir

from utils.io import read_parquet, write_parquet
from utils.logger import get_logger

logger = get_logger(__name__)

PROCESSED_DIR = get_ds2_processed_dir()
EVENTS_PATH = PROCESSED_DIR / "mlops_processed" / "mlops_events.parquet"

# Rules: (event_type, [keywords to search in message+template combined])
# Applied in order — first match wins
RULES = [
    ("auth_failure",        ["authentication failure", "failed password",
                             "login failed", "pam_unix"]),
    ("permission_denied",   ["permission denied", "not permitted",
                             "access denied", "authorization"]),
    ("storage_unavailable", ["unavailable", "state_change.unavailable",
                             "block missing", "namenode down", "disk failure"]),
    ("compute_oom",         ["outofmemory", "oom", "java heap space",
                             "killed container"]),
    ("network_issue",       ["connection refused", "timed out",
                             "unable to connect", "broken pipe", "reset by peer"]),
    ("data_ingestion_failed", ["ingest", "input split", "failed to read",
                               "could not read", "ioexception"]),
    ("executor_failure",    ["executor lost", "task failed",
                             "stage failed", "container killed"]),
    ("system_crash",        ["clusteraddmember", "risboot", "bootgenvmunix",
                             "invalid acpi", "irq routing", "pci:",
                             "halt", "reboot", "kernel panic", "segfault"]),
    ("job_failed",          ["exception", "failed", "failure", "error"]),
]


def label_event_type(message: str, template: str, severity: str) -> str:
    combined = (str(message) + " " + str(template)).lower()

    for event_type, keywords in RULES:
        for kw in keywords:
            if kw in combined:
                return event_type

    if severity == "INFO":
        return "normal_ops"
    return "unknown"


def label_event_types(events_path: Path = EVENTS_PATH):
    logger.info("Reading %s", events_path)
    df = read_parquet(str(events_path))

    df["event_type"] = df.apply(
        lambda r: label_event_type(r["message"], r["event_template"], r["severity"]),
        axis=1
    )

    logger.info("Event type distribution:")
    for etype, count in df["event_type"].value_counts().items():
        logger.info("  %s: %d", etype, count)

    write_parquet(df, str(events_path))
    logger.info("Done — %d rows labeled.", len(df))
    return df


if __name__ == "__main__":
    label_event_types()
