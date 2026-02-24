"""Data quality validation for the MLOps events dataset.

Checks:
  1. Schema — all required columns present
  2. Allowed values — system, severity, event_type
  3. Null checks — event_id, event_template, message, raw_id not null/empty
  4. Sampling sanity — sampled % between 5% and 100% of total
  5. Template coverage — every event_id in events exists in templates file
  6. No duplicate EventIds in templates

Writes: data/processed/ds2_loghub/mlops_processed/validation_report.json
Exits with code 1 if any check fails (so Airflow marks task as failed).
"""
import re
import sys
import json
from pathlib import Path

DS2_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DS2_SRC))
from utils.paths import get_ds2_processed_dir

from utils.io import read_parquet, read_csv, write_json
from utils.logger import get_logger

logger = get_logger(__name__)

PROCESSED_DIR = get_ds2_processed_dir()
EVENTS_PATH    = PROCESSED_DIR / "mlops_processed" / "mlops_events.parquet"
TEMPLATES_PATH = PROCESSED_DIR / "mlops_processed" / "mlops_templates.csv"
REPORT_PATH    = PROCESSED_DIR / "mlops_processed" / "validation_report.json"

TOTAL_ROWS   = 10_000   # 5 systems × 2000 rows each
REQUIRED_COLS = [
    "system", "timestamp", "severity", "source",
    "event_id", "event_template", "message", "raw_id", "extras", "event_type",
]
ALLOWED_SYSTEMS   = {"linux", "hpc", "hdfs", "hadoop", "spark"}
ALLOWED_SEVERITIES = {"INFO", "WARN", "ERROR"}
ALLOWED_EVENT_TYPES = {
    "auth_failure", "permission_denied", "storage_unavailable",
    "data_ingestion_failed", "compute_oom", "executor_failure",
    "network_issue", "job_failed", "system_crash", "normal_ops", "unknown",
}
EVENT_ID_PATTERN = re.compile(r"^E\d+$")


def validate_quality(events_path: Path = EVENTS_PATH,
                     templates_path: Path = TEMPLATES_PATH,
                     report_path: Path = REPORT_PATH) -> bool:
    report = {"checks": {}, "passed": True, "errors": []}

    def fail(check_name: str, msg: str):
        report["checks"][check_name] = {"status": "FAIL", "detail": msg}
        report["errors"].append(f"{check_name}: {msg}")
        report["passed"] = False
        logger.error("[FAIL] %s: %s", check_name, msg)

    def ok(check_name: str, msg: str = ""):
        report["checks"][check_name] = {"status": "PASS", "detail": msg}
        logger.info("[PASS] %s%s", check_name, f": {msg}" if msg else "")

    logger.info("Loading data...")
    df = read_parquet(str(events_path))
    tmpl = read_csv(str(templates_path))

    logger.info("Running checks...")

    # --- 1. Schema check ---
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        fail("schema", f"Missing columns: {missing_cols}")
    else:
        ok("schema", f"All {len(REQUIRED_COLS)} required columns present")

    # --- 2. Allowed values ---
    bad_systems = set(df["system"].unique()) - ALLOWED_SYSTEMS
    if bad_systems:
        fail("allowed_systems", f"Unknown system values: {bad_systems}")
    else:
        ok("allowed_systems", f"All system values valid: {set(df['system'].unique())}")

    bad_sev = set(df["severity"].unique()) - ALLOWED_SEVERITIES
    if bad_sev:
        fail("allowed_severities", f"Invalid severity values: {bad_sev}")
    else:
        ok("allowed_severities", f"Severity distribution: {df['severity'].value_counts().to_dict()}")

    bad_types = set(df["event_type"].unique()) - ALLOWED_EVENT_TYPES
    if bad_types:
        fail("allowed_event_types", f"Unknown event_type values: {bad_types}")
    else:
        ok("allowed_event_types", f"All event_type values valid")

    # --- 3. Null/empty checks ---
    for col in ["event_id", "event_template", "message", "raw_id"]:
        null_count = df[col].isnull().sum() + (df[col] == "").sum()
        if null_count > 0:
            fail(f"nulls_{col}", f"{null_count} null/empty values in '{col}'")
        else:
            ok(f"nulls_{col}", f"No nulls in '{col}'")

    # --- 4. EventId pattern (optional but recommended) ---
    bad_ids = df[~df["event_id"].str.match(r"^E\d+$", na=False)]["event_id"].unique()
    if len(bad_ids) > 0:
        fail("event_id_pattern", f"EventIds not matching E<digits>: {list(bad_ids)[:5]}")
    else:
        ok("event_id_pattern", "All EventIds match E<digits> pattern")

    # --- 5. Sampling sanity (5%–100%) ---
    pct = len(df) / TOTAL_ROWS * 100
    if not (5 <= pct <= 100):
        fail("sampling_sanity", f"Sampled {len(df)} rows = {pct:.1f}% (expected 5–100%)")
    else:
        ok("sampling_sanity", f"Sampled {len(df)} rows = {pct:.1f}% of {TOTAL_ROWS}")

    # --- 6. Template coverage ---
    tmpl_ids   = set(tmpl["EventId"].unique())
    event_ids  = set(df["event_id"].unique())
    missing_in_templates = event_ids - tmpl_ids
    if missing_in_templates:
        fail("template_coverage",
             f"{len(missing_in_templates)} EventIds in events missing from templates: "
             f"{list(missing_in_templates)[:5]}")
    else:
        ok("template_coverage", f"All {len(event_ids)} EventIds covered by templates")

    # --- 7. No duplicate (EventId, system) pairs in templates ---
    dup_tmpl = tmpl[tmpl.duplicated(subset=["EventId", "system"])][["EventId", "system"]].values.tolist()
    if dup_tmpl:
        fail("template_no_duplicates", f"Duplicate (EventId, system) pairs in templates: {dup_tmpl[:5]}")
    else:
        ok("template_no_duplicates", "No duplicate EventIds in templates")

    # --- Summary ---
    report["total_events"]    = len(df)
    report["total_templates"] = len(tmpl)
    report["sample_pct"]      = round(pct, 2)

    write_json(report, str(report_path))
    logger.info("Report written to %s", report_path)

    if report["passed"]:
        logger.info("All checks passed.")
    else:
        logger.error("%d check(s) FAILED: %s", len(report["errors"]), report["errors"])

    return report["passed"]


if __name__ == "__main__":
    passed = validate_quality()
    if not passed:
        sys.exit(1)
