"""Generate data schema validation and statistics using Great Expectations.

Reads the labeled events parquet, validates schema/values using GE,
and produces a combined statistics + GE report.

Output: data/processed/ds2_loghub/mlops_processed/statistics_report.json
Exits with code 1 if GE schema validation fails.
"""
import sys
import json
from pathlib import Path

DS2_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DS2_SRC))
from utils.paths import get_ds2_processed_dir

import pandas as pd
from utils.io import read_parquet, write_json
from utils.logger import get_logger

logger = get_logger(__name__)

PROCESSED_DIR = get_ds2_processed_dir()
EVENTS_PATH = PROCESSED_DIR / "mlops_processed" / "mlops_events.parquet"
REPORT_PATH = PROCESSED_DIR / "mlops_processed" / "statistics_report.json"

REQUIRED_COLS = [
    "system", "timestamp", "severity", "source",
    "event_id", "event_template", "message", "raw_id", "extras", "event_type",
]
ALLOWED_SYSTEMS    = ["linux", "hpc", "hdfs", "hadoop", "spark"]
ALLOWED_SEVERITIES = ["INFO", "WARN", "ERROR"]
ALLOWED_EVENT_TYPES = [
    "auth_failure", "permission_denied", "storage_unavailable",
    "data_ingestion_failed", "compute_oom", "executor_failure",
    "network_issue", "job_failed", "system_crash", "normal_ops", "unknown",
]


def _create_ge_dataframe(df: pd.DataFrame):
    """
    Create a GE-compatible DataFrame object.
    Handles API differences between GE v0.x and v1.x.
    """
    import great_expectations as ge
    
    # Try legacy API first (most existing code uses this)
    try:
        return ge.from_pandas(df)
    except AttributeError:
        logger.debug("ge.from_pandas() not available, trying PandasDataset")
    
    # Try PandasDataset directly
    try:
        from great_expectations.dataset import PandasDataset
        return PandasDataset(df)
    except ImportError:
        logger.debug("PandasDataset not available")
    
    raise RuntimeError("Failed to create GE DataFrame - API not compatible")


def _run_ge_validation(df: pd.DataFrame) -> dict:
    """Run Great Expectations validation and return result dict."""
    try:
        import great_expectations as ge

        df_ge = _create_ge_dataframe(df)

        # Schema: all required columns present
        df_ge.expect_table_columns_to_match_ordered_list(REQUIRED_COLS)

        # Value set checks
        df_ge.expect_column_values_to_be_in_set("system", ALLOWED_SYSTEMS)
        df_ge.expect_column_values_to_be_in_set("severity", ALLOWED_SEVERITIES)
        df_ge.expect_column_values_to_be_in_set("event_type", ALLOWED_EVENT_TYPES)

        # Null checks
        for col in ["event_id", "event_template", "message", "raw_id", "system"]:
            df_ge.expect_column_values_to_not_be_null(col)

        # EventId format
        df_ge.expect_column_values_to_match_regex("event_id", r"^E\d+$")

        # Row count > 0
        df_ge.expect_table_row_count_to_be_between(min_value=1)

        result = df_ge.validate()
        ge_success = bool(result["success"])
        ge_results = []
        for r in result["results"]:
            ge_results.append({
                "expectation": r["expectation_config"]["expectation_type"],
                "success": r["success"],
                "column": r["expectation_config"].get("kwargs", {}).get("column", "table"),
            })

        logger.info("GE validation: %s (%d expectations)",
                    "PASSED" if ge_success else "FAILED", len(ge_results))
        return {"success": ge_success, "results": ge_results}

    except ImportError:
        logger.warning("great-expectations not installed; skipping GE validation")
        return {"success": None, "results": [], "note": "great-expectations not installed"}
    except RuntimeError as exc:
        logger.warning("GE API not compatible: %s; skipping GE validation", exc)
        return {"success": None, "results": [], "note": str(exc)}
    except Exception as exc:
        logger.error("GE validation error: %s", exc)
        return {"success": False, "results": [], "error": str(exc)}


def generate_statistics(events_path: Path = EVENTS_PATH,
                        report_path: Path = REPORT_PATH) -> dict:
    logger.info("Loading %s", events_path)
    df = read_parquet(str(events_path))

    # ── Great Expectations schema validation ─────────────────────────────────
    ge_result = _run_ge_validation(df)

    # ── Pandas statistics ─────────────────────────────────────────────────────
    null_rates = {
        col: round(df[col].isnull().sum() / len(df) * 100, 2)
        for col in df.columns
    }

    stats = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "rows_per_system": df["system"].value_counts().to_dict(),
        "severity_distribution": df["severity"].value_counts().to_dict(),
        "event_type_distribution": df["event_type"].value_counts().to_dict(),
        "null_rates_pct": null_rates,
        "unique_event_ids": int(df["event_id"].nunique()),
        "unique_event_templates": int(df["event_template"].nunique()),
    }

    logger.info("Total rows: %d | Systems: %s | Unique EventIds: %d",
                stats["total_rows"],
                list(stats["rows_per_system"].keys()),
                stats["unique_event_ids"])

    report = {
        "ge_validation": ge_result,
        "statistics": stats,
    }

    write_json(report, str(report_path))
    logger.info("Statistics report written to %s", report_path)

    # Fail pipeline if GE found schema violations
    if ge_result["success"] is False:
        logger.error("GE schema validation FAILED — check statistics_report.json")
        return report

    return report


if __name__ == "__main__":
    result = generate_statistics()
    if result["ge_validation"]["success"] is False:
        sys.exit(1)
