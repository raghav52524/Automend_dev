"""
Schema Validation for Glaive Function Calling v2
Uses Great Expectations to validate processed data quality.
Falls back to simple validation if GE is unavailable.
"""

import json
import logging
import sys
from pathlib import Path

import pandas as pd

# Add project root to path for centralized utilities
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.ge_utils import (
    is_ge_available,
    create_ge_dataframe,
    validate_dataframe_simple,
    get_ge_version,
)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Config
PROCESSED_FILE = Path(__file__).resolve().parents[1] / "data" / "processed" / "glaive_processed.jsonl"
VALIDATION_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "validation"

# Schema definition for simple validation fallback
GLAIVE_SCHEMA = {
    "required_columns": [
        "system", "chat", "num_turns", "num_calls",
        "complexity_tier", "has_parallel", "has_malformed",
        "function_calls", "num_defined_functions",
        "defined_function_names", "function_signatures",
        "has_error_handling", "has_function_error_response",
        "has_conditional_error", "error_keywords_found"
    ],
    "not_null_columns": ["chat", "num_turns", "num_calls", "complexity_tier"],
    "column_value_sets": {
        "complexity_tier": ["none", "simple", "moderate", "complex", "malformed"],
        "has_parallel": [True, False],
        "has_malformed": [True, False],
        "has_error_handling": [True, False],
        "has_function_error_response": [True, False],
        "has_conditional_error": [True, False],
    },
    "numeric_ranges": {
        "num_turns": (0, 50),
        "num_calls": (0, 20),
        "num_defined_functions": (0, 20),
    },
    "row_count_range": (4000, 6000),
}


def load_processed_data(filepath: Path) -> pd.DataFrame:
    """Load processed JSONL into a DataFrame."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)


def _run_ge_validation(df: pd.DataFrame) -> dict:
    """
    Run Great Expectations validation suite on processed DataFrame.
    Returns validation results summary.
    """
    logger.info("Converting DataFrame to GE dataset...")
    ge_df, api_type = create_ge_dataframe(df)
    logger.info(f"Using GE API type: {api_type} (version: {get_ge_version()})")
    
    if api_type == "v1":
        logger.warning("GE v1.x detected - falling back to simple validation")
        return None

    results = {}

    # 1. Required columns exist
    logger.info("Validating required columns...")
    required_columns = GLAIVE_SCHEMA["required_columns"]
    for col in required_columns:
        result = ge_df.expect_column_to_exist(col)
        results[f"column_exists_{col}"] = result["success"]

    # 2. No nulls in critical columns
    logger.info("Validating no nulls in critical columns...")
    critical_cols = GLAIVE_SCHEMA["not_null_columns"]
    for col in critical_cols:
        result = ge_df.expect_column_values_to_not_be_null(col)
        results[f"no_nulls_{col}"] = result["success"]

    # 3. Numeric column ranges
    logger.info("Validating numeric ranges...")
    for col, (min_val, max_val) in GLAIVE_SCHEMA["numeric_ranges"].items():
        result = ge_df.expect_column_values_to_be_between(
            col, min_value=min_val, max_value=max_val
        )
        results[f"{col}_range"] = result["success"]

    # 4. Complexity tier values
    logger.info("Validating complexity tier values...")
    result = ge_df.expect_column_values_to_be_in_set(
        "complexity_tier",
        GLAIVE_SCHEMA["column_value_sets"]["complexity_tier"]
    )
    results["complexity_tier_values"] = result["success"]

    # 5. Boolean columns
    logger.info("Validating boolean columns...")
    bool_cols = [
        "has_parallel", "has_malformed",
        "has_error_handling", "has_function_error_response",
        "has_conditional_error"
    ]
    for col in bool_cols:
        result = ge_df.expect_column_values_to_be_in_set(col, [True, False])
        results[f"bool_values_{col}"] = result["success"]

    # 6. Chat field not empty
    logger.info("Validating chat field not empty...")
    result = ge_df.expect_column_value_lengths_to_be_between("chat", min_value=10)
    results["chat_not_empty"] = result["success"]

    # 7. Dataset size
    logger.info("Validating dataset size...")
    min_rows, max_rows = GLAIVE_SCHEMA["row_count_range"]
    result = ge_df.expect_table_row_count_to_be_between(
        min_value=min_rows, max_value=max_rows
    )
    results["row_count"] = result["success"]

    return results


def _run_simple_validation(df: pd.DataFrame) -> dict:
    """
    Run simple validation without Great Expectations.
    Used as fallback when GE is unavailable or fails.
    """
    logger.info("Running simple validation (GE fallback)...")
    
    result = validate_dataframe_simple(df, GLAIVE_SCHEMA)
    
    # Convert to same format as GE validation
    results = {}
    for r in result["results"]:
        results[r["check"]] = r["success"]
    
    # Add chat length check (not covered by simple validation)
    if "chat" in df.columns:
        chat_lengths = df["chat"].astype(str).str.len()
        results["chat_not_empty"] = bool((chat_lengths >= 10).all())
    
    return results


def run_validation(df: pd.DataFrame) -> dict:
    """
    Run validation suite on processed DataFrame.
    Tries Great Expectations first, falls back to simple validation.
    Returns validation results summary.
    """
    if is_ge_available():
        try:
            results = _run_ge_validation(df)
            if results is not None:
                return results
            logger.info("GE validation returned None, using simple validation")
        except Exception as e:
            logger.warning(f"GE validation failed: {e}, falling back to simple validation")
    else:
        logger.info("Great Expectations not available, using simple validation")
    
    return _run_simple_validation(df)


def print_validation_report(results: dict) -> None:
    """Print a clean validation report."""
    passed = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)
    total = len(results)

    print("\n" + "="*55)
    print("          DATA VALIDATION REPORT")
    print("="*55)

    # Print failed ones first
    failures = {k: v for k, v in results.items() if not v}
    if failures:
        print("\n FAILED CHECKS:")
        for name in failures:
            print(f"   - {name}")

    print(f"\n Passed: {passed}/{total}")
    print(f" Failed: {failed}/{total}")
    print(f" Success Rate: {passed/total*100:.1f}%")
    print("="*55)


def save_validation_report(results: dict) -> None:
    """Save validation results to JSON file."""
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    report_path = VALIDATION_DIR / "validation_report.json"

    report = {
        "total_expectations": len(results),
        "passed": sum(1 for v in results.values() if v),
        "failed": sum(1 for v in results.values() if not v),
        "success_rate": sum(1 for v in results.values() if v) / len(results),
        "results": results
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Validation report saved to %s", report_path)


if __name__ == "__main__":
    logger.info("Loading processed data...")
    df = load_processed_data(PROCESSED_FILE)
    logger.info("Loaded %d records", len(df))

    results = run_validation(df)
    print_validation_report(results)
    save_validation_report(results)
