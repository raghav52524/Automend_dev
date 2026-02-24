"""
Great Expectations Compatibility Layer
======================================
Centralized utilities for Great Expectations that handle API differences
between GE v0.x and v1.x versions.

This module provides a consistent interface for data validation regardless
of which GE version is installed.
"""

import logging
from typing import Any, Tuple, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Detect GE availability and version
_GE_VERSION = None
_GE_AVAILABLE = False
_GE_API = None  # 'v1', 'legacy', or None

try:
    import great_expectations as ge
    _GE_AVAILABLE = True
    _GE_VERSION = getattr(ge, "__version__", "unknown")
    
    # Check which API is available
    if hasattr(ge, "get_context"):
        _GE_API = "v1"
    elif hasattr(ge, "from_pandas"):
        _GE_API = "legacy"
    else:
        # Try importing PandasDataset as last resort
        try:
            from great_expectations.dataset import PandasDataset
            _GE_API = "pandas_dataset"
        except ImportError:
            _GE_API = None
            
except ImportError:
    pass


def is_ge_available() -> bool:
    """Check if Great Expectations is installed and usable."""
    return _GE_AVAILABLE and _GE_API is not None


def get_ge_version() -> Optional[str]:
    """Get the installed Great Expectations version."""
    return _GE_VERSION


def get_ge_api() -> Optional[str]:
    """Get the detected GE API type: 'v1', 'legacy', 'pandas_dataset', or None."""
    return _GE_API


def create_ge_dataframe(df: pd.DataFrame) -> Tuple[Any, str]:
    """
    Create a GE-compatible DataFrame object for validation.
    
    Handles API differences between GE v0.x and v1.x:
    1. Tries legacy ge.from_pandas() first (most compatible with existing code)
    2. Falls back to PandasDataset() if from_pandas is unavailable
    3. For v1.x, returns a validator object instead
    
    Args:
        df: pandas DataFrame to wrap
        
    Returns:
        Tuple of (ge_dataframe_or_validator, api_type)
        - api_type is one of: 'legacy', 'pandas_dataset', 'v1'
        
    Raises:
        RuntimeError: If GE is not available or all methods fail
    """
    if not _GE_AVAILABLE:
        raise RuntimeError("Great Expectations is not installed")
    
    import great_expectations as ge
    
    # Try legacy API first (most existing code uses this)
    try:
        ge_df = ge.from_pandas(df)
        logger.debug("Created GE DataFrame using ge.from_pandas()")
        return ge_df, "legacy"
    except AttributeError:
        logger.debug("ge.from_pandas() not available, trying PandasDataset")
    except Exception as e:
        logger.debug(f"ge.from_pandas() failed: {e}, trying PandasDataset")
    
    # Try PandasDataset directly
    try:
        from great_expectations.dataset import PandasDataset
        ge_df = PandasDataset(df)
        logger.debug("Created GE DataFrame using PandasDataset()")
        return ge_df, "pandas_dataset"
    except ImportError:
        logger.debug("PandasDataset not available")
    except Exception as e:
        logger.debug(f"PandasDataset() failed: {e}")
    
    # For v1.x, we need a different approach - return context info
    if _GE_API == "v1":
        try:
            context = ge.get_context()
            logger.debug("GE v1.x detected, returning context for validator-based validation")
            return context, "v1"
        except Exception as e:
            logger.debug(f"ge.get_context() failed: {e}")
    
    raise RuntimeError(
        f"Failed to create GE DataFrame. GE version: {_GE_VERSION}, "
        f"detected API: {_GE_API}. Please check your Great Expectations installation."
    )


def run_legacy_expectations(ge_df: Any, expectations: list) -> dict:
    """
    Run expectations on a legacy GE DataFrame (from from_pandas or PandasDataset).
    
    Args:
        ge_df: GE DataFrame object
        expectations: List of dicts with 'method' and 'kwargs' keys
                     e.g. [{'method': 'expect_column_to_exist', 'kwargs': {'column': 'id'}}]
    
    Returns:
        Dict with 'success' (bool) and 'results' (list of individual results)
    """
    results = []
    all_success = True
    
    for exp in expectations:
        method_name = exp.get("method")
        kwargs = exp.get("kwargs", {})
        
        try:
            method = getattr(ge_df, method_name)
            result = method(**kwargs)
            success = result.get("success", False) if isinstance(result, dict) else bool(result)
            results.append({
                "expectation": method_name,
                "kwargs": kwargs,
                "success": success,
                "result": result if isinstance(result, dict) else {"success": success}
            })
            if not success:
                all_success = False
        except Exception as e:
            logger.warning(f"Expectation {method_name} failed: {e}")
            results.append({
                "expectation": method_name,
                "kwargs": kwargs,
                "success": False,
                "error": str(e)
            })
            all_success = False
    
    return {"success": all_success, "results": results}


def validate_dataframe_simple(df: pd.DataFrame, schema: dict) -> dict:
    """
    Simple DataFrame validation without Great Expectations.
    
    Use this as a fallback when GE is unavailable or fails.
    
    Args:
        df: DataFrame to validate
        schema: Dict with validation rules:
            - required_columns: list of column names that must exist
            - not_null_columns: list of columns that cannot have nulls
            - column_value_sets: dict mapping column -> list of allowed values
            - row_count_range: tuple of (min, max) row counts
            
    Returns:
        Dict with 'success' (bool), 'results' (list), and 'summary'
    """
    results = []
    all_success = True
    
    # Check required columns
    required_cols = schema.get("required_columns", [])
    for col in required_cols:
        exists = col in df.columns
        results.append({
            "check": f"column_exists_{col}",
            "success": exists,
            "message": f"Column '{col}' exists" if exists else f"Column '{col}' is missing"
        })
        if not exists:
            all_success = False
    
    # Check not-null columns
    not_null_cols = schema.get("not_null_columns", [])
    for col in not_null_cols:
        if col not in df.columns:
            continue
        null_count = df[col].isna().sum()
        success = null_count == 0
        results.append({
            "check": f"no_nulls_{col}",
            "success": success,
            "message": f"Column '{col}' has no nulls" if success else f"Column '{col}' has {null_count} null values"
        })
        if not success:
            all_success = False
    
    # Check column value sets
    value_sets = schema.get("column_value_sets", {})
    for col, allowed_values in value_sets.items():
        if col not in df.columns:
            continue
        invalid = ~df[col].isin(allowed_values) & df[col].notna()
        invalid_count = invalid.sum()
        success = invalid_count == 0
        results.append({
            "check": f"values_in_set_{col}",
            "success": success,
            "message": f"Column '{col}' values are valid" if success else f"Column '{col}' has {invalid_count} invalid values"
        })
        if not success:
            all_success = False
    
    # Check row count range
    row_range = schema.get("row_count_range")
    if row_range:
        min_rows, max_rows = row_range
        row_count = len(df)
        success = min_rows <= row_count <= max_rows
        results.append({
            "check": "row_count_range",
            "success": success,
            "message": f"Row count {row_count} is within [{min_rows}, {max_rows}]" if success 
                      else f"Row count {row_count} is outside [{min_rows}, {max_rows}]"
        })
        if not success:
            all_success = False
    
    # Check numeric column ranges
    numeric_ranges = schema.get("numeric_ranges", {})
    for col, (min_val, max_val) in numeric_ranges.items():
        if col not in df.columns:
            continue
        col_min = df[col].min()
        col_max = df[col].max()
        success = col_min >= min_val and col_max <= max_val
        results.append({
            "check": f"range_{col}",
            "success": success,
            "message": f"Column '{col}' values in range" if success 
                      else f"Column '{col}' has values outside [{min_val}, {max_val}]"
        })
        if not success:
            all_success = False
    
    passed = sum(1 for r in results if r["success"])
    failed = len(results) - passed
    
    return {
        "success": all_success,
        "results": results,
        "summary": {
            "total": len(results),
            "passed": passed,
            "failed": failed,
            "success_rate": passed / len(results) if results else 1.0
        }
    }
