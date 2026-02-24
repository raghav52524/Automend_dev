"""
Unit tests for schema validation pipeline.
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from schema_validation import run_validation


def _find_key(results: dict, *patterns: str) -> str:
    """Find a key in results that contains any of the given patterns."""
    for key in results:
        for pattern in patterns:
            if pattern in key:
                return key
    return None


@pytest.fixture
def valid_df():
    """Fixture providing a valid DataFrame matching expected schema."""
    size = 5000
    return pd.DataFrame({
        "system":                    ["SYSTEM: test"] * size,
        "chat":                      ["USER: hello ASSISTANT: hi"] * size,
        "num_turns":                 [1] * size,
        "num_calls":                 [1] * size,
        "complexity_tier":           ["simple"] * size,
        "has_parallel":              [False] * size,
        "has_malformed":             [False] * size,
        "function_calls":            ["[]"] * size,
        "num_defined_functions":     [1] * size,
        "defined_function_names":    ["[]"] * size,
        "function_signatures":       ["{}"] * size,
        "has_error_handling":        [False] * size,
        "has_function_error_response": [False] * size,
        "has_conditional_error":     [False] * size,
        "error_keywords_found":      ["[]"] * size,
    })


class TestRunValidation:

    def test_valid_df_passes_all(self, valid_df):
        """Test valid DataFrame passes all expectations."""
        results = run_validation(valid_df)
        assert all(results.values())

    def test_returns_dict(self, valid_df):
        """Test validation returns a dictionary."""
        results = run_validation(valid_df)
        assert isinstance(results, dict)

    def test_all_expected_keys_present(self, valid_df):
        """Test all expected expectation keys are in results.
        
        Handles both GE and simple validation key naming conventions:
        - GE: 'row_count', 'complexity_tier_values'
        - Simple: 'row_count_range', 'values_in_set_complexity_tier'
        """
        results = run_validation(valid_df)
        # Row count check (GE: 'row_count', simple: 'row_count_range')
        row_key = _find_key(results, "row_count")
        assert row_key is not None, f"No row count key found in {list(results.keys())}"
        
        # Complexity tier check (GE: 'complexity_tier_values', simple: 'values_in_set_complexity_tier')
        complexity_key = _find_key(results, "complexity_tier")
        assert complexity_key is not None, f"No complexity tier key found in {list(results.keys())}"
        
        # Null check (both use 'no_nulls_chat')
        assert "no_nulls_chat" in results

    def test_invalid_complexity_fails(self, valid_df):
        """Test DataFrame with invalid complexity tier values fails."""
        valid_df["complexity_tier"] = "invalid_tier"
        results = run_validation(valid_df)
        # Find the complexity tier validation key (varies by validation method)
        complexity_key = _find_key(results, "complexity_tier_values", "values_in_set_complexity_tier")
        assert complexity_key is not None, f"No complexity tier key found in {list(results.keys())}"
        assert results[complexity_key] is False or results[complexity_key] == False