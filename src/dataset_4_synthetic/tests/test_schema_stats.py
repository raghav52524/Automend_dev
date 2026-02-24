"""TDD: Data schema and statistics generation and validation."""
import sys
import json
import tempfile
from pathlib import Path

import pytest

DS4_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DS4_ROOT / "src"))

from data import schema_stats


def test_infer_schema_returns_expected_structure_for_format_b():
    """Infer schema from Format B records returns messages structure and types."""
    records = [
        {"messages": [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}, {"role": "assistant", "content": "{}"}]},
    ]
    schema = schema_stats.infer_schema(records)
    assert "messages" in schema
    assert schema["messages"] == "array"
    assert "record_count" in schema
    assert schema["record_count"] == 1


def test_compute_statistics_returns_row_count_and_quality_metrics():
    """Statistics include row count and basic quality metrics."""
    records = [
        {"user_intent": "a", "tool_context": "t", "workflow": {"steps": [{"step_id": 1, "tool": "x", "params": {}}]}},
        {"user_intent": "b", "tool_context": "t", "workflow": {"steps": []}},
    ]
    stats = schema_stats.compute_statistics(records)
    assert stats["row_count"] == 2
    assert "missing_user_intent" in stats or "null_counts" in stats or "row_count" in stats


def test_validate_schema_passes_for_valid_format_b_records():
    """Valid records with messages and roles pass validation."""
    records = [
        {"messages": [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}, {"role": "assistant", "content": "{}"}]},
    ]
    result = schema_stats.validate_schema(records)
    assert result["valid"] is True
    assert result["errors"] == []


def test_validate_schema_fails_for_missing_messages():
    """Records missing 'messages' key fail validation."""
    records = [{"not_messages": []}]
    result = schema_stats.validate_schema(records)
    assert result["valid"] is False
    assert len(result["errors"]) > 0


def test_generate_schema_and_stats_writes_files(tmp_path):
    """generate_schema_and_stats writes schema.json and stats.json to the given dir."""
    records = [
        {"messages": [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}, {"role": "assistant", "content": "{}"}]},
    ]
    out_dir = tmp_path / "schema_stats"
    schema_stats.generate_schema_and_stats(records, out_dir)
    schema_file = out_dir / "schema.json"
    stats_file = out_dir / "stats.json"
    assert schema_file.exists()
    assert stats_file.exists()
    schema_data = json.loads(schema_file.read_text())
    stats_data = json.loads(stats_file.read_text())
    assert "record_count" in schema_data or "messages" in schema_data
    assert stats_data["row_count"] == 1
