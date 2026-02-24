"""TDD tests for Parquet operations."""

import os
import sys
import tempfile
from pathlib import Path
import pytest

DS4_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DS4_ROOT / "src"))


def test_write_raw_parquet_creates_file():
    """Verify write_raw_parquet creates a .parquet file."""
    from data.parquet_ops import write_raw_parquet

    records = [
        {
            "user_intent": "Scale deployment to 3 replicas",
            "tool_context": "scale_service, restart_pod",
            "workflow_json": '{"steps": [{"step_id": 1, "tool": "scale_service", "params": {"deployment": "web", "replicas": 3}}]}',
        },
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "raw_batch.parquet")
        write_raw_parquet(records, out_path)
        assert os.path.exists(out_path), "Parquet file was not created"


def test_read_raw_parquet_roundtrip():
    """Write then read, assert data equality."""
    from data.parquet_ops import write_raw_parquet, read_raw_parquet

    records = [
        {
            "user_intent": "Restart pod backend-1",
            "tool_context": "restart_pod",
            "workflow_json": '{"steps": [{"step_id": 1, "tool": "restart_pod", "params": {"pod": "backend-1"}}]}',
        },
        {
            "user_intent": "Scale api to 5",
            "tool_context": "scale_service",
            "workflow_json": '{"steps": [{"step_id": 1, "tool": "scale_service", "params": {"deployment": "api", "replicas": 5}}]}',
        },
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "raw_batch.parquet")
        write_raw_parquet(records, out_path)
        loaded = read_raw_parquet(out_path)

        assert len(loaded) == len(records)
        for orig, read in zip(records, loaded):
            assert orig["user_intent"] == read["user_intent"]
            assert orig["tool_context"] == read["tool_context"]
            assert orig["workflow_json"] == read["workflow_json"]


def test_parquet_schema_has_expected_columns():
    """Verify Parquet file has columns: user_intent, tool_context, workflow_json."""
    import pyarrow.parquet as pq
    from data.parquet_ops import write_raw_parquet

    records = [
        {
            "user_intent": "Test intent",
            "tool_context": "tool1, tool2",
            "workflow_json": '{"steps": []}',
        },
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "raw_batch.parquet")
        write_raw_parquet(records, out_path)

        table = pq.read_table(out_path)
        column_names = table.column_names

        assert "user_intent" in column_names
        assert "tool_context" in column_names
        assert "workflow_json" in column_names
