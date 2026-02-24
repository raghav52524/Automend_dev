"""Phase 5: Preprocessor tests - raw to Format B JSONL."""
import sys
import json
import tempfile
from pathlib import Path

import pytest

DS4_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DS4_ROOT / "src"))

from data import preprocessor
from schemas.workflow_schema import Workflow


def test_format_b_record_has_messages_with_three_roles():
    """Output has system, user, assistant messages."""
    raw_item = {
        "user_intent": "Fix the latency on the fraud model.",
        "tool_context": "scale_service, restart_pod",
        "workflow": {"steps": [{"step_id": 1, "tool": "scale_service", "params": {"deployment": "fraud-model", "replicas": 5}}]},
    }
    record = preprocessor.raw_to_format_b_record(raw_item)
    assert "messages" in record
    assert len(record["messages"]) == 3
    roles = [m["role"] for m in record["messages"]]
    assert roles == ["system", "user", "assistant"]


def test_format_b_assistant_content_is_workflow_json_string():
    """Assistant content is a JSON string containing workflow with steps."""
    raw_item = {
        "user_intent": "Restart pod",
        "tool_context": "restart_pod",
        "workflow": {"steps": [{"step_id": 1, "tool": "restart_pod", "params": {"pod": "fraud-model"}}]},
    }
    record = preprocessor.raw_to_format_b_record(raw_item)
    assistant_msg = next(m for m in record["messages"] if m["role"] == "assistant")
    content = assistant_msg["content"]
    parsed = json.loads(content)
    assert "workflow" in parsed
    assert "steps" in parsed["workflow"]
    assert parsed["workflow"]["steps"][0]["tool"] == "restart_pod"


def test_write_format_b_jsonl_writes_one_line_per_record():
    """JSONL file has one JSON object per line."""
    raw_records = [
        {
            "user_intent": "Fix latency",
            "tool_context": "scale_service",
            "workflow": {"steps": [{"step_id": 1, "tool": "scale_service", "params": {}}]},
        },
        {
            "user_intent": "Restart",
            "tool_context": "restart_pod",
            "workflow": {"steps": []},
        },
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        out_path = f.name
    try:
        preprocessor.write_format_b_jsonl(raw_records, out_path)
        lines = Path(out_path).read_text().strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            obj = json.loads(line)
            assert "messages" in obj
    finally:
        Path(out_path).unlink(missing_ok=True)
