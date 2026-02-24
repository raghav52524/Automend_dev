"""Format raw Gemini output into Format B (ChatML JSONL)."""
import json
from pathlib import Path

from data.pipeline_logger import get_logger

logger = get_logger(__name__)


def raw_to_format_b_record(raw_item: dict) -> dict:
    """Convert one raw item (user_intent, tool_context, workflow) into a Format B message record."""
    tool_context = raw_item.get("tool_context", "")
    tools_desc = f"You are AutoMend. Available Tools: {tool_context}"
    workflow = raw_item["workflow"]
    if hasattr(workflow, "model_dump"):
        workflow = workflow.model_dump()
    workflow_json_str = json.dumps({"workflow": workflow})
    return {
        "messages": [
            {"role": "system", "content": tools_desc},
            {"role": "user", "content": raw_item["user_intent"]},
            {"role": "assistant", "content": workflow_json_str},
        ]
    }


def write_format_b_jsonl(raw_records: list[dict], out_path: str | Path) -> None:
    """Write a list of raw records to a JSONL file in Format B (one JSON object per line)."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(raw_records)
    logger.info("Writing Format B JSONL: %d records to %s", n, path)
    with path.open("w", encoding="utf-8") as f:
        for raw in raw_records:
            record = raw_to_format_b_record(raw)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Wrote %s", path)
