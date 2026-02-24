"""Data schema inference, statistics, and validation for pipeline monitoring."""
import json
from pathlib import Path


def infer_schema(records: list[dict]) -> dict:
    """Infer a simple schema from a list of records (e.g. Format B with 'messages')."""
    if not records:
        return {"record_count": 0}
    schema = {}
    first = records[0]
    if isinstance(first, dict):
        if "messages" in first:
            schema["messages"] = "array"
        for key in first:
            if key not in schema:
                val = first[key]
                if isinstance(val, list):
                    schema[key] = "array"
                elif isinstance(val, dict):
                    schema[key] = "object"
                elif isinstance(val, str):
                    schema[key] = "string"
                elif isinstance(val, (int, float)):
                    schema[key] = "number"
                else:
                    schema[key] = type(val).__name__
    schema["record_count"] = len(records)
    return schema


def compute_statistics(records: list[dict]) -> dict:
    """Compute row count and basic data quality metrics."""
    stats = {"row_count": len(records)}
    if not records:
        return stats
    missing_intent = 0
    missing_messages = 0
    for r in records:
        if isinstance(r, dict):
            if "user_intent" in r and (r.get("user_intent") is None or r.get("user_intent") == ""):
                missing_intent += 1
            if "messages" in r:
                if not r["messages"] or len(r["messages"]) < 3:
                    missing_messages += 1
            else:
                missing_messages += 1
    stats["missing_user_intent"] = missing_intent
    stats["records_missing_valid_messages"] = missing_messages
    return stats


def validate_schema(records: list[dict]) -> dict:
    """Validate that records conform to expected Format B–like structure. Returns {valid: bool, errors: list}."""
    errors = []
    for i, r in enumerate(records):
        if not isinstance(r, dict):
            errors.append(f"Record {i}: not a dict")
            continue
        if "messages" not in r:
            errors.append(f"Record {i}: missing 'messages'")
            continue
        msgs = r["messages"]
        if not isinstance(msgs, list) or len(msgs) < 3:
            errors.append(f"Record {i}: 'messages' must be array of at least 3 (system, user, assistant)")
            continue
        roles = {m.get("role") for m in msgs if isinstance(m, dict)}
        if "system" not in roles or "user" not in roles or "assistant" not in roles:
            errors.append(f"Record {i}: messages must include system, user, assistant roles")
    return {"valid": len(errors) == 0, "errors": errors}


def generate_schema_and_stats(records: list[dict], out_dir: str | Path) -> None:
    """Write schema.json and stats.json to out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    schema = infer_schema(records)
    stats = compute_statistics(records)
    (out_dir / "schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
