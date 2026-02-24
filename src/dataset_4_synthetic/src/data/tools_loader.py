"""Load the list of available tools from a JSON file (single source for all prompts)."""
import json
from pathlib import Path


def load_available_tools(path: str | Path) -> list[str]:
    """Load tool names from a JSON array file. Returns e.g. ['scale_service', 'restart_pod']."""
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array of tool names, got {type(data)}")
    return [str(t) for t in data]
