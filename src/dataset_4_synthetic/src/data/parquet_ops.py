"""Parquet I/O operations for raw batch data."""

import pyarrow as pa
import pyarrow.parquet as pq

from data.pipeline_logger import get_logger

logger = get_logger(__name__)


def write_raw_parquet(records: list[dict], out_path: str) -> None:
    """
    Write raw batch records to a Parquet file.

    Args:
        records: List of dicts with keys: user_intent, tool_context, workflow_json
        out_path: Path to write the .parquet file
    """
    if not records:
        logger.warning("No records to write to Parquet")
        return

    user_intents = [r["user_intent"] for r in records]
    tool_contexts = [r["tool_context"] for r in records]
    workflow_jsons = [r["workflow_json"] for r in records]

    table = pa.table({
        "user_intent": user_intents,
        "tool_context": tool_contexts,
        "workflow_json": workflow_jsons,
    })

    pq.write_table(table, out_path)
    logger.info("Wrote %d records to Parquet: %s", len(records), out_path)


def read_raw_parquet(path: str) -> list[dict]:
    """
    Read raw batch records from a Parquet file.

    Args:
        path: Path to the .parquet file

    Returns:
        List of dicts with keys: user_intent, tool_context, workflow_json
    """
    table = pq.read_table(path)
    records = table.to_pylist()
    logger.info("Read %d records from Parquet: %s", len(records), path)
    return records
