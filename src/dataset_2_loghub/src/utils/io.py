"""I/O helpers for reading/writing pipeline data."""
import sys
import json
from pathlib import Path

import pandas as pd

DS2_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DS2_SRC))

from utils.logger import get_logger

logger = get_logger(__name__)


def read_csv(path: str, **kwargs) -> pd.DataFrame:
    """Read a CSV with sensible defaults (all columns as str by default)."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, **kwargs)


def write_parquet(df: pd.DataFrame, path: str) -> None:
    """Write a DataFrame to Parquet, creating parent dirs as needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Written %d rows → %s", len(df), path)


def read_parquet(path: str) -> pd.DataFrame:
    """Read a Parquet file."""
    return pd.read_parquet(path)


def write_csv(df: pd.DataFrame, path: str) -> None:
    """Write a DataFrame to CSV, creating parent dirs as needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Written %d rows → %s", len(df), path)


def write_json(data: dict, path: str) -> None:
    """Write a dict as JSON, creating parent dirs as needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info("Written JSON → %s", path)