"""Merge all normalized parquet files into a single unified events dataset.

Currently uses 100% of the data. To switch to deterministic 10% sampling,
uncomment the keep_row logic below (stable_hash % 10 == 0).

Output: data/processed/ds2_loghub/mlops_processed/mlops_events.parquet
"""
import sys
from pathlib import Path

DS2_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DS2_SRC))
from utils.paths import get_ds2_processed_dir

import pandas as pd
from utils.io import read_parquet, write_parquet
from utils.hashing import stable_hash
from utils.logger import get_logger

logger = get_logger(__name__)

PROCESSED_DIR = get_ds2_processed_dir()

NORMALIZED_DIR = PROCESSED_DIR / "normalized"
OUTPUT         = PROCESSED_DIR / "mlops_processed" / "mlops_events.parquet"

SYSTEMS = ["linux", "hpc", "hdfs", "hadoop", "spark"]


def merge_events(normalized_dir: Path = NORMALIZED_DIR, output_path: Path = OUTPUT):
    # 1. Merge all normalized files
    frames = []
    for system in SYSTEMS:
        path = normalized_dir / f"{system}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing normalized file: {path}")
        df = read_parquet(str(path))
        frames.append(df)
        logger.info("Loaded %s: %d rows", system, len(df))

    merged = pd.concat(frames, ignore_index=True)
    logger.info("Total merged: %d rows", len(merged))

    # 2. Use 100% of data (10% sampling commented out)
    # def keep_row(row) -> bool:
    #     key = f"{row['system']}:{row['raw_id']}"
    #     return stable_hash(key) % 10 == 0
    # mask = merged.apply(keep_row, axis=1)
    # sampled = merged[mask].copy().reset_index(drop=True)

    sampled = merged.copy().reset_index(drop=True)

    pct = len(sampled) / len(merged) * 100
    logger.info("Sampled: %d rows (%.1f%%)", len(sampled), pct)

    # 3. Per-system counts
    logger.info("Per-system sample counts:")
    for system, count in sampled["system"].value_counts().sort_index().items():
        logger.info("  %s: %d", system, count)

    write_parquet(sampled, str(output_path))
    logger.info("Done.")
    return sampled


if __name__ == "__main__":
    merge_events()
