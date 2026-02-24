"""
Convert Dataset 1 (Alibaba) JSON output to Parquet format for Track A combiner.

Reads:  data/processed/ds1_alibaba/format_a_sequences.json
Output: data/interim/ds1_alibaba.parquet

This script is run after preprocess.py to convert the JSON output to Parquet,
which is the required format for Track A (Trigger Engine).
"""

import json
import logging
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.config.paths import get_ds1_processed_dir, get_ds1_interim_path, LOGS_DIR
    INPUT_PATH = get_ds1_processed_dir() / "format_a_sequences.json"
    OUTPUT_PATH = get_ds1_interim_path()
except ImportError:
    INPUT_PATH = SCRIPT_DIR.parent / "data" / "processed" / "format_a_sequences.json"
    OUTPUT_PATH = PROJECT_ROOT / "data" / "interim" / "ds1_alibaba.parquet"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)


def convert_to_parquet():
    """Convert JSON sequences to Parquet format."""
    log.info(f"Reading JSON from: {INPUT_PATH}")
    
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")
    
    with open(INPUT_PATH, "r") as f:
        sequences = json.load(f)
    
    log.info(f"Loaded {len(sequences)} sequences")
    
    df = pd.DataFrame(sequences)
    
    log.info(f"DataFrame shape: {df.shape}")
    log.info(f"Columns: {df.columns.tolist()}")
    log.info(f"Sample row:\n{df.iloc[0].to_dict()}")
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(OUTPUT_PATH, engine="pyarrow", index=False)
    
    log.info(f"Saved Parquet to: {OUTPUT_PATH}")
    log.info(f"File size: {OUTPUT_PATH.stat().st_size / 1024:.2f} KB")
    
    return df


if __name__ == "__main__":
    df = convert_to_parquet()
    print(f"\nConversion complete!")
    print(f"Total rows: {len(df)}")
    print(f"Output: {OUTPUT_PATH}")
