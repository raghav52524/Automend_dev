"""
Export Dataset 3 (StackOverflow) output to the standardized interim location.

Reads:  data/processed/ds3_stackoverflow/training/training_data.jsonl
Output: data/interim/ds3_stackoverflow.jsonl

This script copies the training JSONL to the monorepo's data/interim/ folder
for use by the Track B combiner.
"""

import shutil
import logging
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.config.paths import get_ds3_processed_dir, get_ds3_interim_path
    INPUT_PATH = get_ds3_processed_dir() / "training" / "training_data.jsonl"
    OUTPUT_PATH = get_ds3_interim_path()
except ImportError:
    DATASET_DIR = SCRIPT_DIR.parent
    INPUT_PATH = DATASET_DIR / "data" / "training" / "training_data.jsonl"
    OUTPUT_PATH = PROJECT_ROOT / "data" / "interim" / "ds3_stackoverflow.jsonl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)


def export_to_interim():
    """Copy the training JSONL to the interim directory."""
    log.info(f"Source: {INPUT_PATH}")
    log.info(f"Destination: {OUTPUT_PATH}")
    
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    shutil.copy2(INPUT_PATH, OUTPUT_PATH)
    
    log.info(f"Exported to: {OUTPUT_PATH}")
    log.info(f"File size: {OUTPUT_PATH.stat().st_size / 1024:.2f} KB")


if __name__ == "__main__":
    export_to_interim()
    print(f"Export complete: {OUTPUT_PATH}")
