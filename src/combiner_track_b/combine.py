"""
Track B Combiner - Generative Architect
=======================================
Combines Dataset 3-6 JSONL files into a single combined JSONL file
for the Generative Architect ML track.

Input files (from data/interim/):
    - ds3_stackoverflow.jsonl
    - ds4_synthetic.jsonl
    - ds5_glaive.jsonl
    - ds6_the_stack.jsonl

Output file (to data/processed/):
    - track_B_combined.jsonl

Format B schema (ChatML):
    {"messages": [{"role": str, "content": str}, ...], ...}
"""

import json
import logging
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

try:
    from src.config.paths import (
        get_ds3_interim_path, get_ds4_interim_path,
        get_ds5_interim_path, get_ds6_interim_path,
        get_track_b_output, INTERIM_DIR, PROCESSED_DIR
    )
    INPUT_FILES = [
        get_ds3_interim_path(),
        get_ds4_interim_path(),
        get_ds5_interim_path(),
        get_ds6_interim_path(),
    ]
    OUTPUT_FILE = get_track_b_output()
except ImportError:
    INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    INPUT_FILES = [
        INTERIM_DIR / "ds3_stackoverflow.jsonl",
        INTERIM_DIR / "ds4_synthetic.jsonl",
        INTERIM_DIR / "ds5_glaive.jsonl",
        INTERIM_DIR / "ds6_the_stack.jsonl",
    ]
    OUTPUT_FILE = PROCESSED_DIR / "track_B_combined.jsonl"


def combine_track_b():
    """Concatenate all Track B JSONL files into one."""
    log.info("=" * 60)
    log.info("TRACK B COMBINER - Generative Architect")
    log.info("=" * 60)
    
    total_records = 0
    records_by_source = {}
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_fh:
        for input_path in INPUT_FILES:
            log.info(f"Reading: {input_path}")
            
            if not input_path.exists():
                log.warning(f"  File not found: {input_path} - skipping")
                records_by_source[input_path.stem] = 0
                continue
            
            count = 0
            with open(input_path, "r", encoding="utf-8") as in_fh:
                for line in in_fh:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        record = json.loads(line)
                        record["source_dataset"] = input_path.stem
                        out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                        count += 1
                    except json.JSONDecodeError as e:
                        log.warning(f"  Skipping malformed JSON: {e}")
            
            log.info(f"  Loaded {count} records")
            records_by_source[input_path.stem] = count
            total_records += count
    
    log.info("-" * 40)
    log.info(f"Combined total: {total_records} records")
    log.info("Records by source:")
    for source, count in records_by_source.items():
        log.info(f"  {source}: {count}")
    
    log.info(f"Saved to: {OUTPUT_FILE}")
    log.info(f"File size: {OUTPUT_FILE.stat().st_size / 1024:.2f} KB")
    log.info("=" * 60)
    
    return total_records, records_by_source


if __name__ == "__main__":
    total, by_source = combine_track_b()
    print(f"\nCombination complete!")
    print(f"Total records: {total}")
    print(f"Output: {OUTPUT_FILE}")
