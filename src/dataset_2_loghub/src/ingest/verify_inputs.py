"""Verify that all required LogHub input files are present in data/raw/ds2_loghub/."""
import sys
from pathlib import Path

DS2_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DS2_SRC))

# Allow running as script or imported by Airflow
from utils.paths import get_ds2_raw_dir, get_legacy_raw_dir
from utils.logger import get_logger

# Use centralized data path, fallback to legacy
DATA_RAW = get_ds2_raw_dir() / "loghub"
if not DATA_RAW.exists():
    DATA_RAW = get_legacy_raw_dir() / "loghub"

logger = get_logger(__name__)

REQUIRED = {
    "Linux":  ["Linux_2k.log_structured.csv",  "Linux_2k.log_templates.csv"],
    "HPC":    ["HPC_2k.log_structured.csv",    "HPC_2k.log_templates.csv"],
    "HDFS":   ["HDFS_2k.log_structured.csv",   "HDFS_2k.log_templates.csv"],
    "Hadoop": ["Hadoop_2k.log_structured.csv",  "Hadoop_2k.log_templates.csv"],
    "Spark":  ["Spark_2k.log_structured.csv",   "Spark_2k.log_templates.csv"],
}


def verify_inputs(data_raw: Path = DATA_RAW) -> bool:
    """Check all required files exist. Returns True if all present."""
    missing = []
    for system, files in REQUIRED.items():
        for fname in files:
            fpath = data_raw / system / fname
            if not fpath.exists():
                missing.append(str(fpath))

    if missing:
        logger.error("MISSING FILES:")
        for m in missing:
            logger.error("  ✗ %s", m)
        return False

    logger.info("All required input files are present:")
    for system, files in REQUIRED.items():
        for fname in files:
            fpath = data_raw / system / fname
            size_kb = fpath.stat().st_size // 1024
            logger.info("  ✓ %s/%s  (%d KB)", system, fname, size_kb)
    return True


if __name__ == "__main__":
    ok = verify_inputs()
    if not ok:
        sys.exit(1)
    logger.info("Verification passed.")
