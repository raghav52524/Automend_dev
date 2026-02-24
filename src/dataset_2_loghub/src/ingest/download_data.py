"""Download LogHub 2k log datasets from the LogPAI GitHub repository.

Downloads both structured CSV and templates CSV for each of the 5 systems:
  Linux, HPC, HDFS, Hadoop, Spark

Usage:
    python -m src.ingest.download_data
    python -m src.ingest.download_data --force   # re-download even if file exists

Files are saved to: data/raw/ds2_loghub/loghub/<System>/<filename>.csv
"""
import argparse
import sys
from pathlib import Path

DS2_SRC = Path(__file__).resolve().parent.parent  # .../src/dataset_2_loghub/src
PROJECT_ROOT = DS2_SRC.parent.parent.parent       # .../Automend
sys.path.insert(0, str(DS2_SRC))
sys.path.insert(0, str(PROJECT_ROOT))

# Try to use centralized paths, fall back to local definitions
# Note: We add /loghub subdirectory to match what other DS2 scripts expect
try:
    from src.config.paths import get_ds2_raw_dir
    RAW_DIR = get_ds2_raw_dir() / "loghub"
except ImportError:
    RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ds2_loghub" / "loghub"

from utils.logger import get_logger
logger = get_logger(__name__)

# Base URL for LogPAI/loghub raw files on GitHub
_BASE = "https://raw.githubusercontent.com/logpai/loghub/master"

# (system_folder, filename) pairs to download
DOWNLOAD_MANIFEST = [
    ("Linux",  "Linux_2k.log_structured.csv"),
    ("Linux",  "Linux_2k.log_templates.csv"),
    ("HPC",    "HPC_2k.log_structured.csv"),
    ("HPC",    "HPC_2k.log_templates.csv"),
    ("HDFS",   "HDFS_2k.log_structured.csv"),
    ("HDFS",   "HDFS_2k.log_templates.csv"),
    ("Hadoop", "Hadoop_2k.log_structured.csv"),
    ("Hadoop", "Hadoop_2k.log_templates.csv"),
    ("Spark",  "Spark_2k.log_structured.csv"),
    ("Spark",  "Spark_2k.log_templates.csv"),
]


def download_file(url: str, dest: Path) -> bool:
    """Download a single file. Returns True on success."""
    try:
        import requests
    except ImportError:
        logger.error("requests library not installed. Run: pip install requests")
        return False

    try:
        logger.info("Downloading %s", url)
        resp = requests.get(url, timeout=60, stream=True)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        size_kb = dest.stat().st_size // 1024
        logger.info("  Saved → %s  (%d KB)", dest, size_kb)
        return True
    except Exception as exc:
        logger.error("  FAILED to download %s: %s", url, exc)
        return False


def download_all(force: bool = False) -> bool:
    """Download all LogHub files. Skips existing files unless force=True.

    Returns True if all files are present after the run.
    """
    success_count = 0
    skip_count = 0
    fail_count = 0

    for system, filename in DOWNLOAD_MANIFEST:
        dest = RAW_DIR / system / filename
        if dest.exists() and not force:
            logger.info("Already exists — skipping %s/%s", system, filename)
            skip_count += 1
            success_count += 1
            continue

        url = f"{_BASE}/{system}/{filename}"
        if download_file(url, dest):
            success_count += 1
        else:
            fail_count += 1

    total = len(DOWNLOAD_MANIFEST)
    logger.info(
        "Download complete: %d/%d succeeded (%d skipped, %d failed)",
        success_count, total, skip_count, fail_count,
    )

    if fail_count > 0:
        logger.error(
            "%d file(s) failed to download. "
            "Try downloading manually from https://github.com/logpai/loghub",
            fail_count,
        )
        return False

    logger.info("All %d files present in %s", total, RAW_DIR)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download LogHub 2k log datasets")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist",
    )
    args = parser.parse_args()

    ok = download_all(force=args.force)
    if not ok:
        sys.exit(1)
