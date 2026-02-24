"""Centralized logging setup for the MLOps pipeline.

Usage in any pipeline script:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Task started")
"""
import logging
import sys
from pathlib import Path


def get_logger(name: str, log_file: str = "logs/pipeline.log") -> logging.Logger:
    """Return a logger that writes to both stdout and logs/pipeline.log.

    Args:
        name: Logger name (use __name__ in calling module).
        log_file: Path to log file, relative to project root.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # --- Console handler (INFO and above) ---
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # --- File handler (DEBUG and above, rotating) ---
    # Resolve log_file relative to project root (two levels up from src/utils/)
    project_root = Path(__file__).resolve().parent.parent.parent
    log_path = project_root / log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger
