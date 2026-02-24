"""Centralized pipeline logging for progress tracking and error monitoring."""
import logging
import sys

PIPELINE_LOG_LEVEL = logging.INFO


def get_logger(name: str) -> logging.Logger:
    """Return a logger for pipeline components. Configured with handler if not yet set."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(PIPELINE_LOG_LEVEL)
    return logger
