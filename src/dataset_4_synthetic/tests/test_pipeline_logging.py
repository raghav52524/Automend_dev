"""TDD: Pipeline logging - assert pipeline modules log progress and errors."""
import sys
from pathlib import Path
import logging

import pytest

DS4_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DS4_ROOT / "src"))


def test_pipeline_logger_exists_and_logs_info(caplog):
    """Pipeline logger should be obtainable and log at INFO level."""
    from data import pipeline_logger

    caplog.set_level(logging.INFO)
    logger = pipeline_logger.get_logger(__name__)
    assert logger is not None
    logger.info("test message")
    assert any("test message" in r.message for r in caplog.records)


def test_pipeline_logger_emits_when_used(caplog):
    """When pipeline code logs, records appear at INFO."""
    from data import pipeline_logger

    caplog.set_level(logging.INFO)
    logger = pipeline_logger.get_logger("test")
    logger.info("progress: step 1")
    assert any("progress" in r.message for r in caplog.records)
    assert any(r.levelname == "INFO" for r in caplog.records)


def test_pipeline_logger_emits_errors_on_error(caplog):
    """Pipeline logger can log errors for alerting."""
    from data import pipeline_logger

    caplog.set_level(logging.ERROR)
    logger = pipeline_logger.get_logger("test")
    logger.error("anomaly detected: missing values")
    assert any("anomaly" in r.message for r in caplog.records)
    assert any(r.levelname == "ERROR" for r in caplog.records)
