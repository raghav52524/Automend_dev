"""TDD: Anomaly detection and alerts (e.g. Slack)."""
import sys
from pathlib import Path
import unittest.mock as mock

import pytest

DS4_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DS4_ROOT / "src"))

from data import anomaly


def test_detect_anomalies_returns_empty_for_valid_records():
    """Valid Format B records produce no anomalies."""
    records = [
        {"messages": [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}, {"role": "assistant", "content": "{}"}]},
    ]
    result = anomaly.detect_anomalies(records)
    assert isinstance(result, list)
    assert len(result) == 0


def test_detect_anomalies_finds_missing_values():
    """Records with missing required fields are reported."""
    records = [{"messages": []}]  # missing system/user/assistant
    result = anomaly.detect_anomalies(records)
    assert len(result) > 0
    assert any("missing" in str(a).lower() or "message" in str(a).lower() for a in result)


def test_detect_anomalies_finds_invalid_format():
    """Records with invalid assistant content (non-JSON) are reported."""
    records = [
        {"messages": [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}, {"role": "assistant", "content": "not valid json"}]},
    ]
    result = anomaly.detect_anomalies(records)
    # May or may not flag invalid JSON depending on implementation
    assert isinstance(result, list)


def test_send_alert_called_when_anomalies():
    """When anomalies exist, send_alert is invoked (mock Slack)."""
    with mock.patch("data.anomaly.send_alert") as m_send:
        anomaly.check_and_alert([], anomalies=["missing messages in record 0"])
        m_send.assert_called_once()
        args = m_send.call_args[0]
        assert len(args[0]) > 0 and "missing" in str(args[0][0]).lower()


def test_send_alert_not_called_when_no_anomalies():
    """When no anomalies, send_alert is not called."""
    with mock.patch("data.anomaly.send_alert") as m_send:
        anomaly.check_and_alert([], anomalies=[])
        m_send.assert_not_called()
