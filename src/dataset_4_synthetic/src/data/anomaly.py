"""Anomaly detection and alerts for pipeline monitoring.

Uses centralized alerting via src/utils/alerting.py
"""
import json
import sys
from pathlib import Path

# Ensure project root is on path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.pipeline_logger import get_logger

logger = get_logger(__name__)


def detect_anomalies(records: list[dict]) -> list[str]:
    """Detect missing values, invalid formats, and structural issues. Returns list of anomaly messages."""
    anomalies = []
    for i, r in enumerate(records):
        if not isinstance(r, dict):
            anomalies.append(f"Record {i}: not a dict")
            continue
        if "messages" not in r:
            anomalies.append(f"Record {i}: missing 'messages'")
            continue
        msgs = r["messages"]
        if not isinstance(msgs, list):
            anomalies.append(f"Record {i}: 'messages' is not a list")
            continue
        if len(msgs) < 3:
            anomalies.append(f"Record {i}: missing messages (expected at least 3: system, user, assistant)")
            continue
        roles = {}
        for m in msgs:
            if isinstance(m, dict) and "role" in m:
                roles[m["role"]] = m.get("content")
        for required in ("system", "user", "assistant"):
            if required not in roles:
                anomalies.append(f"Record {i}: missing role '{required}'")
        # Validate assistant content is valid JSON (workflow)
        if "assistant" in roles and roles["assistant"]:
            try:
                json.loads(roles["assistant"])
            except (json.JSONDecodeError, TypeError):
                anomalies.append(f"Record {i}: assistant content is not valid JSON")
    return anomalies


def send_alert(anomalies: list[str], channel: str = "slack") -> None:
    """
    Send anomaly alert using centralized alerting.
    
    Note: This function is kept for backwards compatibility but now delegates
    to the centralized alerting module.
    """
    if not anomalies:
        return
    
    from src.utils.alerting import alert_anomaly_detected
    
    message = "Pipeline anomaly detected:\n" + "\n".join(f"- {a}" for a in anomalies)
    logger.error("Anomalies detected: %s", message)
    
    # Send via centralized alerting
    alert_anomaly_detected(
        pipeline_name="ds4_synthetic",
        anomaly_count=len(anomalies),
        anomaly_types=["structural", "missing_field"],
        details={"sample_issues": anomalies[:5]}
    )


def check_and_alert(records: list[dict], anomalies: list[str] | None = None) -> list[str]:
    """Run detect_anomalies if anomalies not provided; if any, log and send alert. Returns list of anomalies."""
    if anomalies is None:
        anomalies = detect_anomalies(records)
    if anomalies:
        send_alert(anomalies)
    return anomalies
