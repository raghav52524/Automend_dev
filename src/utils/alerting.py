"""
Centralized Alerting Module for AutoMend MLOps Pipeline
========================================================
Unified Slack webhook-based alerting for all dataset pipelines.

Configuration via environment variables:
    SLACK_WEBHOOK_URL - Required for Slack alerts
    SLACK_CHANNEL - Optional, for display purposes (defaults to #automend-alerts)

Usage:
    from src.utils.alerting import (
        alert_pipeline_start,
        alert_pipeline_success,
        alert_pipeline_failure,
        alert_anomaly_detected,
        alert_validation_failure,
    )
    
    # Simple usage
    alert_pipeline_start("ds1_alibaba_dag", "run_123")
    
    # With details
    alert_anomaly_detected(
        pipeline_name="ds5_glaive",
        anomaly_count=3,
        anomaly_types=["malformed_rate", "record_count"],
        details={"threshold_exceeded": "5%"}
    )
"""

import json
import logging
import os
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import requests

# Get LOGS_DIR from paths module if available, otherwise use default
try:
    from src.config.paths import LOGS_DIR
except ImportError:
    LOGS_DIR = Path(__file__).resolve().parent.parent.parent / "logs"

LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Configure module logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("automend.alerting")

# Add file handler for alert logs
_alert_log_handler = logging.FileHandler(LOGS_DIR / "alerts.log", encoding="utf-8")
_alert_log_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
logger.addHandler(_alert_log_handler)


class AlertSeverity(Enum):
    """Severity levels for alerts."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts that can be sent."""
    PIPELINE_START = "pipeline_start"
    PIPELINE_SUCCESS = "pipeline_success"
    PIPELINE_FAILURE = "pipeline_failure"
    VALIDATION_FAILURE = "validation_failure"
    ANOMALY_DETECTED = "anomaly_detected"
    BIAS_DETECTED = "bias_detected"
    DATA_QUALITY_ISSUE = "data_quality_issue"


# Slack message formatting colors
_SEVERITY_COLORS = {
    AlertSeverity.INFO: "#36a64f",      # Green
    AlertSeverity.WARNING: "#ffcc00",   # Yellow
    AlertSeverity.ERROR: "#ff6600",     # Orange
    AlertSeverity.CRITICAL: "#ff0000",  # Red
}

# Emojis for alert types
_ALERT_EMOJIS = {
    AlertType.PIPELINE_START: ":rocket:",
    AlertType.PIPELINE_SUCCESS: ":white_check_mark:",
    AlertType.PIPELINE_FAILURE: ":x:",
    AlertType.VALIDATION_FAILURE: ":warning:",
    AlertType.ANOMALY_DETECTED: ":mag:",
    AlertType.BIAS_DETECTED: ":scales:",
    AlertType.DATA_QUALITY_ISSUE: ":bar_chart:",
}


def _get_slack_config() -> tuple[Optional[str], str]:
    """Get Slack configuration from environment variables."""
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL", "").strip()
    channel = os.environ.get("SLACK_CHANNEL", "#automend-alerts")
    return webhook_url if webhook_url else None, channel


def format_slack_message(
    alert_type: AlertType,
    severity: AlertSeverity,
    title: str,
    message: str,
    details: Optional[dict] = None,
    pipeline_name: Optional[str] = None,
) -> dict:
    """
    Format alert as a rich Slack message with blocks.
    
    Args:
        alert_type: Type of alert
        severity: Severity level
        title: Alert title
        message: Main message text
        details: Additional key-value details to display
        pipeline_name: Optional pipeline name for context
    
    Returns:
        dict: Slack message payload with attachments and blocks
    """
    emoji = _ALERT_EMOJIS.get(alert_type, ":bell:")
    color = _SEVERITY_COLORS.get(severity, "#808080")
    
    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{emoji} {title}",
                "emoji": True
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": message
            }
        }
    ]
    
    # Add pipeline context if provided
    if pipeline_name:
        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f":gear: *Pipeline:* {pipeline_name}"
                }
            ]
        })
    
    # Add details if provided
    if details:
        detail_lines = [f"*{k}:* {v}" for k, v in details.items()]
        detail_text = "\n".join(detail_lines)
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Details:*\n{detail_text}"
            }
        })
    
    # Add timestamp footer
    blocks.append({
        "type": "context",
        "elements": [
            {
                "type": "mrkdwn",
                "text": f":clock1: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}"
            }
        ]
    })
    
    return {
        "attachments": [{
            "color": color,
            "blocks": blocks
        }]
    }


def send_slack_alert(
    alert_type: AlertType,
    severity: AlertSeverity,
    title: str,
    message: str,
    details: Optional[dict] = None,
    pipeline_name: Optional[str] = None,
) -> bool:
    """
    Send alert to Slack via webhook.
    
    Args:
        alert_type: Type of alert
        severity: Severity level
        title: Alert title
        message: Main message text
        details: Additional details dict
        pipeline_name: Optional pipeline name for context
    
    Returns:
        bool: True if alert was sent successfully, False otherwise
    """
    webhook_url, channel = _get_slack_config()
    
    if not webhook_url:
        logger.warning(
            "SLACK_WEBHOOK_URL not configured - alert logged only: %s", title
        )
        return False
    
    try:
        payload = format_slack_message(
            alert_type, severity, title, message, details, pipeline_name
        )
        
        response = requests.post(
            webhook_url,
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            logger.info("Slack alert sent successfully: %s", title)
            return True
        else:
            logger.error(
                "Slack alert failed (HTTP %d): %s - %s",
                response.status_code, title, response.text
            )
            return False
            
    except requests.RequestException as e:
        logger.error("Failed to send Slack alert: %s - %s", title, e)
        return False


def log_alert(
    alert_type: AlertType,
    severity: AlertSeverity,
    title: str,
    message: str,
    details: Optional[dict] = None,
    pipeline_name: Optional[str] = None,
) -> None:
    """
    Log alert to file (always executes as fallback).
    
    Args:
        alert_type: Type of alert
        severity: Severity level
        title: Alert title
        message: Main message text
        details: Additional details dict
        pipeline_name: Optional pipeline name
    """
    alert_record = {
        "timestamp": datetime.now().isoformat(),
        "type": alert_type.value,
        "severity": severity.value,
        "title": title,
        "message": message,
        "pipeline": pipeline_name,
        "details": details
    }
    
    # Log to appropriate level
    log_func = {
        AlertSeverity.INFO: logger.info,
        AlertSeverity.WARNING: logger.warning,
        AlertSeverity.ERROR: logger.error,
        AlertSeverity.CRITICAL: logger.critical,
    }.get(severity, logger.info)
    
    pipeline_ctx = f"[{pipeline_name}] " if pipeline_name else ""
    log_func("ALERT %s[%s]: %s - %s", pipeline_ctx, alert_type.value, title, message)
    
    # Append to JSON history file
    alerts_file = LOGS_DIR / "alerts_history.json"
    
    try:
        alerts = []
        if alerts_file.exists():
            try:
                content = alerts_file.read_text(encoding="utf-8").strip()
                if content:
                    alerts = json.loads(content)
                    if not isinstance(alerts, list):
                        alerts = []
            except (json.JSONDecodeError, ValueError):
                logger.warning("alerts_history.json was corrupted, starting fresh")
                alerts = []
        
        alerts.append(alert_record)
        
        # Keep only last 1000 alerts
        alerts = alerts[-1000:]
        
        alerts_file.write_text(
            json.dumps(alerts, indent=2, default=str),
            encoding="utf-8"
        )
        
    except Exception as e:
        logger.error("Failed to write alert to history: %s", e)


def send_alert(
    alert_type: AlertType,
    severity: AlertSeverity,
    title: str,
    message: str,
    details: Optional[dict] = None,
    pipeline_name: Optional[str] = None,
) -> dict:
    """
    Send alert through all configured channels.
    
    Args:
        alert_type: Type of alert
        severity: Severity level
        title: Alert title
        message: Main message text
        details: Additional details dict
        pipeline_name: Optional pipeline name
    
    Returns:
        dict: Results of sending through each channel
    """
    results = {"log": True}
    
    # Always log to file
    log_alert(alert_type, severity, title, message, details, pipeline_name)
    
    # Send to Slack for WARNING and above (skip INFO to reduce noise)
    if severity != AlertSeverity.INFO:
        results["slack"] = send_slack_alert(
            alert_type, severity, title, message, details, pipeline_name
        )
    else:
        # For INFO, still try to send but don't require it
        results["slack"] = send_slack_alert(
            alert_type, severity, title, message, details, pipeline_name
        )
    
    return results


# =============================================================================
# Convenience Functions for Common Alert Scenarios
# =============================================================================

def alert_pipeline_start(
    dag_id: str,
    run_id: str,
    extra_details: Optional[dict] = None,
) -> dict:
    """
    Send pipeline start notification.
    
    Args:
        dag_id: Airflow DAG ID
        run_id: Airflow run ID
        extra_details: Additional context to include
    
    Returns:
        dict: Alert sending results
    """
    details = {"dag_id": dag_id, "run_id": run_id}
    if extra_details:
        details.update(extra_details)
    
    return send_alert(
        AlertType.PIPELINE_START,
        AlertSeverity.INFO,
        "Pipeline Started",
        f"DAG `{dag_id}` has started execution.",
        details=details,
        pipeline_name=dag_id,
    )


def alert_pipeline_success(
    dag_id: str,
    run_id: str,
    duration_seconds: Optional[float] = None,
    stats: Optional[dict] = None,
) -> dict:
    """
    Send pipeline success notification.
    
    Args:
        dag_id: Airflow DAG ID
        run_id: Airflow run ID
        duration_seconds: Pipeline execution duration
        stats: Additional statistics to include
    
    Returns:
        dict: Alert sending results
    """
    details = {"dag_id": dag_id, "run_id": run_id}
    if duration_seconds is not None:
        details["duration"] = f"{duration_seconds:.1f}s"
    if stats:
        details.update(stats)
    
    return send_alert(
        AlertType.PIPELINE_SUCCESS,
        AlertSeverity.INFO,
        "Pipeline Completed Successfully",
        f"DAG `{dag_id}` completed successfully.",
        details=details,
        pipeline_name=dag_id,
    )


def alert_pipeline_failure(
    dag_id: str,
    run_id: str,
    task_id: str,
    error: Union[str, Exception],
) -> dict:
    """
    Send pipeline failure notification.
    
    Args:
        dag_id: Airflow DAG ID
        run_id: Airflow run ID
        task_id: Failed task ID
        error: Error message or exception
    
    Returns:
        dict: Alert sending results
    """
    error_str = str(error)[:500]  # Truncate long errors
    
    return send_alert(
        AlertType.PIPELINE_FAILURE,
        AlertSeverity.CRITICAL,
        "Pipeline Failed",
        f"DAG `{dag_id}` failed at task `{task_id}`.",
        details={
            "dag_id": dag_id,
            "run_id": run_id,
            "failed_task": task_id,
            "error": error_str,
        },
        pipeline_name=dag_id,
    )


def alert_anomaly_detected(
    pipeline_name: str,
    anomaly_count: int,
    anomaly_types: Optional[list] = None,
    details: Optional[dict] = None,
) -> dict:
    """
    Send anomaly detection notification.
    
    Args:
        pipeline_name: Name of the pipeline/dataset
        anomaly_count: Number of anomalies detected
        anomaly_types: List of anomaly type names
        details: Additional anomaly details
    
    Returns:
        dict: Alert sending results
    """
    alert_details = {"anomaly_count": anomaly_count}
    
    if anomaly_types:
        unique_types = list(set(anomaly_types))[:5]
        alert_details["types"] = ", ".join(unique_types)
    
    if details:
        alert_details.update(details)
    
    severity = AlertSeverity.WARNING
    if anomaly_count > 10:
        severity = AlertSeverity.ERROR
    
    return send_alert(
        AlertType.ANOMALY_DETECTED,
        severity,
        "Data Anomalies Detected",
        f"Found {anomaly_count} anomalies in `{pipeline_name}` pipeline.",
        details=alert_details,
        pipeline_name=pipeline_name,
    )


def alert_validation_failure(
    pipeline_name: str,
    validation_results: Optional[dict] = None,
    failed_checks: Optional[list] = None,
) -> dict:
    """
    Send validation failure notification.
    
    Args:
        pipeline_name: Name of the pipeline/dataset
        validation_results: Full validation results dict
        failed_checks: List of failed check names
    
    Returns:
        dict: Alert sending results
    """
    if validation_results:
        failed_checks = failed_checks or [
            f.get("check", str(f)) for f in validation_results.get("failed", [])
        ]
    
    failed_checks = failed_checks or []
    check_count = len(failed_checks)
    
    details = {
        "failed_checks": ", ".join(failed_checks[:5]) if failed_checks else "unknown",
        "total_failures": check_count,
    }
    
    return send_alert(
        AlertType.VALIDATION_FAILURE,
        AlertSeverity.ERROR,
        "Data Validation Failed",
        f"Data validation failed with {check_count} check(s) in `{pipeline_name}`.",
        details=details,
        pipeline_name=pipeline_name,
    )


def alert_bias_detected(
    pipeline_name: str,
    biased_slices: list,
    severity_counts: Optional[dict] = None,
) -> dict:
    """
    Send bias detection notification.
    
    Args:
        pipeline_name: Name of the pipeline/dataset
        biased_slices: List of data slices with detected bias
        severity_counts: Dict with high/medium/low bias counts
    
    Returns:
        dict: Alert sending results
    """
    severity_counts = severity_counts or {}
    
    severity = AlertSeverity.WARNING
    if severity_counts.get("high", 0) > 0:
        severity = AlertSeverity.ERROR
    
    details = {
        "biased_slices": ", ".join(biased_slices[:5]),
        "high_severity": severity_counts.get("high", 0),
        "medium_severity": severity_counts.get("medium", 0),
        "low_severity": severity_counts.get("low", 0),
    }
    
    return send_alert(
        AlertType.BIAS_DETECTED,
        severity,
        "Data Bias Detected",
        f"Bias detected in {len(biased_slices)} data slice(s) in `{pipeline_name}`.",
        details=details,
        pipeline_name=pipeline_name,
    )


def alert_data_quality_issue(
    pipeline_name: str,
    issue_description: str,
    details: Optional[dict] = None,
    severity: AlertSeverity = AlertSeverity.WARNING,
) -> dict:
    """
    Send generic data quality issue notification.
    
    Args:
        pipeline_name: Name of the pipeline/dataset
        issue_description: Description of the quality issue
        details: Additional context
        severity: Alert severity level
    
    Returns:
        dict: Alert sending results
    """
    return send_alert(
        AlertType.DATA_QUALITY_ISSUE,
        severity,
        "Data Quality Issue",
        issue_description,
        details=details,
        pipeline_name=pipeline_name,
    )


# =============================================================================
# Airflow Callback Functions
# =============================================================================

def on_failure_callback(context: dict) -> None:
    """
    Airflow callback function for task failures.
    
    Usage in DAG default_args:
        default_args = {
            "on_failure_callback": on_failure_callback,
        }
    
    Args:
        context: Airflow context dictionary
    """
    dag_id = context.get("dag").dag_id if context.get("dag") else "unknown"
    task_id = context.get("task_instance").task_id if context.get("task_instance") else "unknown"
    run_id = context.get("run_id", "unknown")
    exception = context.get("exception", "Unknown error")
    
    alert_pipeline_failure(
        dag_id=dag_id,
        run_id=run_id,
        task_id=task_id,
        error=exception,
    )


def on_success_callback(context: dict) -> None:
    """
    Airflow callback function for DAG success.
    
    Note: This should be used sparingly as it can be noisy.
    Consider using only for critical pipelines.
    
    Args:
        context: Airflow context dictionary
    """
    dag_id = context.get("dag").dag_id if context.get("dag") else "unknown"
    run_id = context.get("run_id", "unknown")
    
    # Calculate duration if available
    duration = None
    ti = context.get("task_instance")
    if ti and hasattr(ti, "start_date") and hasattr(ti, "end_date"):
        if ti.start_date and ti.end_date:
            duration = (ti.end_date - ti.start_date).total_seconds()
    
    alert_pipeline_success(
        dag_id=dag_id,
        run_id=run_id,
        duration_seconds=duration,
    )


# Module exports
__all__ = [
    # Enums
    "AlertSeverity",
    "AlertType",
    # Core functions
    "format_slack_message",
    "send_slack_alert",
    "log_alert",
    "send_alert",
    # Convenience functions
    "alert_pipeline_start",
    "alert_pipeline_success",
    "alert_pipeline_failure",
    "alert_anomaly_detected",
    "alert_validation_failure",
    "alert_bias_detected",
    "alert_data_quality_issue",
    # Airflow callbacks
    "on_failure_callback",
    "on_success_callback",
]
