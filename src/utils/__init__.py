"""Shared utilities for the Automend MLOps project."""

from .dvc_utils import (
    dvc_add,
    dvc_push,
    dvc_pull,
    dvc_version_path,
    check_raw_data_exists,
    version_raw_data,
)
from .ge_utils import (
    is_ge_available,
    get_ge_version,
    get_ge_api,
    create_ge_dataframe,
    run_legacy_expectations,
    validate_dataframe_simple,
)
from .alerting import (
    AlertSeverity,
    AlertType,
    format_slack_message,
    send_slack_alert,
    log_alert,
    send_alert,
    alert_pipeline_start,
    alert_pipeline_success,
    alert_pipeline_failure,
    alert_anomaly_detected,
    alert_validation_failure,
    alert_bias_detected,
    alert_data_quality_issue,
    on_failure_callback,
    on_success_callback,
)

__all__ = [
    # DVC utilities
    "dvc_add",
    "dvc_push",
    "dvc_pull",
    "dvc_version_path",
    "check_raw_data_exists",
    "version_raw_data",
    # Great Expectations utilities
    "is_ge_available",
    "get_ge_version",
    "get_ge_api",
    "create_ge_dataframe",
    "run_legacy_expectations",
    "validate_dataframe_simple",
    # Alerting utilities
    "AlertSeverity",
    "AlertType",
    "format_slack_message",
    "send_slack_alert",
    "log_alert",
    "send_alert",
    "alert_pipeline_start",
    "alert_pipeline_success",
    "alert_pipeline_failure",
    "alert_anomaly_detected",
    "alert_validation_failure",
    "alert_bias_detected",
    "alert_data_quality_issue",
    "on_failure_callback",
    "on_success_callback",
]
