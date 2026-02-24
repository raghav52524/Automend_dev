"""
Airflow DAG for IaC Payload Pipeline (The Stack Dataset)
download -> analyze -> preprocess -> validate -> anomaly -> bias -> dvc_version

Uses centralized Slack alerting via src/utils/alerting.py

Moved from src/dataset_6_the_stack/dags/iac_pipeline_dag.py
"""

import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
from airflow import DAG
try:
    from airflow.operators.python import PythonOperator
except ImportError:
    from airflow.operators.python_operator import PythonOperator

logger = logging.getLogger(__name__)

# Project paths - handle both local dev and Docker
PROJECT_ROOT = Path("/opt/airflow")
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT / "data"
DS6_RAW = DATA_ROOT / "raw" / "ds6_the_stack"
DS6_PROCESSED = DATA_ROOT / "processed" / "ds6_the_stack"
DS6_ROOT = PROJECT_ROOT / "src" / "dataset_6_the_stack"

sys.path.insert(0, str(DS6_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

# Centralized alerting
from src.utils.alerting import (
    alert_anomaly_detected,
    alert_validation_failure,
    alert_bias_detected,
    on_failure_callback,
)

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "on_failure_callback": on_failure_callback,
}


def task_download(**ctx):
    import os
    from src.utils.dvc_utils import check_raw_data_exists, version_raw_data
    
    os.chdir(PROJECT_ROOT)
    
    # Check if raw data exists locally or in DVC
    if check_raw_data_exists(DS6_RAW, project_root=PROJECT_ROOT):
        logger.info("Raw data found (local or DVC), skipping download")
        return {"status": "cached"}
    
    # Data not found - download fresh
    token = ctx["var"]["value"].get("HF_TOKEN", "") if "var" in ctx else ""
    if token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    
    DS6_RAW.mkdir(parents=True, exist_ok=True)
    from scripts.download.stack_iac_sample import download
    download()
    
    # Version the raw data after successful download
    version_raw_data(str(DS6_RAW), cwd=str(PROJECT_ROOT))
    logger.info("Raw data versioned with DVC")
    return {"status": "downloaded"}


def task_analyze(**ctx):
    import os
    os.chdir(PROJECT_ROOT)
    from scripts.analyze.stack_iac_analysis import analyze
    analyze()


def task_preprocess(**ctx):
    import os
    os.chdir(PROJECT_ROOT)
    from scripts.preprocess.payload_pipeline import run
    run()


def task_validate(**ctx):
    import os
    os.chdir(PROJECT_ROOT)
    from scripts.validate.schema_stats import run_validation
    run_validation()


def task_anomaly(**ctx):
    import os
    os.chdir(PROJECT_ROOT)
    from scripts.validate.anomaly_alerts import run_anomaly_check
    result = run_anomaly_check()
    
    # Send centralized alert if anomalies detected
    if result.get("anomalies_found", 0) > 0:
        alert_anomaly_detected(
            pipeline_name="ds6_the_stack",
            anomaly_count=result["anomalies_found"],
            anomaly_types=result.get("anomalies", []),
        )
    return result


def task_bias(**ctx):
    import os
    os.chdir(PROJECT_ROOT)
    from scripts.validate.bias_detection import run_bias_detection
    result = run_bias_detection()
    
    # Send centralized alert if bias detected
    if result and result.get("bias_detected"):
        biased_slices = [s.get("slice", "unknown") for s in result.get("biased_slices", [])]
        alert_bias_detected(
            pipeline_name="ds6_the_stack",
            biased_slices=biased_slices,
            severity_counts={"high": len(biased_slices)}
        )
    return result


def task_dvc_version(**ctx):
    from src.utils.dvc_utils import dvc_version_path
    logger.info("DVC versioning DS6 The Stack processed output")
    result = dvc_version_path(str(DS6_PROCESSED), cwd=str(PROJECT_ROOT), push=True)
    if result["add_result"]["returncode"] == 0:
        logger.info("DS6 The Stack output versioned successfully")
    else:
        logger.warning("DVC versioning had issues: %s", result["add_result"]["stderr"])
    return result


with DAG(
    dag_id="ds6_iac_pipeline",
    description="Dataset 6 - IaC Payload Layer: download -> analyze -> preprocess -> validate -> anomaly -> bias",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["ds6_the_stack", "iac", "track_b"],
) as dag:

    download = PythonOperator(
        task_id="download",
        python_callable=task_download,
        doc_md="Stream IaC YAML files from The Stack -> data/raw/ds6_the_stack/",
    )

    analyze = PythonOperator(
        task_id="analyze",
        python_callable=task_analyze,
        doc_md="Analyze raw chunks -> logs/analysis_report.json",
    )

    preprocess = PythonOperator(
        task_id="preprocess",
        python_callable=task_preprocess,
        doc_md="Filter + redact + wrap -> data/processed/ds6_the_stack/training_records.jsonl",
    )

    validate = PythonOperator(
        task_id="validate",
        python_callable=task_validate,
        doc_md="Schema + PII validation -> logs/schema_report.json",
    )

    anomaly = PythonOperator(
        task_id="anomaly_check",
        python_callable=task_anomaly,
        doc_md="Threshold alerts -> logs/anomaly_report.json",
    )

    bias = PythonOperator(
        task_id="bias_detection",
        python_callable=task_bias,
        doc_md="Slice imbalance report -> logs/bias_report.json",
    )

    dvc_version = PythonOperator(
        task_id="dvc_version",
        python_callable=task_dvc_version,
        doc_md="Version processed output with DVC",
    )

    download >> analyze >> preprocess >> validate >> anomaly >> bias >> dvc_version
