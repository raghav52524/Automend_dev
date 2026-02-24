"""
Master Track B DAG - Generative Architect Pipeline
==================================================
Orchestrates the Track B (Generative Architect) data pipeline:
  1. Trigger Dataset 3 (StackOverflow) pipeline
  2. Trigger Dataset 4 (Synthetic) pipeline
  3. Trigger Dataset 5 (Glaive) pipeline
  4. Trigger Dataset 6 (The Stack) pipeline
  5. Run Track B Combiner (waits for 1-4)
  6. Version combined output with DVC

Uses TriggerDagRunOperator to trigger individual dataset DAGs,
then runs the combiner and DVC versioning.
All dataset tasks run in parallel for efficiency.

Uses centralized Slack alerting via src/utils/alerting.py
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
try:
    from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
except ImportError:
    try:
        from airflow.operators.trigger_dagrun import TriggerDagRunOperator
    except ImportError:
        from airflow.operators.dagrun_operator import TriggerDagRunOperator
try:
    from airflow.providers.standard.operators.python import PythonOperator
except ImportError:
    from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

# Project paths (relative to /opt/airflow in Docker)
PROJECT_ROOT = Path("/opt/airflow")
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))

DATA_ROOT = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_ROOT / "processed"
TRACK_B_OUTPUT = PROCESSED_DIR / "track_B_combined.jsonl"

# Centralized alerting
from src.utils.alerting import (
    alert_pipeline_start,
    alert_pipeline_success,
    on_failure_callback,
)

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "on_failure_callback": on_failure_callback,
}


def pipeline_start(**context):
    """Send pipeline start notification."""
    dag_id = context['dag'].dag_id
    run_id = context['run_id']
    alert_pipeline_start(dag_id, run_id)
    return {"start_time": datetime.now().isoformat()}


def run_combiner():
    """Run the Track B combiner script."""
    from src.combiner_track_b.combine import combine_track_b
    
    logger.info("Running Track B combiner")
    total, by_source = combine_track_b()
    logger.info("Track B combiner complete: %d total records", total)
    return {"total": total, "by_source": by_source}


def dvc_version_combined(**context):
    """Version the combined output with DVC and send success alert."""
    from src.utils.dvc_utils import dvc_version_path
    
    logger.info("DVC versioning Track B combined output")
    result = dvc_version_path(str(TRACK_B_OUTPUT), cwd=str(PROJECT_ROOT), push=True)
    if result["add_result"]["returncode"] == 0:
        logger.info("Track B combined output versioned successfully")
    else:
        logger.warning("DVC versioning had issues: %s", result["add_result"]["stderr"])
    
    # Send success notification
    dag_id = context['dag'].dag_id
    run_id = context['run_id']
    alert_pipeline_success(dag_id, run_id, stats={"output": str(TRACK_B_OUTPUT)})
    
    return result


with DAG(
    dag_id="master_track_b",
    default_args=default_args,
    description="Track B (Generative Architect) - DS3-6 -> Combined JSONL ChatML",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["track_b", "generative_architect", "jsonl", "chatml", "master"],
) as dag:

    start = PythonOperator(
        task_id="pipeline_start",
        python_callable=pipeline_start,
    )

    trigger_ds3_stackoverflow = TriggerDagRunOperator(
        task_id="trigger_ds3_stackoverflow",
        trigger_dag_id="ds3_stackoverflow_pipeline",
        wait_for_completion=True,
        poke_interval=30,
    )

    trigger_ds4_synthetic = TriggerDagRunOperator(
        task_id="trigger_ds4_synthetic",
        trigger_dag_id="ds4_synthetic_dag",
        wait_for_completion=True,
        poke_interval=30,
    )

    trigger_ds5_glaive = TriggerDagRunOperator(
        task_id="trigger_ds5_glaive",
        trigger_dag_id="ds5_glaive_pipeline",
        wait_for_completion=True,
        poke_interval=30,
    )

    trigger_ds6_the_stack = TriggerDagRunOperator(
        task_id="trigger_ds6_the_stack",
        trigger_dag_id="ds6_iac_pipeline",
        wait_for_completion=True,
        poke_interval=30,
    )

    run_track_b_combiner = PythonOperator(
        task_id="run_combiner",
        python_callable=run_combiner,
    )

    dvc_version = PythonOperator(
        task_id="dvc_version_combined",
        python_callable=dvc_version_combined,
    )

    start >> [
        trigger_ds3_stackoverflow,
        trigger_ds4_synthetic,
        trigger_ds5_glaive,
        trigger_ds6_the_stack,
    ] >> run_track_b_combiner >> dvc_version
