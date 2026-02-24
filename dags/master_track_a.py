"""
Master Track A DAG - Trigger Engine Pipeline
=============================================
Orchestrates the Track A (Trigger Engine) data pipeline:
  1. Trigger Dataset 1 (Alibaba) pipeline
  2. Trigger Dataset 2 (Loghub) pipeline
  3. Run Track A Combiner (waits for 1 & 2)
  4. Version combined output with DVC

Uses TriggerDagRunOperator to trigger individual dataset DAGs,
then runs the combiner and DVC versioning.

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
TRACK_A_OUTPUT = PROCESSED_DIR / "track_A_combined.parquet"

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
    """Run the Track A combiner script."""
    from src.combiner_track_a.combine import combine_track_a
    
    logger.info("Running Track A combiner")
    df = combine_track_a()
    logger.info("Track A combiner complete: %d rows", len(df))
    return len(df)


def dvc_version_combined(**context):
    """Version the combined output with DVC and send success alert."""
    from src.utils.dvc_utils import dvc_version_path
    
    logger.info("DVC versioning Track A combined output")
    result = dvc_version_path(str(TRACK_A_OUTPUT), cwd=str(PROJECT_ROOT), push=True)
    if result["add_result"]["returncode"] == 0:
        logger.info("Track A combined output versioned successfully")
    else:
        logger.warning("DVC versioning had issues: %s", result["add_result"]["stderr"])
    
    # Send success notification
    dag_id = context['dag'].dag_id
    run_id = context['run_id']
    alert_pipeline_success(dag_id, run_id, stats={"output": str(TRACK_A_OUTPUT)})
    
    return result


with DAG(
    dag_id="master_track_a",
    default_args=default_args,
    description="Track A (Trigger Engine) - Alibaba + Loghub -> Combined Parquet",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["track_a", "trigger_engine", "parquet", "master"],
) as dag:

    start = PythonOperator(
        task_id="pipeline_start",
        python_callable=pipeline_start,
    )

    trigger_ds1_alibaba = TriggerDagRunOperator(
        task_id="trigger_ds1_alibaba",
        trigger_dag_id="ds1_alibaba_pipeline",
        wait_for_completion=True,
        poke_interval=30,
    )

    trigger_ds2_loghub = TriggerDagRunOperator(
        task_id="trigger_ds2_loghub",
        trigger_dag_id="ds2_loghub_pipeline",
        wait_for_completion=True,
        poke_interval=30,
    )

    run_track_a_combiner = PythonOperator(
        task_id="run_combiner",
        python_callable=run_combiner,
    )

    dvc_version = PythonOperator(
        task_id="dvc_version_combined",
        python_callable=dvc_version_combined,
    )

    start >> [trigger_ds1_alibaba, trigger_ds2_loghub] >> run_track_a_combiner >> dvc_version
