"""Phase 5: Airflow DAG integrity tests - DAG load, 8 tasks, dependencies."""
import os
import sys
from pathlib import Path

import pytest


def _get_dag():
    """Load DAG by importing the module (DagBag uses Unix signals and fails on Windows)."""
    # Navigate to monorepo root (3 levels up from tests/)
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if str(project_root / "dags") not in sys.path:
        sys.path.insert(0, str(project_root / "dags"))
    os.chdir(project_root)
    from ds4_synthetic_dag import dag
    return dag


def test_dag_loads_without_errors():
    """DAG module imports with no errors."""
    dag = _get_dag()
    assert dag is not None
    assert dag.dag_id == "ds4_synthetic_dag"


def test_dag_has_exactly_eight_tasks():
    """DAG contains exactly 8 tasks (incl. schema/stats, validate_and_alert, export_to_interim)."""
    dag = _get_dag()
    task_ids = [t.task_id for t in dag.tasks]
    assert len(task_ids) == 8, f"Expected 8 tasks, got {len(task_ids)}: {task_ids}"


def test_dag_has_expected_task_ids():
    """DAG has pull_data, fetch_prompts, generate_synthetic_data, format_data, generate_schema_and_stats, validate_and_alert, export_to_interim, commit_and_push."""
    dag = _get_dag()
    task_ids = {t.task_id for t in dag.tasks}
    expected = {"pull_data", "fetch_prompts", "generate_synthetic_data", "format_data", "generate_schema_and_stats", "validate_and_alert", "export_to_interim", "commit_and_push"}
    assert expected <= task_ids, f"Missing tasks: {expected - task_ids}"


def test_dag_dependencies_linear_order():
    """Tasks run in order: pull_data -> ... -> format_data -> generate_schema_and_stats -> validate_and_alert -> export_to_interim -> commit_and_push."""
    dag = _get_dag()
    upstream = {
        "pull_data": set(),
        "fetch_prompts": {"pull_data"},
        "generate_synthetic_data": {"fetch_prompts"},
        "format_data": {"generate_synthetic_data"},
        "generate_schema_and_stats": {"format_data"},
        "validate_and_alert": {"generate_schema_and_stats"},
        "export_to_interim": {"validate_and_alert"},
        "commit_and_push": {"export_to_interim"},
    }
    for task in dag.tasks:
        tid = task.task_id
        if tid not in upstream:
            continue
        upstream_ids = {d.task_id for d in task.upstream_list}
        assert upstream[tid] <= upstream_ids, f"Task {tid} should have upstream {upstream[tid]}, got {upstream_ids}"
