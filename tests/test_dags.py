"""
DAG Integrity Tests for Airflow Master DAGs.

Tests that the master_track_a and master_track_b DAGs:
- Load without errors
- Have the expected number of tasks
- Have correct task IDs
- Have correct task dependencies

Run with: pytest tests/test_dags.py -v

Note: These tests require airflow to be installed. Since Airflow runs via
Docker in this project, these tests will be skipped if airflow is not available.
"""

import os
import sys
from pathlib import Path

import pytest

# Add project root and dags to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "dags"))

# Check if airflow is available
try:
    import airflow
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False

# Skip all tests in this module if airflow is not installed
pytestmark = pytest.mark.skipif(
    not AIRFLOW_AVAILABLE,
    reason="Airflow not installed locally (runs via Docker)"
)


def _get_dag_track_a():
    """Load Track A DAG by importing the module."""
    os.chdir(PROJECT_ROOT)
    from dags.master_track_a import dag
    return dag


def _get_dag_track_b():
    """Load Track B DAG by importing the module."""
    os.chdir(PROJECT_ROOT)
    from dags.master_track_b import dag
    return dag


# =============================================================================
# Track A DAG Tests
# =============================================================================

class TestMasterTrackADag:
    """Tests for the master_track_a DAG."""

    @pytest.mark.dag
    def test_master_track_a_loads(self):
        """DAG module imports without errors."""
        dag = _get_dag_track_a()
        assert dag is not None
        assert dag.dag_id == "master_track_a"

    @pytest.mark.dag
    def test_master_track_a_has_5_tasks(self):
        """DAG contains exactly 5 tasks (start + 2 triggers + combiner + DVC)."""
        dag = _get_dag_track_a()
        task_ids = [t.task_id for t in dag.tasks]
        assert len(task_ids) == 5, f"Expected 5 tasks, got {len(task_ids)}: {task_ids}"

    @pytest.mark.dag
    def test_master_track_a_task_ids(self):
        """DAG has the expected task IDs."""
        dag = _get_dag_track_a()
        task_ids = {t.task_id for t in dag.tasks}
        expected = {
            "pipeline_start",
            "trigger_ds1_alibaba",
            "trigger_ds2_loghub",
            "run_combiner",
            "dvc_version_combined",
        }
        assert expected == task_ids, f"Task IDs mismatch. Expected: {expected}, Got: {task_ids}"

    @pytest.mark.dag
    def test_master_track_a_dependencies(self):
        """pipeline_start -> triggers in parallel -> combiner -> DVC."""
        dag = _get_dag_track_a()

        # Build task lookup
        tasks = {t.task_id: t for t in dag.tasks}

        # pipeline_start should have no upstream
        start_upstream = {d.task_id for d in tasks["pipeline_start"].upstream_list}
        assert start_upstream == set(), f"pipeline_start should have no upstream, got: {start_upstream}"

        # Trigger tasks should have pipeline_start as upstream
        ds1_upstream = {d.task_id for d in tasks["trigger_ds1_alibaba"].upstream_list}
        ds2_upstream = {d.task_id for d in tasks["trigger_ds2_loghub"].upstream_list}
        assert ds1_upstream == {"pipeline_start"}, f"DS1 trigger should have pipeline_start upstream, got: {ds1_upstream}"
        assert ds2_upstream == {"pipeline_start"}, f"DS2 trigger should have pipeline_start upstream, got: {ds2_upstream}"

        # Combiner should wait for both triggers
        combiner_upstream = {d.task_id for d in tasks["run_combiner"].upstream_list}
        expected_upstream = {"trigger_ds1_alibaba", "trigger_ds2_loghub"}
        assert combiner_upstream == expected_upstream, (
            f"Combiner should wait for {expected_upstream}, got: {combiner_upstream}"
        )

        # DVC version should wait for combiner
        dvc_upstream = {d.task_id for d in tasks["dvc_version_combined"].upstream_list}
        assert dvc_upstream == {"run_combiner"}, (
            f"DVC should wait for combiner, got: {dvc_upstream}"
        )

    @pytest.mark.dag
    def test_master_track_a_tags(self):
        """DAG has expected tags."""
        dag = _get_dag_track_a()
        assert "track_a" in dag.tags
        assert "trigger_engine" in dag.tags

    @pytest.mark.dag
    def test_master_track_a_uses_trigger_dag_run(self):
        """DAG uses TriggerDagRunOperator for dataset tasks."""
        dag = _get_dag_track_a()
        tasks = {t.task_id: t for t in dag.tasks}
        
        trigger_task = tasks["trigger_ds1_alibaba"]
        assert "TriggerDagRunOperator" in type(trigger_task).__name__


# =============================================================================
# Track B DAG Tests
# =============================================================================

class TestMasterTrackBDag:
    """Tests for the master_track_b DAG."""

    @pytest.mark.dag
    def test_master_track_b_loads(self):
        """DAG module imports without errors."""
        dag = _get_dag_track_b()
        assert dag is not None
        assert dag.dag_id == "master_track_b"

    @pytest.mark.dag
    def test_master_track_b_has_7_tasks(self):
        """DAG contains exactly 7 tasks (start + 4 triggers + combiner + DVC)."""
        dag = _get_dag_track_b()
        task_ids = [t.task_id for t in dag.tasks]
        assert len(task_ids) == 7, f"Expected 7 tasks, got {len(task_ids)}: {task_ids}"

    @pytest.mark.dag
    def test_master_track_b_task_ids(self):
        """DAG has the expected task IDs."""
        dag = _get_dag_track_b()
        task_ids = {t.task_id for t in dag.tasks}
        expected = {
            "pipeline_start",
            "trigger_ds3_stackoverflow",
            "trigger_ds4_synthetic",
            "trigger_ds5_glaive",
            "trigger_ds6_the_stack",
            "run_combiner",
            "dvc_version_combined",
        }
        assert expected == task_ids, f"Task IDs mismatch. Expected: {expected}, Got: {task_ids}"

    @pytest.mark.dag
    def test_master_track_b_dependencies(self):
        """pipeline_start -> DS3-6 triggers in parallel -> combiner -> DVC."""
        dag = _get_dag_track_b()

        # Build task lookup
        tasks = {t.task_id: t for t in dag.tasks}

        # pipeline_start should have no upstream
        start_upstream = {d.task_id for d in tasks["pipeline_start"].upstream_list}
        assert start_upstream == set(), f"pipeline_start should have no upstream, got: {start_upstream}"

        # All trigger tasks should have pipeline_start as upstream
        trigger_tasks = [
            "trigger_ds3_stackoverflow",
            "trigger_ds4_synthetic",
            "trigger_ds5_glaive",
            "trigger_ds6_the_stack",
        ]
        for task_id in trigger_tasks:
            upstream = {d.task_id for d in tasks[task_id].upstream_list}
            assert upstream == {"pipeline_start"}, f"{task_id} should have pipeline_start upstream, got: {upstream}"

        # Combiner should wait for all 4 trigger tasks
        combiner_upstream = {d.task_id for d in tasks["run_combiner"].upstream_list}
        expected_upstream = set(trigger_tasks)
        assert combiner_upstream == expected_upstream, (
            f"Combiner should wait for {expected_upstream}, got: {combiner_upstream}"
        )

        # DVC version should wait for combiner
        dvc_upstream = {d.task_id for d in tasks["dvc_version_combined"].upstream_list}
        assert dvc_upstream == {"run_combiner"}, (
            f"DVC should wait for combiner, got: {dvc_upstream}"
        )

    @pytest.mark.dag
    def test_master_track_b_tags(self):
        """DAG has expected tags."""
        dag = _get_dag_track_b()
        assert "track_b" in dag.tags
        assert "generative_architect" in dag.tags

    @pytest.mark.dag
    def test_master_track_b_uses_trigger_dag_run(self):
        """DAG uses TriggerDagRunOperator for dataset tasks."""
        dag = _get_dag_track_b()
        tasks = {t.task_id: t for t in dag.tasks}
        
        trigger_task = tasks["trigger_ds3_stackoverflow"]
        assert "TriggerDagRunOperator" in type(trigger_task).__name__
