"""
DVC Configuration and Utility Tests.

Tests that:
- DVC is initialized in the project
- .dvcignore exists
- Shared DVC utility functions work correctly

Run with: pytest tests/test_dvc_config.py -v
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# General DVC Configuration Tests
# =============================================================================

class TestDvcConfigGeneral:
    """General DVC configuration tests."""

    @pytest.mark.unit
    def test_project_has_dvc_initialized(self, project_root):
        """Project has .dvc directory (DVC initialized)."""
        dvc_dir = project_root / ".dvc"
        assert dvc_dir.exists(), "Project should have .dvc directory"

    @pytest.mark.unit
    def test_dvc_config_exists(self, project_root):
        """DVC config file exists."""
        config_path = project_root / ".dvc" / "config"
        assert config_path.exists(), ".dvc/config should exist"

    @pytest.mark.unit
    def test_dvc_config_has_local_remote(self, project_root):
        """DVC config has local remote configured."""
        config_path = project_root / ".dvc" / "config"
        content = config_path.read_text()
        assert "local_remote" in content, "DVC config should have local_remote defined"
        assert "remote = local_remote" in content, "DVC config should set local_remote as default"

    @pytest.mark.unit
    def test_dvcignore_exists(self, project_root):
        """Project has .dvcignore file."""
        dvcignore = project_root / ".dvcignore"
        assert dvcignore.exists(), ".dvcignore should exist"

    @pytest.mark.unit
    def test_dvcignore_excludes_prompts_db(self, project_root):
        """DVC ignore file excludes DS4 prompts.db."""
        dvcignore = project_root / ".dvcignore"
        content = dvcignore.read_text()
        assert "prompts.db" in content, ".dvcignore should exclude prompts.db"


# =============================================================================
# DVC Utility Module Tests
# =============================================================================

class TestDvcUtils:
    """Tests for the shared DVC utility module."""

    @pytest.mark.unit
    def test_dvc_utils_module_exists(self, project_root):
        """DVC utils module exists."""
        dvc_utils_path = project_root / "src" / "utils" / "dvc_utils.py"
        assert dvc_utils_path.exists(), "src/utils/dvc_utils.py should exist"

    @pytest.mark.unit
    def test_dvc_utils_can_import(self):
        """DVC utils module can be imported."""
        from src.utils import dvc_utils
        assert hasattr(dvc_utils, 'dvc_add')
        assert hasattr(dvc_utils, 'dvc_push')
        assert hasattr(dvc_utils, 'dvc_pull')
        assert hasattr(dvc_utils, 'dvc_version_path')
        assert hasattr(dvc_utils, 'check_raw_data_exists')
        assert hasattr(dvc_utils, 'version_raw_data')

    @pytest.mark.unit
    def test_dvc_add_calls_subprocess(self):
        """dvc_add calls subprocess with correct arguments."""
        from src.utils.dvc_utils import dvc_add
        
        with patch('src.utils.dvc_utils.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="Adding test.txt...",
                stderr=""
            )
            
            result = dvc_add("test.txt", cwd="/tmp")
            
            mock_run.assert_called_once_with(
                ["dvc", "add", "test.txt"],
                capture_output=True,
                text=True,
                cwd="/tmp",
                check=False,
            )
            assert result.returncode == 0

    @pytest.mark.unit
    def test_dvc_push_calls_subprocess(self):
        """dvc_push calls subprocess with correct arguments."""
        from src.utils.dvc_utils import dvc_push
        
        with patch('src.utils.dvc_utils.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="Pushing...",
                stderr=""
            )
            
            result = dvc_push(cwd="/tmp")
            
            mock_run.assert_called_once_with(
                ["dvc", "push"],
                capture_output=True,
                text=True,
                cwd="/tmp",
                check=False,
            )
            assert result.returncode == 0

    @pytest.mark.unit
    def test_dvc_pull_calls_subprocess(self):
        """dvc_pull calls subprocess with correct arguments."""
        from src.utils.dvc_utils import dvc_pull
        
        with patch('src.utils.dvc_utils.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="Pulling...",
                stderr=""
            )
            
            result = dvc_pull(cwd="/tmp")
            
            mock_run.assert_called_once_with(
                ["dvc", "pull"],
                capture_output=True,
                text=True,
                cwd="/tmp",
                check=False,
            )
            assert result.returncode == 0

    @pytest.mark.unit
    def test_dvc_version_path_combines_add_and_push(self):
        """dvc_version_path calls both add and push."""
        from src.utils.dvc_utils import dvc_version_path
        
        with patch('src.utils.dvc_utils.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="Success",
                stderr=""
            )
            
            result = dvc_version_path("data/processed/output.parquet", cwd="/tmp", push=True)
            
            assert mock_run.call_count == 2
            
            first_call = mock_run.call_args_list[0]
            assert first_call[0][0] == ["dvc", "add", "data/processed/output.parquet"]
            
            second_call = mock_run.call_args_list[1]
            assert second_call[0][0] == ["dvc", "push"]
            
            assert "add_result" in result
            assert "push_result" in result

    @pytest.mark.unit
    def test_dvc_version_path_skips_push_if_add_fails(self):
        """dvc_version_path skips push if add fails."""
        from src.utils.dvc_utils import dvc_version_path
        
        with patch('src.utils.dvc_utils.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stdout="",
                stderr="Error: file not found"
            )
            
            result = dvc_version_path("nonexistent.txt", cwd="/tmp", push=True)
            
            assert mock_run.call_count == 1
            assert "add_result" in result
            assert "push_result" not in result

    @pytest.mark.unit
    def test_dvc_version_path_respects_push_flag(self):
        """dvc_version_path respects push=False flag."""
        from src.utils.dvc_utils import dvc_version_path
        
        with patch('src.utils.dvc_utils.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="Success",
                stderr=""
            )
            
            result = dvc_version_path("data/output.parquet", cwd="/tmp", push=False)
            
            assert mock_run.call_count == 1
            assert "add_result" in result
            assert "push_result" not in result

    @pytest.mark.unit
    def test_check_raw_data_exists_returns_true_when_data_present(self, tmp_path):
        """check_raw_data_exists returns True when directory has files."""
        from src.utils.dvc_utils import check_raw_data_exists
        
        # Create a temp directory with some files
        raw_dir = tmp_path / "data" / "raw" / "ds_test"
        raw_dir.mkdir(parents=True)
        (raw_dir / "test_file.csv").write_text("col1,col2\n1,2")
        
        result = check_raw_data_exists(raw_dir, project_root=tmp_path)
        assert result is True

    @pytest.mark.unit
    def test_check_raw_data_exists_returns_false_when_empty(self, tmp_path):
        """check_raw_data_exists returns False when directory is empty and DVC pull fails."""
        from src.utils.dvc_utils import check_raw_data_exists
        
        # Create an empty directory
        raw_dir = tmp_path / "data" / "raw" / "ds_test"
        raw_dir.mkdir(parents=True)
        
        with patch('src.utils.dvc_utils.dvc_pull') as mock_pull:
            mock_pull.return_value = MagicMock(returncode=1)
            result = check_raw_data_exists(raw_dir, project_root=tmp_path)
        
        assert result is False

    @pytest.mark.unit
    def test_version_raw_data_calls_dvc_version_path(self):
        """version_raw_data delegates to dvc_version_path."""
        from src.utils.dvc_utils import version_raw_data
        
        with patch('src.utils.dvc_utils.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="Success",
                stderr=""
            )
            
            result = version_raw_data("data/raw/ds_test", cwd="/tmp", push=True)
            
            assert mock_run.call_count == 2  # add + push
            assert "add_result" in result
            assert "push_result" in result


# =============================================================================
# DAG DVC Integration Tests
# =============================================================================

class TestDagDvcIntegration:
    """Tests to verify DAGs have DVC versioning tasks."""

    @pytest.mark.unit
    def test_ds1_dag_has_dvc_task(self, project_root):
        """DS1 Alibaba DAG has DVC versioning task."""
        dag_path = project_root / "dags" / "ds1_alibaba_dag.py"
        content = dag_path.read_text()
        assert "dvc_version" in content, "DS1 DAG should have dvc_version task"
        assert "dvc_version_path" in content, "DS1 DAG should use dvc_version_path"

    @pytest.mark.unit
    def test_ds1_dag_has_download_once_logic(self, project_root):
        """DS1 Alibaba DAG has download-once logic."""
        dag_path = project_root / "dags" / "ds1_alibaba_dag.py"
        content = dag_path.read_text()
        assert "check_raw_data_exists" in content, "DS1 DAG should use check_raw_data_exists"
        assert "version_raw_data" in content, "DS1 DAG should use version_raw_data"

    @pytest.mark.unit
    def test_ds2_dag_has_dvc_task(self, project_root):
        """DS2 Loghub DAG has DVC versioning task."""
        dag_path = project_root / "dags" / "ds2_loghub_dag.py"
        content = dag_path.read_text()
        assert "dvc_version" in content, "DS2 DAG should have dvc_version task"
        assert "dvc_version_path" in content, "DS2 DAG should use dvc_version_path"

    @pytest.mark.unit
    def test_ds2_dag_has_download_once_logic(self, project_root):
        """DS2 Loghub DAG has download-once logic."""
        dag_path = project_root / "dags" / "ds2_loghub_dag.py"
        content = dag_path.read_text()
        assert "check_raw_data_exists" in content, "DS2 DAG should use check_raw_data_exists"
        assert "version_raw_data" in content, "DS2 DAG should use version_raw_data"

    @pytest.mark.unit
    def test_ds3_dag_has_dvc_task(self, project_root):
        """DS3 StackOverflow DAG has DVC versioning task."""
        dag_path = project_root / "dags" / "ds3_stackoverflow_dag.py"
        content = dag_path.read_text()
        assert "dvc_version" in content, "DS3 DAG should have dvc_version task"
        assert "dvc_version_path" in content, "DS3 DAG should use dvc_version_path"

    @pytest.mark.unit
    def test_ds3_dag_has_download_once_logic(self, project_root):
        """DS3 StackOverflow DAG has download-once logic."""
        dag_path = project_root / "dags" / "ds3_stackoverflow_dag.py"
        content = dag_path.read_text()
        assert "check_raw_data_exists" in content, "DS3 DAG should use check_raw_data_exists"
        assert "version_raw_data" in content, "DS3 DAG should use version_raw_data"

    @pytest.mark.unit
    def test_ds4_dag_uses_shared_utils(self, project_root):
        """DS4 Synthetic DAG uses shared DVC utils."""
        dag_path = project_root / "dags" / "ds4_synthetic_dag.py"
        content = dag_path.read_text()
        assert "from src.utils.dvc_utils" in content, "DS4 DAG should import from shared dvc_utils"

    @pytest.mark.unit
    def test_ds5_dag_tracks_folder(self, project_root):
        """DS5 Glaive DAG tracks folder instead of individual files."""
        dag_path = project_root / "dags" / "ds5_glaive_dag.py"
        content = dag_path.read_text()
        assert "DS5_PROCESSED" in content, "DS5 DAG should reference processed folder"
        assert "dvc_version_path" in content, "DS5 DAG should use dvc_version_path"

    @pytest.mark.unit
    def test_ds5_dag_has_download_once_logic(self, project_root):
        """DS5 Glaive DAG has download-once logic."""
        dag_path = project_root / "dags" / "ds5_glaive_dag.py"
        content = dag_path.read_text()
        assert "check_raw_data_exists" in content, "DS5 DAG should use check_raw_data_exists"
        assert "version_raw_data" in content, "DS5 DAG should use version_raw_data"

    @pytest.mark.unit
    def test_ds6_dag_has_dvc_task(self, project_root):
        """DS6 IaC DAG has DVC versioning task."""
        dag_path = project_root / "dags" / "ds6_iac_dag.py"
        content = dag_path.read_text()
        assert "dvc_version" in content, "DS6 DAG should have dvc_version task"
        assert "dvc_version_path" in content, "DS6 DAG should use dvc_version_path"

    @pytest.mark.unit
    def test_ds6_dag_has_download_once_logic(self, project_root):
        """DS6 IaC DAG has download-once logic."""
        dag_path = project_root / "dags" / "ds6_iac_dag.py"
        content = dag_path.read_text()
        assert "check_raw_data_exists" in content, "DS6 DAG should use check_raw_data_exists"
        assert "version_raw_data" in content, "DS6 DAG should use version_raw_data"

    @pytest.mark.unit
    def test_master_track_a_has_dvc_task(self, project_root):
        """Master Track A DAG has DVC versioning task."""
        dag_path = project_root / "dags" / "master_track_a.py"
        content = dag_path.read_text()
        assert "dvc_version_combined" in content, "Master Track A should have dvc_version_combined task"
        assert "TriggerDagRunOperator" in content, "Master Track A should use TriggerDagRunOperator"
        assert "dvc repro" not in content, "Master Track A should not use dvc repro"

    @pytest.mark.unit
    def test_master_track_b_has_dvc_task(self, project_root):
        """Master Track B DAG has DVC versioning task."""
        dag_path = project_root / "dags" / "master_track_b.py"
        content = dag_path.read_text()
        assert "dvc_version_combined" in content, "Master Track B should have dvc_version_combined task"
        assert "TriggerDagRunOperator" in content, "Master Track B should use TriggerDagRunOperator"
        assert "dvc repro" not in content, "Master Track B should not use dvc repro"


# =============================================================================
# Obsolete DVC Pipeline Files Removed Tests
# =============================================================================

class TestObsoleteDvcYamlRemoved:
    """Tests to verify obsolete dvc.yaml files have been removed."""

    @pytest.mark.unit
    def test_combiner_track_a_dvc_yaml_removed(self, combiner_track_a_dir):
        """combiner_track_a/dvc.yaml should be removed (DVC is data-only now)."""
        dvc_path = combiner_track_a_dir / "dvc.yaml"
        assert not dvc_path.exists(), (
            "src/combiner_track_a/dvc.yaml should be removed - "
            "DVC is now used for data versioning only, not pipeline definition"
        )

    @pytest.mark.unit
    def test_combiner_track_b_dvc_yaml_removed(self, combiner_track_b_dir):
        """combiner_track_b/dvc.yaml should be removed (DVC is data-only now)."""
        dvc_path = combiner_track_b_dir / "dvc.yaml"
        assert not dvc_path.exists(), (
            "src/combiner_track_b/dvc.yaml should be removed - "
            "DVC is now used for data versioning only, not pipeline definition"
        )
