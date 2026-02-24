"""
DVC Utility Functions for Data Versioning.

Provides reusable functions for DVC operations across all DAGs:
- dvc_add: Add file/folder to DVC tracking
- dvc_push: Push tracked files to remote storage
- dvc_pull: Pull tracked files from remote storage
- dvc_version_path: Combined add + push for convenience

These functions are used by Airflow DAGs to version data outputs
after each pipeline run.
"""

import logging
import subprocess
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)


def dvc_add(
    path: Union[str, Path],
    cwd: Optional[Union[str, Path]] = None
) -> subprocess.CompletedProcess:
    """
    Add a file or folder to DVC tracking.
    
    Args:
        path: Path to the file or folder to track
        cwd: Working directory for the command (defaults to current dir)
    
    Returns:
        CompletedProcess with return code and output
    """
    path_str = str(path)
    logger.info("DVC add: %s", path_str)
    
    result = subprocess.run(
        ["dvc", "add", path_str],
        capture_output=True,
        text=True,
        cwd=cwd,
        check=False,
    )
    
    if result.returncode == 0:
        logger.info("DVC tracked successfully: %s", path_str)
    else:
        logger.warning("DVC add failed for %s: %s", path_str, result.stderr)
    
    return result


def dvc_push(cwd: Optional[Union[str, Path]] = None) -> subprocess.CompletedProcess:
    """
    Push all DVC-tracked files to the configured remote storage.
    
    Args:
        cwd: Working directory for the command (defaults to current dir)
    
    Returns:
        CompletedProcess with return code and output
    """
    logger.info("DVC push: pushing to remote")
    
    result = subprocess.run(
        ["dvc", "push"],
        capture_output=True,
        text=True,
        cwd=cwd,
        check=False,
    )
    
    if result.returncode == 0:
        logger.info("DVC push completed successfully")
    else:
        logger.warning("DVC push failed: %s", result.stderr)
    
    return result


def dvc_pull(cwd: Optional[Union[str, Path]] = None) -> subprocess.CompletedProcess:
    """
    Pull DVC-tracked files from the configured remote storage.
    
    Args:
        cwd: Working directory for the command (defaults to current dir)
    
    Returns:
        CompletedProcess with return code and output
    """
    logger.info("DVC pull: pulling from remote")
    
    result = subprocess.run(
        ["dvc", "pull"],
        capture_output=True,
        text=True,
        cwd=cwd,
        check=False,
    )
    
    if result.returncode == 0:
        logger.info("DVC pull completed successfully")
    else:
        logger.warning("DVC pull failed: %s", result.stderr)
    
    return result


def dvc_version_path(
    path: Union[str, Path],
    cwd: Optional[Union[str, Path]] = None,
    push: bool = True
) -> dict:
    """
    Version a path with DVC: add it to tracking and optionally push to remote.
    
    This is a convenience function that combines dvc_add and dvc_push.
    
    Args:
        path: Path to the file or folder to version
        cwd: Working directory for the commands (defaults to current dir)
        push: Whether to push to remote after adding (default: True)
    
    Returns:
        Dict with 'add_result' and optionally 'push_result'
    """
    results = {}
    
    add_result = dvc_add(path, cwd=cwd)
    results["add_result"] = {
        "returncode": add_result.returncode,
        "stdout": add_result.stdout,
        "stderr": add_result.stderr,
    }
    
    if push and add_result.returncode == 0:
        push_result = dvc_push(cwd=cwd)
        results["push_result"] = {
            "returncode": push_result.returncode,
            "stdout": push_result.stdout,
            "stderr": push_result.stderr,
        }
    
    return results


def check_raw_data_exists(
    raw_dir: Union[str, Path],
    project_root: Optional[Union[str, Path]] = None
) -> bool:
    """
    Check if raw data exists locally or can be pulled from DVC remote.
    
    This implements the "download-once" pattern:
    1. Check if data exists locally (non-empty directory)
    2. If not, try to pull from DVC remote
    3. Return True if data is available after these checks
    
    Args:
        raw_dir: Path to the raw data directory to check
        project_root: Project root for DVC commands (defaults to raw_dir's grandparent)
    
    Returns:
        True if raw data exists or was successfully pulled, False otherwise
    """
    raw_path = Path(raw_dir)
    
    # Determine project root for DVC commands
    if project_root is None:
        # raw_dir is typically data/raw/ds{N}_name, so project root is 3 levels up
        project_root = raw_path.parent.parent.parent
    
    # Check if directory exists and has files
    if raw_path.exists() and any(raw_path.iterdir()):
        logger.info("Raw data exists locally at %s", raw_path)
        return True
    
    # Try to pull from DVC remote
    logger.info("Raw data not found locally, attempting DVC pull")
    result = dvc_pull(cwd=str(project_root))
    
    # Check again after pull
    if raw_path.exists() and any(raw_path.iterdir()):
        logger.info("Raw data pulled successfully from DVC remote")
        return True
    
    logger.info("Raw data not available (not local, not in DVC remote)")
    return False


def version_raw_data(
    raw_dir: Union[str, Path],
    cwd: Optional[Union[str, Path]] = None,
    push: bool = True
) -> dict:
    """
    Version raw data after successful download.
    
    This should be called after downloading fresh data to track it with DVC.
    It's a thin wrapper around dvc_version_path for semantic clarity.
    
    Args:
        raw_dir: Path to the raw data directory to version
        cwd: Working directory for DVC commands (defaults to current dir)
        push: Whether to push to remote after adding (default: True)
    
    Returns:
        Dict with 'add_result' and optionally 'push_result'
    """
    logger.info("Versioning raw data: %s", raw_dir)
    return dvc_version_path(raw_dir, cwd=cwd, push=push)
