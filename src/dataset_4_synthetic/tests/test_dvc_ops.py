"""Phase 2: DVC CLI wrapper tests (TDD). Mock subprocess.run."""
import sys
from pathlib import Path
import unittest.mock as mock

import pytest

DS4_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DS4_ROOT / "src"))

from data import dvc_ops


@mock.patch("data.dvc_ops.subprocess.run")
def test_dvc_pull_calls_subprocess_with_correct_args(mock_run):
    dvc_ops.dvc_pull()
    mock_run.assert_called_once()
    call_args = mock_run.call_args
    assert call_args[0][0] == ["dvc", "pull"]
    assert call_args[1].get("check") is True


@mock.patch("data.dvc_ops.subprocess.run")
def test_dvc_add_calls_subprocess_with_path(mock_run):
    dvc_ops.dvc_add("data/processed")
    mock_run.assert_called_once()
    call_args = mock_run.call_args
    assert call_args[0][0] == ["dvc", "add", "data/processed"]
    assert call_args[1].get("check") is True


@mock.patch("data.dvc_ops.subprocess.run")
def test_dvc_push_calls_subprocess_with_correct_args(mock_run):
    dvc_ops.dvc_push()
    mock_run.assert_called_once()
    call_args = mock_run.call_args
    assert call_args[0][0] == ["dvc", "push"]
    assert call_args[1].get("check") is True
