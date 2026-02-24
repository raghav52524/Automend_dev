"""Resolve project root — works both locally and inside Docker containers.

Inside Docker: files are mounted at /opt/airflow/
Locally:       files are at the repo root (parent of src/)

This module provides paths for Dataset 2 (Loghub), using the centralized
data structure at PROJECT_ROOT/data/ when available.
"""
import os
import sys
from pathlib import Path


def get_project_root() -> Path:
    """Return the monorepo project root regardless of local vs Docker environment."""
    # Docker sets AIRFLOW_HOME; our docker-compose mounts project at /opt/airflow
    airflow_home = os.environ.get("AIRFLOW_HOME", "")
    if airflow_home and Path("/opt/airflow/src").exists():
        return Path("/opt/airflow")
    # Local: root is four levels above this file (dataset_2_loghub/src/utils/paths.py → root)
    return Path(__file__).resolve().parent.parent.parent.parent.parent


def get_ds2_root() -> Path:
    """Return the dataset_2_loghub directory."""
    return Path(__file__).resolve().parent.parent.parent


# Centralized data paths
def get_ds2_raw_dir() -> Path:
    """Get Dataset 2 raw data directory (centralized)."""
    return get_project_root() / "data" / "raw" / "ds2_loghub"


def get_ds2_processed_dir() -> Path:
    """Get Dataset 2 processed data directory (centralized)."""
    return get_project_root() / "data" / "processed" / "ds2_loghub"


def get_ds2_interim_dir() -> Path:
    """Get Dataset 2 interim data directory (centralized)."""
    return get_project_root() / "data" / "interim"


# Legacy paths (for backward compatibility during migration)
def get_legacy_raw_dir() -> Path:
    """Get legacy data_raw directory within DS2."""
    return get_ds2_root() / "data_raw"


def get_legacy_processed_dir() -> Path:
    """Get legacy data_processed directory within DS2."""
    return get_ds2_root() / "data_processed"
