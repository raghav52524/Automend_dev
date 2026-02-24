"""
Centralized path definitions for the Automend MLOps Monorepo.

This module provides a single source of truth for all data paths,
supporting both local development and Docker/Airflow deployment.

Usage:
    from src.config.paths import DATA_ROOT, get_ds1_raw_dir
    
    raw_data = get_ds1_raw_dir() / "server_usage.csv"
"""

import os
from pathlib import Path

# Detect environment: Docker/Airflow uses /opt/airflow
_AIRFLOW_HOME = Path("/opt/airflow")
if _AIRFLOW_HOME.exists() and (_AIRFLOW_HOME / "dags").exists():
    PROJECT_ROOT = _AIRFLOW_HOME
else:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Main data directories
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DIR = DATA_ROOT / "raw"
INTERIM_DIR = DATA_ROOT / "interim"
PROCESSED_DIR = DATA_ROOT / "processed"

# Source code directory
SRC_DIR = PROJECT_ROOT / "src"

# DAGs directory  
DAGS_DIR = PROJECT_ROOT / "dags"

# Logs directory
LOGS_DIR = PROJECT_ROOT / "logs"


def get_project_root() -> Path:
    """Get the project root directory."""
    return PROJECT_ROOT


def get_data_root() -> Path:
    """Get the main data directory."""
    return DATA_ROOT


# Dataset 1: Alibaba Cluster Trace 2017
def get_ds1_raw_dir() -> Path:
    """Get Dataset 1 (Alibaba) raw data directory."""
    return RAW_DIR / "ds1_alibaba"


def get_ds1_processed_dir() -> Path:
    """Get Dataset 1 (Alibaba) processed data directory."""
    return PROCESSED_DIR / "ds1_alibaba"


def get_ds1_interim_path() -> Path:
    """Get Dataset 1 (Alibaba) interim file path."""
    return INTERIM_DIR / "ds1_alibaba.parquet"


# Dataset 2: Loghub
def get_ds2_raw_dir() -> Path:
    """Get Dataset 2 (Loghub) raw data directory."""
    return RAW_DIR / "ds2_loghub"


def get_ds2_processed_dir() -> Path:
    """Get Dataset 2 (Loghub) processed data directory."""
    return PROCESSED_DIR / "ds2_loghub"


def get_ds2_interim_path() -> Path:
    """Get Dataset 2 (Loghub) interim file path."""
    return INTERIM_DIR / "ds2_loghub.parquet"


# Dataset 3: StackOverflow
def get_ds3_raw_dir() -> Path:
    """Get Dataset 3 (StackOverflow) raw data directory."""
    return RAW_DIR / "ds3_stackoverflow"


def get_ds3_processed_dir() -> Path:
    """Get Dataset 3 (StackOverflow) processed data directory."""
    return PROCESSED_DIR / "ds3_stackoverflow"


def get_ds3_interim_path() -> Path:
    """Get Dataset 3 (StackOverflow) interim file path."""
    return INTERIM_DIR / "ds3_stackoverflow.jsonl"


# Dataset 4: Synthetic
def get_ds4_raw_dir() -> Path:
    """Get Dataset 4 (Synthetic) raw data directory."""
    return RAW_DIR / "ds4_synthetic"


def get_ds4_processed_dir() -> Path:
    """Get Dataset 4 (Synthetic) processed data directory."""
    return PROCESSED_DIR / "ds4_synthetic"


def get_ds4_interim_path() -> Path:
    """Get Dataset 4 (Synthetic) interim file path."""
    return INTERIM_DIR / "ds4_synthetic.jsonl"


def get_ds4_prompts_db() -> Path:
    """Get Dataset 4 (Synthetic) prompts database path.
    
    The prompts database is stored in raw/ because it contains input data
    (prompts to be processed), not output data.
    """
    return RAW_DIR / "ds4_synthetic" / "prompts.db"


# Dataset 5: Glaive
def get_ds5_raw_dir() -> Path:
    """Get Dataset 5 (Glaive) raw data directory."""
    return RAW_DIR / "ds5_glaive"


def get_ds5_processed_dir() -> Path:
    """Get Dataset 5 (Glaive) processed data directory."""
    return PROCESSED_DIR / "ds5_glaive"


def get_ds5_interim_path() -> Path:
    """Get Dataset 5 (Glaive) interim file path."""
    return INTERIM_DIR / "ds5_glaive.jsonl"


# Dataset 6: The Stack (IaC)
def get_ds6_raw_dir() -> Path:
    """Get Dataset 6 (The Stack) raw data directory."""
    return RAW_DIR / "ds6_the_stack"


def get_ds6_processed_dir() -> Path:
    """Get Dataset 6 (The Stack) processed data directory."""
    return PROCESSED_DIR / "ds6_the_stack"


def get_ds6_interim_path() -> Path:
    """Get Dataset 6 (The Stack) interim file path."""
    return INTERIM_DIR / "ds6_the_stack.jsonl"


# Combined outputs (Track A and Track B)
def get_track_a_output() -> Path:
    """Get Track A combined output path."""
    return PROCESSED_DIR / "track_A_combined.parquet"


def get_track_b_output() -> Path:
    """Get Track B combined output path."""
    return PROCESSED_DIR / "track_B_combined.jsonl"


# Helper to ensure directories exist
def ensure_dirs_exist():
    """Create all data directories if they don't exist."""
    dirs = [
        RAW_DIR, INTERIM_DIR, PROCESSED_DIR, LOGS_DIR,
        get_ds1_raw_dir(), get_ds1_processed_dir(),
        get_ds2_raw_dir(), get_ds2_processed_dir(),
        get_ds3_raw_dir(), get_ds3_processed_dir(),
        get_ds4_raw_dir(), get_ds4_processed_dir(),
        get_ds5_raw_dir(), get_ds5_processed_dir(),
        get_ds6_raw_dir(), get_ds6_processed_dir(),
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


# Export all public names
__all__ = [
    "PROJECT_ROOT",
    "DATA_ROOT",
    "RAW_DIR",
    "INTERIM_DIR",
    "PROCESSED_DIR",
    "SRC_DIR",
    "DAGS_DIR",
    "LOGS_DIR",
    "get_project_root",
    "get_data_root",
    "get_ds1_raw_dir",
    "get_ds1_processed_dir",
    "get_ds1_interim_path",
    "get_ds2_raw_dir",
    "get_ds2_processed_dir",
    "get_ds2_interim_path",
    "get_ds3_raw_dir",
    "get_ds3_processed_dir",
    "get_ds3_interim_path",
    "get_ds4_raw_dir",
    "get_ds4_processed_dir",
    "get_ds4_interim_path",
    "get_ds4_prompts_db",
    "get_ds5_raw_dir",
    "get_ds5_processed_dir",
    "get_ds5_interim_path",
    "get_ds6_raw_dir",
    "get_ds6_processed_dir",
    "get_ds6_interim_path",
    "get_track_a_output",
    "get_track_b_output",
    "ensure_dirs_exist",
]
