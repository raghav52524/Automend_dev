"""
Preprocessing Pipeline - Track A: Trigger Engine (Anomaly Classification)
Alibaba Cluster Trace 2017

Steps:
1. Load all 3 raw files with correct column names
2. Feature selection
3. Discretization (tokenization) - bin floats into token IDs for BERT
4. Sliding window - group into 5-minute windows
5. Label logic - assign labels 0-4
6. Class balancing - undersample Normal, oversample Failures
7. Output Format A: {"sequence_ids": [...], "label": int}
"""

import pandas as pd
import numpy as np
import json
import os
import sys
import logging
from pathlib import Path
from collections import Counter

# Add project root to path for config imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.config.paths import get_ds1_raw_dir, get_ds1_processed_dir, LOGS_DIR
    RAW_DIR = get_ds1_raw_dir()
    PROCESSED_DIR = get_ds1_processed_dir()
    LOG_DIR = LOGS_DIR
except ImportError:
    RAW_DIR = SCRIPT_DIR.parent / "data" / "raw"
    PROCESSED_DIR = SCRIPT_DIR.parent / "data" / "processed"
    LOG_DIR = PROJECT_ROOT / "logs"

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "preprocess.log", mode="a")
    ]
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
SERVER_USAGE   = RAW_DIR / "server_usage_sample.csv"
BATCH_TASK     = RAW_DIR / "batch_task_sample.csv"
SERVER_EVENT   = RAW_DIR / "server_event_sample.csv"
OUTPUT_PATH    = PROCESSED_DIR / "format_a_sequences.json"


# ── Token vocabulary ──────────────────────────────────────────────────────────
# CPU bins: 0-10, 10-20, ..., 90-100 → token IDs 100-109
# MEM bins: 0-10, 10-20, ..., 90-100 → token IDs 200-209
# Status tokens
STATUS_TOKENS = {
    "Terminated": 300,
    "Failed":     301,
    "Waiting":    302,
    "Running":    303,
    "Unknown":    304,
}

EVENT_TOKENS = {
    "add":     400,
    "remove":  401,
    "failure": 402,
    "unknown": 403,
}


def discretize_cpu(value):
    """Bin CPU % into token ID 100-109"""
    try:
        val = float(value)
        bin_idx = min(int(val // 10), 9)
        return 100 + bin_idx
    except:
        return 104  # default mid-range


def discretize_mem(value):
    """Bin memory % into token ID 200-209"""
    try:
        val = float(value)
        bin_idx = min(int(val // 10), 9)
        return 200 + bin_idx
    except:
        return 204


def load_server_usage():
    log.info("Loading server_usage_sample.csv...")
    df = pd.read_csv(SERVER_USAGE, header=None, names=[
        "time_stamp", "machine_id", "cpu_util_percent",
        "mem_util_percent", "net_in", "net_out",
        "disk_io_percent", "extra"
    ])
    log.info(f"  Loaded {len(df)} rows")
    return df


def load_batch_task():
    log.info("Loading batch_task_sample.csv...")
    df = pd.read_csv(BATCH_TASK, header=None, names=[
        "start_time", "end_time", "inst_num", "task_type",
        "job_id", "status", "plan_cpu", "plan_mem"
    ])
    log.info(f"  Loaded {len(df)} rows")
    return df


def load_server_event():
    log.info("Loading server_event_sample.csv...")
    df = pd.read_csv(SERVER_EVENT, header=None, names=[
        "time_stamp", "machine_id", "event_type",
        "event_detail", "plan_cpu", "plan_mem", "extra"
    ])
    log.info(f"  Loaded {len(df)} rows")
    return df


def assign_label(status, cpu_token, mem_token):
    """
    Label Logic (from scoping doc):
    0 = Normal
    1 = Resource_Exhaustion (Terminated + High Memory >= 70%)
    2 = System_Crash (Failed + Low Resources <= 30%)
    3 = Network_Failure (event_type = failure)
    4 = Data_Drift (default anomaly)
    """
    if status == "Failed" and cpu_token <= 103:  # CPU <= 30%
        return 2  # System_Crash
    if status == "Terminated" and mem_token >= 207:  # Mem >= 70%
        return 1  # Resource_Exhaustion
    if status in ["Failed", "Waiting"]:
        return 4  # Data_Drift
    return 0  # Normal


def create_sequences_from_server_usage(df):
    """
    Sliding window over server_usage:
    - Sort by time_stamp
    - Create windows of 5 rows (representing ~5 min intervals)
    - Each window becomes a sequence of token IDs
    """
    log.info("Creating sequences from server_usage with sliding window...")
    df = df.sort_values("time_stamp").reset_index(drop=True)

    sequences = []
    window_size = 5

    for i in range(0, len(df) - window_size + 1, window_size):
        window = df.iloc[i:i + window_size]
        token_ids = []

        for _, row in window.iterrows():
            cpu_token = discretize_cpu(row["cpu_util_percent"])
            mem_token = discretize_mem(row["mem_util_percent"])
            token_ids.extend([cpu_token, mem_token])

        # Label based on last row in window
        last_cpu = discretize_cpu(window.iloc[-1]["cpu_util_percent"])
        last_mem = discretize_mem(window.iloc[-1]["mem_util_percent"])
        label = 0  # server_usage has no status — default Normal unless high resource

        if last_mem >= 207:  # mem >= 70%
            label = 1
        elif last_cpu >= 208:  # cpu >= 80%
            label = 1

        sequences.append({"sequence_ids": token_ids, "label": label})

    log.info(f"  Created {len(sequences)} sequences from server_usage")
    return sequences


def create_sequences_from_batch_task(df):
    """
    Each batch task row becomes a sequence:
    [cpu_token, mem_token, status_token] → label
    """
    log.info("Creating sequences from batch_task...")
    sequences = []

    for _, row in df.iterrows():
        cpu_token    = discretize_cpu(row["plan_cpu"])
        mem_token    = discretize_mem(str(float(row["plan_mem"]) * 100) if pd.notna(row["plan_mem"]) else 0)
        status       = str(row["status"]) if pd.notna(row["status"]) else "Unknown"
        status_token = STATUS_TOKENS.get(status, 304)
        label        = assign_label(status, cpu_token, mem_token)

        sequences.append({
            "sequence_ids": [cpu_token, mem_token, status_token],
            "label": label
        })

    log.info(f"  Created {len(sequences)} sequences from batch_task")
    return sequences


def create_sequences_from_server_event(df):
    """
    Each server event becomes a sequence:
    [event_token, cpu_token, mem_token] → label
    """
    log.info("Creating sequences from server_event...")
    sequences = []

    for _, row in df.iterrows():
        event_type  = str(row["event_type"]).lower() if pd.notna(row["event_type"]) else "unknown"
        event_token = EVENT_TOKENS.get(event_type, 403)
        cpu_token   = discretize_cpu(row["plan_cpu"])
        mem_token   = discretize_mem(str(float(row["plan_mem"]) * 100) if pd.notna(row["plan_mem"]) else 0)

        # Network failure label for failure events
        label = 3 if event_type == "failure" else 0

        sequences.append({
            "sequence_ids": [event_token, cpu_token, mem_token],
            "label": label
        })

    log.info(f"  Created {len(sequences)} sequences from server_event")
    return sequences


def balance_classes(sequences):
    """
    Class balancing:
    - Undersample Normal (label=0)
    - Oversample Failure classes (labels 1-4)
    """
    log.info("Balancing classes...")
    label_counts = Counter([s["label"] for s in sequences])
    log.info(f"  Before balancing: {dict(label_counts)}")

    # Separate by label
    by_label = {label: [] for label in range(5)}
    for s in sequences:
        by_label[s["label"]].append(s)

    # Find max non-zero failure class count
    failure_counts = [len(by_label[l]) for l in range(1, 5) if len(by_label[l]) > 0]
    if not failure_counts:
        log.warning("  No failure samples found — skipping balancing")
        return sequences

    target_failure = max(failure_counts)
    target_normal  = min(len(by_label[0]), target_failure * 3)

    balanced = []

    # Undersample Normal
    normal_samples = by_label[0]
    if len(normal_samples) > target_normal:
        np.random.seed(42)
        indices = np.random.choice(len(normal_samples), target_normal, replace=False)
        normal_samples = [normal_samples[i] for i in indices]
    balanced.extend(normal_samples)

    # Oversample failure classes
    for label in range(1, 5):
        samples = by_label[label]
        if len(samples) == 0:
            continue
        if len(samples) < target_failure:
            np.random.seed(42)
            indices = np.random.choice(len(samples), target_failure, replace=True)
            samples = [samples[i] for i in indices]
        balanced.extend(samples)

    np.random.shuffle(balanced)
    label_counts_after = Counter([s["label"] for s in balanced])
    log.info(f"  After balancing: {dict(label_counts_after)}")
    return balanced


def run_preprocessing():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Load
    server_usage  = load_server_usage()
    batch_task    = load_batch_task()
    server_event  = load_server_event()

    # Create sequences
    sequences  = []
    sequences += create_sequences_from_server_usage(server_usage)
    sequences += create_sequences_from_batch_task(batch_task)
    sequences += create_sequences_from_server_event(server_event)

    log.info(f"Total sequences before balancing: {len(sequences)}")

    # Balance classes
    sequences = balance_classes(sequences)

    log.info(f"Total sequences after balancing: {len(sequences)}")

    # Save as JSON (Format A)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(sequences, f, indent=2)

    log.info(f"Saved Format A sequences -> {OUTPUT_PATH}")
    log.info(f"Sample: {sequences[0]}")
    return sequences


if __name__ == "__main__":
    seqs = run_preprocessing()
    print(f"\nTotal sequences: {len(seqs)}")
    print(f"Sample output: {seqs[0]}")
    print(f"Label distribution: {dict(Counter([s['label'] for s in seqs]))}")