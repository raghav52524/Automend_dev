"""
Seed script for DS2 (Loghub) - generates sample CSVs for E2E testing.

Creates 10 CSV files (structured + templates for 5 systems) with realistic
data matching the LogHub 2k log schema that the normalize scripts expect.

Systems: Linux, HPC, HDFS, Hadoop, Spark
Each gets: <System>_2k.log_structured.csv and <System>_2k.log_templates.csv

Usage:
    python src/dataset_2_loghub/scripts/seed_data.py
"""

import csv
import io
import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.config.paths import get_ds2_raw_dir
    RAW_DIR = get_ds2_raw_dir() / "loghub"
except ImportError:
    RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ds2_loghub" / "loghub"

random.seed(42)

ROWS_PER_SYSTEM = 20


def _write_csv(path: Path, header: list[str], rows: list[list]):
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(header)
    writer.writerows(rows)
    path.write_text(buf.getvalue(), encoding="utf-8")


def _templates_from_rows(rows: list[list], eid_col: int, etpl_col: int) -> list[list]:
    """Deduplicate (EventId, EventTemplate) pairs and count occurrences."""
    counts: dict[str, tuple[str, int]] = {}
    for r in rows:
        eid, etpl = str(r[eid_col]), str(r[etpl_col])
        if eid in counts:
            counts[eid] = (etpl, counts[eid][1] + 1)
        else:
            counts[eid] = (etpl, 1)
    return [[eid, etpl, occ] for eid, (etpl, occ) in counts.items()]


# ---------------------------------------------------------------------------
# Linux: LineId, Month, Date, Time, Level, Component, PID, Content, EventId, EventTemplate
# ---------------------------------------------------------------------------
LINUX_COMPONENTS = ["sshd", "CRON", "su", "systemd", "kernel", "sudo"]
LINUX_TEMPLATES = [
    ("E1", "Accepted password for <*> from <*> port <*> ssh2"),
    ("E2", "Failed password for <*> from <*> port <*> ssh2"),
    ("E3", "session opened for user <*> by (uid=<*>)"),
    ("E4", "session closed for user <*>"),
    ("E5", "pam_unix(cron:session): session opened for user <*>"),
]

def _gen_linux():
    header = ["LineId", "Month", "Date", "Time", "Level", "Component", "PID", "Content", "EventId", "EventTemplate"]
    rows = []
    months = ["Jun", "Jul", "Aug"]
    for i in range(1, ROWS_PER_SYSTEM + 1):
        eid, etpl = random.choice(LINUX_TEMPLATES)
        content = etpl.replace("<*>", str(random.randint(1, 9999)))
        rows.append([
            i,
            random.choice(months),
            str(random.randint(1, 28)),
            f"{random.randint(0,23):02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d}",
            "combo",
            random.choice(LINUX_COMPONENTS),
            str(random.randint(1000, 9999)),
            content,
            eid,
            etpl,
        ])
    return header, rows


# ---------------------------------------------------------------------------
# HPC: LineId, LogId, Node, Component, State, Time, Flag, Content, EventId, EventTemplate
# ---------------------------------------------------------------------------
HPC_STATES = ["state_change.unavailable", "state_change.available", "state_change.running", "normal_ops"]
HPC_TEMPLATES = [
    ("E1", "node-<*> changed state to <*>"),
    ("E2", "temperature threshold exceeded on node-<*>"),
    ("E3", "job <*> started on node-<*>"),
    ("E4", "memory check passed on node-<*>"),
    ("E5", "heartbeat received from node-<*>"),
]

def _gen_hpc():
    header = ["LineId", "LogId", "Node", "Component", "State", "Time", "Flag", "Content", "EventId", "EventTemplate"]
    rows = []
    for i in range(1, ROWS_PER_SYSTEM + 1):
        eid, etpl = random.choice(HPC_TEMPLATES)
        node = f"node-{random.randint(1, 256)}"
        content = etpl.replace("<*>", str(random.randint(1, 9999)))
        rows.append([
            i,
            f"log-{i}",
            node,
            random.choice(["unix.hw", "unix.sw", "bgl.kernel"]),
            random.choice(HPC_STATES),
            str(1077804742 + i * 60),
            str(random.choice([0, 0, 0, 1])),
            content,
            eid,
            etpl,
        ])
    return header, rows


# ---------------------------------------------------------------------------
# HDFS: LineId, Date, Time, Pid, Level, Component, Content, EventId, EventTemplate
# ---------------------------------------------------------------------------
HDFS_COMPONENTS = ["dfs.DataNode$DataXceiver", "dfs.FSNamesystem", "dfs.DataNode$PacketResponder"]
HDFS_LEVELS = ["INFO", "INFO", "INFO", "WARN", "ERROR"]
HDFS_TEMPLATES = [
    ("E1", "Receiving block <*> src: <*> dest: <*>"),
    ("E2", "Received block <*> of size <*> from <*>"),
    ("E3", "BLOCK* NameSystem.addStoredBlock: blockMap updated: <*> is added to <*>"),
    ("E4", "PacketResponder <*> for block <*> terminating"),
    ("E5", "Deleting block <*> file <*>"),
]

def _gen_hdfs():
    header = ["LineId", "Date", "Time", "Pid", "Level", "Component", "Content", "EventId", "EventTemplate"]
    rows = []
    for i in range(1, ROWS_PER_SYSTEM + 1):
        eid, etpl = random.choice(HDFS_TEMPLATES)
        content = etpl.replace("<*>", str(random.randint(100000, 999999)))
        rows.append([
            i,
            f"08110{random.randint(1,9)}",
            f"{random.randint(0,23):02d}{random.randint(0,59):02d}{random.randint(0,59):02d}",
            str(random.randint(1000, 9999)),
            random.choice(HDFS_LEVELS),
            random.choice(HDFS_COMPONENTS),
            content,
            eid,
            etpl,
        ])
    return header, rows


# ---------------------------------------------------------------------------
# Hadoop: LineId, Date, Time, Level, Process, Component, Content, EventId, EventTemplate
# ---------------------------------------------------------------------------
HADOOP_COMPONENTS = ["RMAppManager", "ResourceManager", "ApplicationMasterLauncher", "LeafQueue"]
HADOOP_TEMPLATES = [
    ("E1", "Application <*> submitted by user <*>"),
    ("E2", "Added Application Attempt <*> to scheduler"),
    ("E3", "AM allocated container <*> on host <*>"),
    ("E4", "Recovering app attempt <*>"),
    ("E5", "Stored application <*> in state store"),
]

def _gen_hadoop():
    header = ["LineId", "Date", "Time", "Level", "Process", "Component", "Content", "EventId", "EventTemplate"]
    rows = []
    levels = ["INFO", "INFO", "INFO", "WARN", "ERROR"]
    for i in range(1, ROWS_PER_SYSTEM + 1):
        eid, etpl = random.choice(HADOOP_TEMPLATES)
        content = etpl.replace("<*>", str(random.randint(1, 9999)))
        rows.append([
            i,
            f"2015-10-{random.randint(10,28)}",
            f"{random.randint(0,23):02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d},{random.randint(0,999):03d}",
            random.choice(levels),
            str(random.randint(1000, 9999)),
            random.choice(HADOOP_COMPONENTS),
            content,
            eid,
            etpl,
        ])
    return header, rows


# ---------------------------------------------------------------------------
# Spark: LineId, Date, Time, Level, Component, Content, EventId, EventTemplate
# ---------------------------------------------------------------------------
SPARK_COMPONENTS = ["TaskSetManager", "BlockManagerInfo", "DAGScheduler", "SparkContext"]
SPARK_TEMPLATES = [
    ("E1", "Starting task <*> in stage <*>"),
    ("E2", "Finished task <*> in stage <*> (TID <*>)"),
    ("E3", "Added broadcast_<*> in memory on <*>"),
    ("E4", "Removed broadcast_<*> on <*> in memory"),
    ("E5", "Job <*> finished: <*> at <*>"),
]

def _gen_spark():
    header = ["LineId", "Date", "Time", "Level", "Component", "Content", "EventId", "EventTemplate"]
    rows = []
    levels = ["INFO", "INFO", "INFO", "WARN", "ERROR"]
    for i in range(1, ROWS_PER_SYSTEM + 1):
        eid, etpl = random.choice(SPARK_TEMPLATES)
        content = etpl.replace("<*>", str(random.randint(1, 9999)))
        rows.append([
            i,
            f"17/06/{random.randint(1,28):02d}",
            f"{random.randint(0,23):02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d}",
            random.choice(levels),
            random.choice(SPARK_COMPONENTS),
            content,
            eid,
            etpl,
        ])
    return header, rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
GENERATORS = {
    "Linux":  (_gen_linux,  "Linux_2k"),
    "HPC":    (_gen_hpc,    "HPC_2k"),
    "HDFS":   (_gen_hdfs,   "HDFS_2k"),
    "Hadoop": (_gen_hadoop, "Hadoop_2k"),
    "Spark":  (_gen_spark,  "Spark_2k"),
}

TEMPLATE_HEADER = ["EventId", "EventTemplate", "Occurrences"]


def main():
    print("Generating DS2 (Loghub) seed data...")
    print(f"Output directory: {RAW_DIR}")

    for system, (gen_fn, prefix) in GENERATORS.items():
        header, rows = gen_fn()

        struct_path = RAW_DIR / system / f"{prefix}.log_structured.csv"
        _write_csv(struct_path, header, rows)
        print(f"  Created: {system}/{struct_path.name} ({len(rows)} rows)")

        eid_col = header.index("EventId")
        etpl_col = header.index("EventTemplate")
        tpl_rows = _templates_from_rows(rows, eid_col, etpl_col)

        tpl_path = RAW_DIR / system / f"{prefix}.log_templates.csv"
        _write_csv(tpl_path, TEMPLATE_HEADER, tpl_rows)
        print(f"  Created: {system}/{tpl_path.name} ({len(tpl_rows)} templates)")

    print(f"\nDS2 seed data generation complete!")
    print(f"Files saved to: {RAW_DIR}")
    return True


if __name__ == "__main__":
    main()
