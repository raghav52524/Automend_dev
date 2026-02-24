# Automend MLOps Monorepo

A production-ready MLOps data pipeline integrating 6 datasets for two ML tracks, built with Apache Airflow orchestration, DVC data versioning, Great Expectations validation, Fairlearn bias detection, and centralized Slack alerting.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [ML Tracks and Datasets](#ml-tracks-and-datasets)
4. [Project Structure](#project-structure)
5. [Quick Start](#quick-start)
6. [Pipeline Components](#pipeline-components)
7. [Data Versioning with DVC](#data-versioning-with-dvc)
8. [Schema Validation and Statistics](#schema-validation-and-statistics)
9. [Anomaly Detection and Alerts](#anomaly-detection-and-alerts)
10. [Bias Detection and Mitigation](#bias-detection-and-mitigation)
11. [Testing](#testing)
12. [Troubleshooting](#troubleshooting)
13. [Evaluation Criteria Checklist](#evaluation-criteria-checklist)

---

## Project Overview

AutoMend is a self-healing MLOps platform, "Zapier for MLOps," that autonomously remediates production ML incidents through event-driven workflows. We have currently implemented the end-to-end MLOps data pipeline designed to process and prepare data for two distinct machine learning models:

- **Track A (Trigger Engine)**: A BERT-based classifier for anomaly detection
- **Track B (Generative Architect)**: A fine-tuned Llama-3 agent for infrastructure remediation

The pipeline transforms raw data from 6 different sources into standardized "Golden Formats" ready for model training.

### Key Features

- **Airflow Orchestration**: All pipelines run as DAGs with dependency management
- **DVC Data Versioning**: Track and version all data artifacts
- **Great Expectations Validation**: Automated schema and data quality checks
- **Fairlearn Bias Detection**: Data slicing and fairness analysis
- **Centralized Alerting**: Slack webhook notifications for all pipeline events
- **Comprehensive Testing**: Unit, integration, and E2E tests with pytest
- **Docker Deployment**: Complete containerized environment

---

## Architecture

### Orchestration Design

Airflow serves as the **sole orchestrator** for all pipelines. DVC is used **only for data versioning**, not for pipeline execution.

```
                         ┌─────────────────────────────────────┐
                         │      AIRFLOW (Orchestration)        │
                         └─────────────────────────────────────┘
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    ▼                                           ▼
          ┌─────────────────┐                         ┌─────────────────┐
          │  master_track_a │                         │  master_track_b │
          └─────────────────┘                         └─────────────────┘
                    │                                           │
        ┌───────────┴───────────┐           ┌───────────┬───────┴───────┬───────────┐
        ▼                       ▼           ▼           ▼               ▼           ▼
  ┌───────────┐           ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐
  │ds1_alibaba│           │ds2_loghub │ │ds3_stack- │ │ds4_synth- │ │ds5_glaive │ │ds6_iac    │
  │_pipeline  │           │_pipeline  │ │overflow   │ │etic_dag   │ │_pipeline  │ │_pipeline  │
  └───────────┘           └───────────┘ └───────────┘ └───────────┘ └───────────┘ └───────────┘
        │                       │           │           │               │           │
        └───────────┬───────────┘           └───────────┴───────┬───────┴───────────┘
                    ▼                                           ▼
          ┌─────────────────┐                         ┌─────────────────┐
          │  Track A        │                         │  Track B        │
          │  Combiner       │                         │  Combiner       │
          └─────────────────┘                         └─────────────────┘
                    │                                           │
                    ▼                                           ▼
    ┌───────────────────────────────────────────────────────────────────────┐
    │                      DVC (Data Versioning Only)                       │
    │  • dvc add data/processed/ds*/            (per-dataset versioning)    │
    │  • dvc add track_A_combined.parquet       (master Track A)            │
    │  • dvc add track_B_combined.jsonl         (master Track B)            │
    │  • dvc push                               (to remote storage)         │
    └───────────────────────────────────────────────────────────────────────┘
```

### DAG Architecture

**Master Track A DAG** (`dags/master_track_a.py`):
```
pipeline_start
      │
      ├──► trigger_ds1_alibaba ──┐
      │                          ├──► run_combiner ──► dvc_version_combined
      └──► trigger_ds2_loghub ───┘
                                          │
                                          ▼
                              track_A_combined.parquet
```

**Master Track B DAG** (`dags/master_track_b.py`):
```
pipeline_start
      │
      ├──► trigger_ds3_stackoverflow ──┐
      ├──► trigger_ds4_synthetic ──────┤
      ├──► trigger_ds5_glaive ─────────┼──► run_combiner ──► dvc_version_combined
      └──► trigger_ds6_the_stack ──────┘
                                                │
                                                ▼
                                    track_B_combined.jsonl
```

**Individual Dataset DAG Tasks** (example for DS1):
```
acquire_data ──► preprocess_data ──► validate_schema ──► schema_stats
                                                              │
                                                              ▼
                          dvc_version ◄── bias_detection ◄── detect_anomalies
```

Each individual DAG includes: acquisition → preprocessing → validation → statistics → anomaly detection → bias detection → DVC versioning.

---

## ML Tracks and Datasets

### Track A: Trigger Engine (Anomaly Classification)

**Target Model**: BERT-based Sequence Classifier (LogBERT)

**Goal**: Classify 5-minute windows of infrastructure activity into 5 states:
- 0 = Normal
- 1 = Resource_Exhaustion
- 2 = System_Crash
- 3 = Network_Failure
- 4 = Data_Drift

**Output Format (Format A)**:
```json
{"sequence_ids": [402, 115, 99, 402], "label": 1}
```
- **Schema**: Parquet with columns `sequence_ids` (List[int]) and `label` (int)
- **Final Output**: `data/processed/track_A_combined.parquet`

#### Dataset 1: Alibaba Cluster Trace 2017

| Attribute | Description |
|-----------|-------------|
| **Purpose** | Detect Resource-Based Anomalies |
| **Source** | Generated CSV files via `seed_data.py` |
| **Processing** | Feature selection → Discretization → Sliding window → Label logic |
| **Balancing** | Undersampling Normal class, oversampling failure classes |

**Preprocessing Pipeline**:
1. **Feature Selection**: Extract `cpu_utilization`, `memory_utilization`, `status`
2. **Discretization**: Bin continuous metrics into token IDs (e.g., 0-10% CPU → Token_A)
3. **Sliding Window**: Group into 5-minute windows to capture trends
4. **Label Logic**: Apply heuristic labeling based on termination status and resource usage

#### Dataset 2: LogHub (LogPAI)

| Attribute | Description |
|-----------|-------------|
| **Purpose** | Detect Log-Based Anomalies |
| **Source** | GitHub download from LogPAI repository |
| **Systems** | Linux, Hadoop, HDFS, Spark, HPC |
| **Processing** | Normalize → Merge → Sample → Label → Validate → Format |

**Preprocessing Pipeline**:
1. **Log Normalization**: Parse raw logs into unified schema with severity levels
2. **Session Grouping**: Group logs by time window
3. **Keyword Labeling**: Scan templates for failure keywords (Timeout, Unreachable, Checksum)
4. **Sequence Truncation**: Limit to 512 tokens, pad shorter sequences

---

### Track B: Generative Architect (Workflow Agent)

**Target Model**: Fine-Tuned Llama-3-8B (Instruction Tuned)

**Goal**: Map User Request + System Context into valid JSON Action Lists

**Output Format (Format B - ChatML)**:
```json
{
  "messages": [
    {"role": "system", "content": "You are AutoMend. Available Tools: [...]"},
    {"role": "user", "content": "Scale my deployment to 5 replicas"},
    {"role": "assistant", "content": "{\"workflow\": {\"steps\": [...]}}"}
  ]
}
```
- **Schema**: JSONL with ChatML structure
- **Final Output**: `data/processed/track_B_combined.jsonl`

#### Dataset 3: StackOverflow Q&A

| Attribute | Description |
|-----------|-------------|
| **Purpose** | Teach real-world user intent (The Intent Layer) |
| **Source** | Generated CSV or StackOverflow API |
| **Processing** | Tag filtering → Teacher-Student transformation → ChatML conversion |

#### Dataset 4: Synthetic MLOps Scenarios

| Attribute | Description |
|-----------|-------------|
| **Purpose** | Teach parameter precision and edge cases (The Tool Drill) |
| **Source** | SQLite prompts database + Google Gemini API |
| **Processing** | Procedural generation → Schema enforcement → ChatML conversion |
| **Requires** | `GOOGLE_API_KEY` environment variable |

#### Dataset 5: Glaive Function Calling v2

| Attribute | Description |
|-----------|-------------|
| **Purpose** | Teach JSON structural stability (The Syntax Layer) |
| **Source** | HuggingFace Hub |
| **Processing** | Subset filtering (10%) → Schema remapping → Context retention |

#### Dataset 6: The Stack (IaC)

| Attribute | Description |
|-----------|-------------|
| **Purpose** | Teach complex configuration generation (The Payload Layer) |
| **Source** | HuggingFace Hub (The Stack dataset) |
| **Processing** | Wrapper injection → PII redaction → Prompt synthesis |

---

## Project Structure

```
Automend/
├── data/
│   ├── raw/                        # Raw input data (DVC tracked)
│   │   ├── ds1_alibaba/
│   │   ├── ds2_loghub/
│   │   ├── ds3_stackoverflow/
│   │   ├── ds4_synthetic/
│   │   ├── ds5_glaive/
│   │   └── ds6_the_stack/
│   ├── interim/                    # Intermediate outputs for combiners
│   │   ├── ds1_alibaba.parquet
│   │   ├── ds2_loghub.parquet
│   │   ├── ds3_stackoverflow.jsonl
│   │   ├── ds4_synthetic.jsonl
│   │   ├── ds5_glaive.jsonl
│   │   └── ds6_the_stack.jsonl
│   └── processed/                  # Final outputs (DVC tracked)
│       ├── ds{1-6}_*/              # Per-dataset outputs
│       ├── track_A_combined.parquet
│       └── track_B_combined.jsonl
├── src/
│   ├── config/
│   │   └── paths.py               # Centralized path configuration
│   ├── utils/
│   │   ├── dvc_utils.py           # DVC utility functions
│   │   ├── alerting.py            # Centralized Slack alerting
│   │   └── ge_utils.py            # Great Expectations utilities
│   ├── dataset_1_alibaba/         # Track A: Alibaba pipeline
│   │   ├── scripts/               # preprocess.py, bias_detection.py, etc.
│   │   └── tests/
│   ├── dataset_2_loghub/          # Track A: LogHub pipeline
│   │   ├── src/                   # ingest/, normalize/, validate/, etc.
│   │   └── tests/
│   ├── dataset_3_stackoverflow/   # Track B: StackOverflow pipeline
│   ├── dataset_4_synthetic/       # Track B: Synthetic pipeline
│   ├── dataset_5_glaive/          # Track B: Glaive pipeline
│   ├── dataset_6_the_stack/       # Track B: The Stack pipeline
│   ├── combiner_track_a/          # Combines DS1 + DS2 → Parquet
│   └── combiner_track_b/          # Combines DS3-6 → JSONL
├── dags/                          # ALL Airflow DAGs (centralized)
│   ├── master_track_a.py          # Orchestrates Track A
│   ├── master_track_b.py          # Orchestrates Track B
│   ├── ds1_alibaba_dag.py
│   ├── ds2_loghub_dag.py
│   ├── ds3_stackoverflow_dag.py
│   ├── ds4_synthetic_dag.py
│   ├── ds5_glaive_dag.py
│   └── ds6_iac_dag.py
├── tests/                         # Root-level integration tests
│   ├── test_dags.py               # DAG integrity tests
│   ├── test_combiner_track_a.py
│   ├── test_combiner_track_b.py
│   ├── test_integration.py
│   ├── test_e2e.py
│   ├── test_schemas.py
│   └── test_dvc_config.py
├── scripts/
│   └── seed_all.py                # Master seed script
├── logs/                          # Airflow and alert logs
├── plugins/                       # Airflow plugins
├── .dvc/                          # DVC configuration
├── docker-compose.yaml            # Airflow Docker setup
├── requirements.txt               # Consolidated dependencies
├── pytest.ini                     # Test configuration
├── run_all_tests.py               # Test runner script
└── .env.example                   # Environment variables template
```

---

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd Automend

# Create and activate conda environment (Python 3.12)
conda create -n mlops_project python=3.12
conda activate mlops_project

# Install dependencies
pip install -r requirements.txt

# Copy environment file and configure
cp .env.example .env
# Edit .env with your API keys
```

### 2. Configure Environment Variables

Edit `.env` with required keys:

```bash
# Required
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
GOOGLE_API_KEY=your_gemini_api_key  # For DS4 synthetic generation

# Optional
HF_TOKEN=your_huggingface_token     # For DS5/DS6 downloads
SLACK_CHANNEL=#automend-alerts      # Display name only
```

### 3. Seed Test Data

```bash
# Seed local datasets (DS1, DS3, DS4)
python scripts/seed_all.py

# Seed all including external downloads (DS2, DS5, DS6)
python scripts/seed_all.py --download

# Seed specific datasets
python scripts/seed_all.py --ds1 --ds2
```

### 4. Start Airflow (Docker)

```bash
# Start all services
docker-compose up -d

# Wait for initialization (~30 seconds)
docker-compose logs airflow-init

# Access Airflow UI
# URL: http://localhost:8080
# Username: airflow
# Password: airflow
```

### 5. Run Pipelines

**Via Airflow UI (Recommended)**:
1. Navigate to http://localhost:8080
2. Enable `master_track_a` or `master_track_b` DAG
3. Click "Trigger DAG" button
4. Monitor progress in Graph or Tree view

**Via CLI**:
```bash
# Trigger from container
docker-compose exec airflow-scheduler airflow dags trigger master_track_a
```

---

## Pipeline Components

### Data Acquisition

Each dataset has its own acquisition scripts located within its source directory:

| Dataset | Acquisition Method | Script Path |
|---------|-------------------|-------------|
| DS1 (Alibaba) | Generate CSV files | `src/dataset_1_alibaba/scripts/seed_data.py` |
| DS2 (LogHub) | GitHub download | `src/dataset_2_loghub/src/ingest/download_data.py` |
| DS3 (StackOverflow) | Generate/API | `src/dataset_3_stackoverflow/scripts/seed_data.py`, `data_acquisition.py` |
| DS4 (Synthetic) | SQLite + Gemini API | `src/dataset_4_synthetic/scripts/seed_prompts.py` |
| DS5 (Glaive) | HuggingFace download | `src/dataset_5_glaive/scripts/data_acquisition.py` |
| DS6 (The Stack) | HuggingFace download | `src/dataset_6_the_stack/scripts/download/stack_iac_sample.py` |

### Data Preprocessing

Preprocessing is modular and reusable:

- **Feature Engineering**: Discretization, tokenization, sliding windows
- **Schema Normalization**: Unified format across all systems
- **Data Cleaning**: PII redaction, JSON escaping, format conversion
- **Label Assignment**: Heuristic and keyword-based labeling

### Tracking and Logging

**Airflow Logging**:
- Task logs available in Airflow UI (Task → Log)
- Logs stored in `logs/` directory

**Alert Logging**:
- All alerts logged to `logs/alerts.log`
- Alert history in `logs/alerts_history.json` (last 1000 alerts)

**Python Logging**:
- Each module uses Python's `logging` library
- Configurable log levels and handlers

---

## Data Versioning with DVC

DVC is used for **data tracking only** (not pipeline orchestration).

### Configuration

```ini
# .dvc/config
[core]
    remote = local_remote
[remote "local_remote"]
    url = ../dvc_storage
```

### Usage

```bash
# Pull data from remote
dvc pull

# Push data to remote
dvc push

# Check status
dvc status

# Add new data manually
dvc add data/processed/my_output/
```

### Download-Once Pattern

Each DAG implements a "download-once" pattern:

```python
from src.utils.dvc_utils import check_raw_data_exists, version_raw_data

# Check if data exists locally or in DVC
if check_raw_data_exists(raw_dir, project_root=PROJECT_ROOT):
    logger.info("Raw data found (local or DVC)")
else:
    # Download and version
    download_data()
    version_raw_data(raw_dir, cwd=PROJECT_ROOT)
```

---

## Schema Validation and Statistics

### Per-Dataset Schema and Statistics

Each dataset pipeline includes dedicated schema validation and statistics generation:

#### DS1 (Alibaba) - `src/dataset_1_alibaba/scripts/`

| Script | Output | Description |
|--------|--------|-------------|
| `validate_schema.py` | Validation result | Validates Format A sequences |
| `schema_stats.py` | `schema_stats.json` | Label distribution, sequence lengths, token stats |
| `anomaly_detection.py` | Anomaly list | Detects resource exhaustion, system crashes |
| `bias_detection.py` | `bias_report.json` | Fairlearn analysis, data slicing |

**Example `schema_stats.json`**:
```json
{
  "total_sequences": 100,
  "label_distribution": {
    "Normal": 45,
    "Resource_Exhaustion": 18,
    "System_Crash": 12
  },
  "sequence_length": {"min": 5, "max": 20, "mean": 12.5},
  "token_stats": {"unique_tokens": 20, "cpu_tokens": 150, "memory_tokens": 120},
  "quality_checks": {"empty_sequences": 0, "invalid_labels": 0}
}
```

#### DS2 (LogHub) - `src/dataset_2_loghub/src/validate/`

| Script | Output | Description |
|--------|--------|-------------|
| `generate_statistics.py` | `statistics_report.json` | GE validation + pandas stats |
| `validate_quality.py` | Quality report | Data quality checks |
| `src/bias/detect_bias.py` | Bias analysis | Slice-based bias detection |

**Example `statistics_report.json`** (with Great Expectations):
```json
{
  "ge_validation": {
    "success": true,
    "results": [
      {"expectation": "expect_table_columns_to_match_ordered_list", "success": true},
      {"expectation": "expect_column_values_to_be_in_set", "success": true, "column": "system"}
    ]
  },
  "statistics": {
    "total_rows": 10000,
    "rows_per_system": {"linux": 2000, "hadoop": 2000, "hdfs": 2000, "spark": 2000, "hpc": 2000},
    "severity_distribution": {"INFO": 7500, "WARN": 2000, "ERROR": 500},
    "event_type_distribution": {"normal_ops": 8000, "network_issue": 1000, "job_failed": 500}
  }
}
```

#### DS3 (StackOverflow) - `src/dataset_3_stackoverflow/scripts/`

| Script | Output | Description |
|--------|--------|-------------|
| `data_validation.py` | Validation report | Schema and format validation |
| `schema_validation.py` | Schema check | ChatML format validation |
| `bias_detection.py` | Bias report | Content bias analysis |

#### DS4 (Synthetic) - `src/dataset_4_synthetic/src/data/`

| Script | Output | Description |
|--------|--------|-------------|
| `schema_stats.py` | `schema.json`, `stats.json` | Format B schema inference, quality metrics |
| `anomaly.py` | Anomaly detection | Invalid JSON, missing fields |

**Example `stats.json`**:
```json
{
  "row_count": 500,
  "missing_user_intent": 0,
  "records_missing_valid_messages": 0
}
```

#### DS5 (Glaive) - `src/dataset_5_glaive/scripts/`

| Script | Output | Description |
|--------|--------|-------------|
| `schema_validation.py` | Validation result | ChatML structure validation |
| `anomaly_detection.py` | Anomaly report | Malformed records, outliers |
| `bias_detection.py` | Bias report | Function call distribution bias |

#### DS6 (The Stack) - `src/dataset_6_the_stack/scripts/validate/`

| Script | Output | Description |
|--------|--------|-------------|
| `schema_stats.py` | Stats report | IaC payload statistics |
| `anomaly_alerts.py` | Anomaly alerts | Invalid YAML, PII detection |
| `bias_detection.py` | Bias report | Language/framework bias analysis |

### Great Expectations Integration

DS2 uses Great Expectations for comprehensive schema validation:

```python
# From src/dataset_2_loghub/src/validate/generate_statistics.py
df_ge = ge.from_pandas(df)

# Schema: all required columns present
df_ge.expect_table_columns_to_match_ordered_list(REQUIRED_COLS)

# Value set checks
df_ge.expect_column_values_to_be_in_set("system", ALLOWED_SYSTEMS)
df_ge.expect_column_values_to_be_in_set("severity", ALLOWED_SEVERITIES)

# Null checks
for col in ["event_id", "event_template", "message"]:
    df_ge.expect_column_values_to_not_be_null(col)

# EventId format regex
df_ge.expect_column_values_to_match_regex("event_id", r"^E\d+$")

result = df_ge.validate()
```

---

## Anomaly Detection and Alerts

### Centralized Alerting System

All alerts flow through `src/utils/alerting.py`, which provides a unified Slack webhook-based alerting system:

```python
from src.utils.alerting import (
    alert_pipeline_start,
    alert_pipeline_success,
    alert_pipeline_failure,
    alert_anomaly_detected,
    alert_validation_failure,
    alert_bias_detected,
)
```

### Slack Webhook Integration

**Setup**:
1. Create a Slack Incoming Webhook for your channel (e.g., `#automend-alerts`)
2. Set `SLACK_WEBHOOK_URL` in your `.env` file
3. Optionally set `SLACK_CHANNEL` for display purposes

**Configuration** (`.env`):
```bash
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
SLACK_CHANNEL=#automend-alerts
```

**Alert Message Format**:
- Rich Slack messages with color-coded severity
- Emoji indicators per alert type (rocket for start, checkmark for success, X for failure)
- Pipeline context and timestamp
- Additional details dict for debugging

**Example Alert (Pipeline Failure)**:
```
:x: Pipeline Failed
DAG `ds1_alibaba_pipeline` failed at task `preprocess_data`.

:gear: Pipeline: ds1_alibaba_pipeline
Details:
  dag_id: ds1_alibaba_pipeline
  run_id: manual__2026-02-24T10:00:00
  failed_task: preprocess_data
  error: FileNotFoundError: Missing raw data file

:clock1: 2026-02-24 10:15:32 UTC
```

### Alert Types and Severity

| Alert Type | Severity | Slack Color | When Triggered |
|------------|----------|-------------|----------------|
| Pipeline Start | INFO | Green | DAG starts execution |
| Pipeline Success | INFO | Green | DAG completes successfully |
| Pipeline Failure | CRITICAL | Red | Task fails |
| Anomaly Detected | WARNING/ERROR | Yellow/Orange | Data anomalies found (>10 = ERROR) |
| Validation Failure | ERROR | Orange | Schema validation fails |
| Bias Detected | WARNING/ERROR | Yellow/Orange | Data bias detected (high = ERROR) |
| Data Quality Issue | WARNING | Yellow | Generic quality issues |

### Alert Logging

All alerts are logged regardless of Slack delivery:

- **File Log**: `logs/alerts.log` - Human-readable log format
- **History JSON**: `logs/alerts_history.json` - Last 1000 alerts in JSON format

```python
# Alert record structure in alerts_history.json
{
  "timestamp": "2026-02-24T10:15:32",
  "type": "pipeline_failure",
  "severity": "critical",
  "title": "Pipeline Failed",
  "message": "DAG `ds1_alibaba_pipeline` failed at task `preprocess_data`.",
  "pipeline": "ds1_alibaba_pipeline",
  "details": {"failed_task": "preprocess_data", "error": "..."}
}
```

### Airflow Callbacks

DAGs use failure callbacks for automatic Slack alerting:

```python
from src.utils.alerting import on_failure_callback, on_success_callback

default_args = {
    "on_failure_callback": on_failure_callback,  # Sends CRITICAL alert
    # "on_success_callback": on_success_callback,  # Optional: sends INFO alert
}
```

### Anomaly Detection

Each dataset implements anomaly detection with automatic alerting:

- **Missing values**: Check for nulls in required fields
- **Outliers**: Statistical outlier detection
- **Format violations**: Invalid JSON, malformed records
- **Record counts**: Unexpected drops in data volume
- **Schema violations**: Unexpected columns or types

When anomalies are detected, alerts are sent automatically:

```python
# From ds1_alibaba_dag.py
if anomalies:
    alert_anomaly_detected(
        pipeline_name="ds1_alibaba",
        anomaly_count=len(anomalies),
        anomaly_types=["Resource_Exhaustion", "System_Crash"],
        details={"critical_count": 5}
    )
```

---

## Bias Detection and Mitigation

### Per-Dataset Bias Detection

Each dataset has dedicated bias detection scripts:

| Dataset | Bias Detection Script | Slicing Features |
|---------|----------------------|------------------|
| DS1 (Alibaba) | `src/dataset_1_alibaba/scripts/bias_detection.py` | `status`, `task_type`, `label` |
| DS2 (LogHub) | `src/dataset_2_loghub/src/bias/detect_bias.py` | `system`, `severity`, `event_type` |
| DS3 (StackOverflow) | `src/dataset_3_stackoverflow/scripts/bias_detection.py` | Tags, answer quality |
| DS5 (Glaive) | `src/dataset_5_glaive/scripts/bias_detection.py` | Function types, parameter distributions |
| DS6 (The Stack) | `src/dataset_6_the_stack/scripts/validate/bias_detection.py` | Language, framework, file types |

### Fairlearn Integration (DS1 Example)

DS1 uses Fairlearn `MetricFrame` for comprehensive data slicing:

```python
# From src/dataset_1_alibaba/scripts/bias_detection.py
from fairlearn.metrics import MetricFrame, selection_rate, count

# Slice by status groups
mf_status = MetricFrame(
    metrics={
        "selection_rate": selection_rate,
        "count": count
    },
    y_true=df["is_failure"],
    y_pred=df["is_failure"],
    sensitive_features=df["status"]
)

# Log per-group metrics
for group, row in mf_status.by_group.iterrows():
    log.info(f"  {group}: failure_rate={row['selection_rate']:.2%}, count={int(row['count'])}")

# Check for disparity between groups
status_disparity = mf_status.difference(method="between_groups")
if status_disparity["selection_rate"] > 0.1:
    log.warning(f"BIAS DETECTED: High disparity: {status_disparity['selection_rate']:.4f}")
```

### Data Slicing Analysis

**Track A (DS1 Alibaba)**:
- **Slice by `status`**: Terminated, Failed, Running, Unknown
- **Slice by `task_type`**: Top 5 most common batch job categories
- **Slice by `label`**: Anomaly classes 0-4 (Normal through Data_Drift)

**Track A (DS2 LogHub)**:
- **Slice by `system`**: Linux, Hadoop, HDFS, Spark, HPC
- **Slice by `severity`**: INFO, WARN, ERROR
- **Slice by `event_type`**: auth_failure, network_issue, job_failed, etc.

**Track B (DS3-6)**:
- **Content type**: Question categories, tool types
- **Prompt complexity**: Simple vs complex workflows
- **Tool usage**: Distribution across available tools

### Mitigation Techniques

1. **Undersampling**: Reduce dominant classes (e.g., Normal label in DS1)
2. **Oversampling**: Increase minority failure classes using random replacement
3. **Fairness Constraints**: Apply during model training phase
4. **Threshold Adjustment**: Per-group decision thresholds
5. **Random Seed**: Set to 42 for reproducibility

### Example Bias Report (DS1)

Output: `data/processed/ds1_alibaba/bias_report.json`

```json
{
  "fairlearn_analysis": {
    "status_by_group": {
      "selection_rate": {"Failed": 1.0, "Terminated": 0.0, "Running": 0.0},
      "count": {"Failed": 15, "Terminated": 70, "Running": 15}
    },
    "status_disparity": 0.0832,
    "bias_detected": false
  },
  "raw_data_slicing": {
    "status_slice": {
      "distribution": {"Terminated": 70.0, "Failed": 15.0, "Running": 15.0},
      "bias_detected": false,
      "dominant_classes": {}
    },
    "task_type_slice": {"distribution": {"1": 30.0, "2": 25.0, "3": 20.0}},
    "failure_rate_by_task_type": {"1": 12.5, "2": 18.0, "3": 10.0}
  },
  "sequence_bias": {
    "label_distribution": {
      "Normal": 45.2,
      "Resource_Exhaustion": 18.3,
      "System_Crash": 12.1,
      "Network_Failure": 14.2,
      "Data_Drift": 10.2
    },
    "normal_dominance_pct": 45.2,
    "bias_detected": false
  },
  "mitigation": {
    "techniques_applied": [
      "Fairlearn MetricFrame used for slice-based bias analysis",
      "Undersampling of Normal class (label=0) to max 3x failure count",
      "Oversampling of minority failure classes (labels 1-4) using random replacement",
      "Random seed=42 for reproducibility"
    ],
    "before_mitigation": {
      "issue": "Dataset highly imbalanced — most jobs succeed (Normal)",
      "normal_dominance": "45.2% Normal before balancing"
    },
    "after_mitigation": {
      "label_distribution": {"Normal": 45.2, "Resource_Exhaustion": 18.3},
      "bias_resolved": true
    },
    "tradeoffs": [
      "Oversampling may cause overfitting on minority classes with small datasets",
      "Undersampling loses some Normal class information"
    ]
  }
}
```

---

## Testing

### Test Structure

```
tests/                           # Root-level tests
├── test_dags.py                # DAG integrity (load, tasks, dependencies)
├── test_combiner_track_a.py    # Track A combiner
├── test_combiner_track_b.py    # Track B combiner
├── test_integration.py         # Data flow integration
├── test_e2e.py                 # End-to-end pipeline
├── test_schemas.py             # Schema validation
└── test_dvc_config.py          # DVC configuration

src/dataset_*/tests/            # Per-dataset tests
└── test_pipeline.py            # Dataset-specific tests
```

### Running Tests

```bash
# Run all tests
python run_all_tests.py

# Run root tests only
python run_all_tests.py --root

# Run specific dataset tests
python run_all_tests.py --ds1
python run_all_tests.py --ds2

# Run with pytest directly
pytest tests/ -v                 # Root tests
pytest src/dataset_1_alibaba/tests/ -v  # DS1 tests

# Run with coverage
pytest --cov=src --cov-report=html

# Run by marker
pytest -m unit                   # Unit tests only
pytest -m integration            # Integration tests
pytest -m dag                    # DAG tests
```

### Test Markers

```ini
# pytest.ini
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    dag: DAG integrity tests
    dataset: Dataset-specific tests
```

---

## Troubleshooting

### Common Issues

**DAGs not showing in Airflow UI**:
```bash
# Check for import errors
docker-compose exec airflow-scheduler python /opt/airflow/dags/master_track_a.py

# View scheduler logs
docker-compose logs airflow-scheduler | tail -50
```

**Task failures**:
1. Check task logs in Airflow UI (Task → Log)
2. Common causes:
   - Missing API keys (DS4)
   - Network connectivity (DS2, DS5, DS6)
   - Missing seed data (run `seed_all.py`)

**Permission errors**:
```bash
# Fix file permissions
echo "AIRFLOW_UID=$(id -u)" >> .env
docker-compose down && docker-compose up -d
```

**DVC issues**:
```bash
# Reinitialize if needed
dvc init --force

# Check remote configuration
dvc remote list -v
```

### Docker Commands

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f
docker-compose logs airflow-scheduler

# Stop services
docker-compose down

# Reset (removes volumes)
docker-compose down -v

# Rebuild
docker-compose build --no-cache
```

## References

- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [DVC Documentation](https://dvc.org/doc)
- [Great Expectations Documentation](https://docs.greatexpectations.io/)
- [Fairlearn Documentation](https://fairlearn.org/)
- [LogPAI/LogHub Repository](https://github.com/logpai/loghub)

---

## Dataset Licenses

This project uses data from multiple sources. Each dataset has its own license that must be respected:

| Dataset | Source | License | Notes |
|---------|--------|---------|-------|
| **DS1: Alibaba Cluster Trace 2017** | [alibaba/clusterdata](https://github.com/alibaba/clusterdata) | Research Use | Available for academic/research purposes. Check repository for current terms. |
| **DS2: LogHub (LogPAI)** | [logpai/loghub](https://github.com/logpai/loghub) | **CC BY 4.0** | Must cite LogHub paper and include license notice. |
| **DS3: StackOverflow** | [StackOverflow](https://stackoverflow.com/) | **CC BY-SA 4.0** | Attribution required. ShareAlike clause applies to derivatives. |
| **DS4: Synthetic** | Generated | N/A | Synthetically generated using Google Gemini API. No external data license. |
| **DS5: Glaive Function Calling v2** | [HuggingFace](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) | Check HuggingFace | Verify license on dataset page before use. |
| **DS6: The Stack** | [bigcode/the-stack-dedup](https://huggingface.co/datasets/bigcode/the-stack-dedup) | **Permissive (varies)** | Contains permissively licensed code only. Must comply with original file licenses. |

### License Compliance

When using this project:

1. **Attribution**: Cite the original data sources when publishing results
2. **ShareAlike**: DS3 (StackOverflow) derivatives must use CC BY-SA 4.0
3. **Research Use**: DS1 (Alibaba) is primarily for academic/research purposes
4. **Opt-Out Compliance**: DS6 (The Stack) requires updating to latest version to respect opt-out requests
5. **API Terms**: DS4 synthetic generation must comply with Google Gemini API terms of service

### Required Citations

**LogHub (DS2)**:
```
Jieming Zhu, Shilin He, Pinjia He, Jinyang Liu, Michael R. Lyu.
"Loghub: A Large Collection of System Log Datasets for AI-driven Log Analytics."
IEEE International Symposium on Software Reliability Engineering (ISSRE), 2023.
```

**The Stack (DS6)**:
```
Denis Kocetkov, Raymond Li, et al.
"The Stack: 3 TB of permissively licensed source code."
Preprint, 2022.
```

---

## License

This project code is developed as part of an MLOps course assignment.

**Important**: The datasets used in this project have their own licenses as detailed above. Users must comply with all applicable dataset licenses when using or redistributing this project.
