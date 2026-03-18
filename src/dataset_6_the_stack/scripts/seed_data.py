"""
Seed script for DS6 (The Stack IaC) - generates sample parquet chunks for E2E testing.

Creates a single chunk_0000.parquet with 20 sample Kubernetes/IaC YAML records
matching the HuggingFace The Stack schema expected by the analysis and preprocess scripts.

Usage:
    python src/dataset_6_the_stack/scripts/seed_data.py
"""

import random
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
DS6_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = DS6_ROOT.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.config.paths import get_ds6_raw_dir
    RAW_DIR = get_ds6_raw_dir()
except ImportError:
    RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ds6_the_stack"

random.seed(42)

SAMPLE_MANIFESTS = [
    {
        "content": """apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-inference-server
  labels:
    app: inference
spec:
  replicas: 3
  selector:
    matchLabels:
      app: inference
  template:
    metadata:
      labels:
        app: inference
    spec:
      containers:
      - name: model-server
        image: tensorflow/serving:latest
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "8Gi"
          requests:
            memory: "4Gi"
        ports:
        - containerPort: 8501
""",
        "iac_hint": "k8s_workload",
    },
    {
        "content": """apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: fraud-detector
spec:
  predictor:
    model:
      modelFormat:
        name: sklearn
      storageUri: gs://models/fraud-v2
      resources:
        limits:
          nvidia.com/gpu: 1
""",
        "iac_hint": "kserve",
    },
    {
        "content": """apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: recommendation-engine
spec:
  predictors:
  - graph:
      implementation: SKLEARN_SERVER
      modelUri: s3://models/reco
      name: classifier
    name: default
    replicas: 2
""",
        "iac_hint": "seldon",
    },
    {
        "content": """apiVersion: v1
kind: Service
metadata:
  name: mlflow-tracking
spec:
  selector:
    app: mlflow
  ports:
  - port: 5000
    targetPort: 5000
  type: ClusterIP
""",
        "iac_hint": "k8s_config",
    },
    {
        "content": """apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: kubeflow-pipeline-db
spec:
  serviceName: pipeline-db
  replicas: 1
  selector:
    matchLabels:
      app: pipeline-db
  template:
    metadata:
      labels:
        app: pipeline-db
    spec:
      containers:
      - name: postgres
        image: postgres:15
        env:
        - name: POSTGRES_DB
          value: mlpipeline
        resources:
          limits:
            memory: "2Gi"
""",
        "iac_hint": "k8s_workload",
    },
    {
        "content": """apiVersion: batch/v1
kind: Job
metadata:
  name: gpu-training-job
spec:
  template:
    spec:
      containers:
      - name: trainer
        image: pytorch/pytorch:2.0-cuda11.7
        command: ["python", "train.py"]
        resources:
          limits:
            nvidia.com/gpu: 4
            memory: "32Gi"
      restartPolicy: Never
  backoffLimit: 3
""",
        "iac_hint": "k8s_workload",
    },
    {
        "content": """apiVersion: v1
kind: ConfigMap
metadata:
  name: inference-config
data:
  MODEL_NAME: "churn-predictor"
  SERVING_PORT: "8080"
  GPU_MEMORY_FRACTION: "0.5"
  BATCH_SIZE: "32"
""",
        "iac_hint": "k8s_config",
    },
    {
        "content": """apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: gpu-device-plugin
spec:
  selector:
    matchLabels:
      name: nvidia-device-plugin
  template:
    metadata:
      labels:
        name: nvidia-device-plugin
    spec:
      containers:
      - name: nvidia-device-plugin
        image: nvcr.io/nvidia/k8s-device-plugin:v0.14.0
        resources:
          limits:
            nvidia.com/gpu: 1
""",
        "iac_hint": "k8s_workload",
    },
]

LICENSES = ["mit", "apache-2.0", "bsd-3-clause"]
REPO_NAMES = [
    "mlops-team/k8s-ml-infra",
    "data-eng/kubeflow-pipelines",
    "aiml/serving-configs",
    "platform/gpu-workloads",
]


def _make_row(idx: int, manifest: dict) -> dict:
    content = manifest["content"]
    return {
        "content": content,
        "ext": "yaml",
        "lang": "YAML",
        "size": len(content.encode()),
        "avg_line_length": len(content) / max(content.count("\n"), 1),
        "max_line_length": max(len(l) for l in content.split("\n")),
        "alphanum_fraction": sum(c.isalnum() for c in content) / max(len(content), 1),
        "hexsha": f"{idx:040x}",
        "max_stars_repo_path": f"manifests/{manifest['iac_hint']}/deploy.yaml",
        "max_stars_repo_name": random.choice(REPO_NAMES),
        "max_stars_repo_licenses": [random.choice(LICENSES)],
        "max_issues_repo_path": f"manifests/{manifest['iac_hint']}/deploy.yaml",
        "max_forks_repo_path": f"manifests/{manifest['iac_hint']}/deploy.yaml",
        "max_issues_repo_licenses": [random.choice(LICENSES)],
        "max_forks_repo_licenses": [random.choice(LICENSES)],
    }


def main():
    print("Generating DS6 (The Stack IaC) seed data...")
    print(f"Output directory: {RAW_DIR}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(20):
        manifest = SAMPLE_MANIFESTS[i % len(SAMPLE_MANIFESTS)]
        rows.append(_make_row(i, manifest))

    chunk_path = RAW_DIR / "chunk_0000.parquet"
    pq.write_table(pa.Table.from_pylist(rows), chunk_path, compression="snappy")
    print(f"  Created: {chunk_path.name} ({len(rows)} rows)")

    print(f"\nDS6 seed data generation complete!")
    print(f"Files saved to: {RAW_DIR}")
    return True


if __name__ == "__main__":
    main()
