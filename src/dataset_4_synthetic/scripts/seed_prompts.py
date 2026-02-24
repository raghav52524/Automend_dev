"""Seed the prompts DB with a dummy dataset (10-15 samples) for e2e testing.
Available tools are loaded from config/available_tools.json at pipeline run time; only user intents are stored here."""
import sqlite3
import sys
from pathlib import Path

# Project root - need to add both DS4 root and project root to path
DS4_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = DS4_ROOT.parent.parent
sys.path.insert(0, str(DS4_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

# Import from DS4 local src.data (not the monorepo src)
from src.dataset_4_synthetic.src.data import db_ops
from src.config.paths import get_ds4_prompts_db

DB_PATH = get_ds4_prompts_db()

# Placeholder for tool_context column (tools are injected from config/available_tools.json when running the pipeline)
TOOL_CONTEXT_PLACEHOLDER = ""

# 15 MLOps-style prompt seeds: user_intent only
SEED_INTENTS = [
    "Fix the latency on the fraud model.",
    "Scale up the recommendation service to handle more load.",
    "Restart the inference pod for the churn model.",
    "Fix the memory leak on the fraud model.",
    "Increase replicas for the API server.",
    "The model server is OOM; restart it and scale down if needed.",
    "Reduce replicas for the batch scoring job to save cost.",
    "Restart all pods for the training pipeline.",
    "Scale the feature store service to 5 replicas.",
    "The A/B test model is stuck; restart its deployment.",
    "Double the replicas for the real-time predictor.",
    "Fix high CPU on the data ingestion pod by restarting it.",
    "Scale down the dev environment to 1 replica.",
    "Restart the model monitoring service.",
    "Handle traffic spike: scale the serving layer to 10 replicas.",
]


def main():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        db_ops.setup_db(conn)
        for intent in SEED_INTENTS:
            conn.execute(
                "INSERT INTO prompts (user_intent, tool_context, processed) VALUES (?, ?, 0)",
                (intent, TOOL_CONTEXT_PLACEHOLDER),
            )
        conn.commit()
        print(f"Seeded {len(SEED_INTENTS)} prompts into {DB_PATH}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
