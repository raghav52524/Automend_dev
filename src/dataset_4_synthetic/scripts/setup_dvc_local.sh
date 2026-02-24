#!/usr/bin/env bash
# Initialize DVC with a local remote so pull_data and commit_and_push tasks can run (e.g. in Docker).
# Run once from project root: ./scripts/setup_dvc_local.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DVC_REMOTE="$PROJECT_ROOT/data/dvc_remote"

cd "$PROJECT_ROOT"

if [ ! -d .dvc ]; then
  echo "Running dvc init..."
  dvc init
else
  echo "DVC already initialized."
fi

mkdir -p "$DVC_REMOTE"
# Use relative path so the same config works in Docker and on host
if ! dvc remote list 2>/dev/null | grep -q '^local'; then
  dvc remote add -d local "data/dvc_remote"
  echo "Added default remote 'local' -> data/dvc_remote"
else
  echo "Remote 'local' already exists."
fi

echo "Done. You can run the DAG; pull_data and commit_and_push will use this local store."
