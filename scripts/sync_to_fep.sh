#!/usr/bin/env bash

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-fep}"
REMOTE_BASE_DIR="${REMOTE_BASE_DIR:-~/diabetic-retinopathy-detection}"
LOCAL_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

if ! command -v rsync >/dev/null 2>&1; then
    echo "rsync is required but was not found in PATH." >&2
    exit 1
fi

echo "Syncing ${LOCAL_DIR} to ${REMOTE_HOST}:${REMOTE_BASE_DIR}"

rsync -avz --delete \
    --exclude '.git/' \
    --exclude '.venv/' \
    --exclude '__pycache__/' \
    --exclude '.ipynb_checkpoints/' \
    --exclude 'logs/' \
    --exclude 'data/' \
    --exclude 'artifacts/' \
    --exclude 'container/build/' \
    --exclude 'slurm/' \
    --exclude 'outputs/' \
    -e ssh \
    "${LOCAL_DIR}/" "${REMOTE_HOST}:${REMOTE_BASE_DIR}/"

echo "Sync complete."
echo "Remote path: ${REMOTE_HOST}:${REMOTE_BASE_DIR}"