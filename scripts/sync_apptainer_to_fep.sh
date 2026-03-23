#!/usr/bin/env bash

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-fep}"
REMOTE_IMAGE_DIR="${REMOTE_IMAGE_DIR:-~/apptainer-images}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_IMAGE="${1:-${PROJECT_DIR}/container/build/dr-detection-cu121.sif}"

if [[ ! -f "${LOCAL_IMAGE}" ]]; then
    echo "Image not found: ${LOCAL_IMAGE}" >&2
    exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
    echo "rsync is required but was not found in PATH." >&2
    exit 1
fi

echo "Syncing ${LOCAL_IMAGE} to ${REMOTE_HOST}:${REMOTE_IMAGE_DIR}"

ssh "${REMOTE_HOST}" "mkdir -p ${REMOTE_IMAGE_DIR}"
rsync -avP -e ssh "${LOCAL_IMAGE}" "${REMOTE_HOST}:${REMOTE_IMAGE_DIR}/"

echo "Image sync complete."
echo "Remote image path: ${REMOTE_HOST}:${REMOTE_IMAGE_DIR}/$(basename "${LOCAL_IMAGE}")"