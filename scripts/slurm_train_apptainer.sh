#!/usr/bin/env bash

set -euo pipefail

if [[ -z "${APPTAINER_IMAGE:-}" ]]; then
    echo "APPTAINER_IMAGE must be set to the .sif image path before running this job." >&2
    exit 1
fi

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-/workspace}"
APPTAINER_EXTRA_BINDS="${APPTAINER_EXTRA_BINDS:-}"
# DATASET_DIR: dataset location on shared storage to be bound directly as /data.
DATASET_DIR="${DATASET_DIR:-}"

mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/artifacts" "${PROJECT_DIR}/slurm"

if [[ -z "${DATASET_DIR}" ]]; then
    echo "WARNING: DATASET_DIR is not set. The container will use relative data/ path." >&2
fi

BIND_PATHS="${PROJECT_DIR}:${CONTAINER_WORKDIR}"
if [[ -n "${APPTAINER_EXTRA_BINDS}" ]]; then
    BIND_PATHS="${BIND_PATHS},${APPTAINER_EXTRA_BINDS}"
fi
if [[ -n "${DATASET_DIR}" ]]; then
    # Bind as /data inside the container so cluster config paths are stable
    BIND_PATHS="${BIND_PATHS},${DATASET_DIR}:/data"
fi

echo "Running on host: $(hostname)"
echo "Project dir: ${PROJECT_DIR}"
echo "Apptainer image: ${APPTAINER_IMAGE}"
echo "Dataset dir (direct bind): ${DATASET_DIR:-none}"
echo "Bind paths: ${BIND_PATHS}"
echo "Train overrides: $*"

apptainer exec \
    --nv \
    --bind "${BIND_PATHS}" \
    --pwd "${CONTAINER_WORKDIR}" \
    "${APPTAINER_IMAGE}" \
    python train.py "$@"