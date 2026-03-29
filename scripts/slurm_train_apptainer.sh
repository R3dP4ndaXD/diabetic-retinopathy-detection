#!/usr/bin/env bash

set -euo pipefail

if [[ -z "${APPTAINER_IMAGE:-}" ]]; then
    echo "APPTAINER_IMAGE must be set to the .sif image path before running this job." >&2
    exit 1
fi

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-/workspace}"
APPTAINER_EXTRA_BINDS="${APPTAINER_EXTRA_BINDS:-}"
# DATASET_SRC_DIR: persistent storage location in home / AFS (set by submit wrapper)
# DATASET_DIR: override to skip the copy and bind an already-prepared directory directly
DATASET_SRC_DIR="${DATASET_SRC_DIR:-}"
DATASET_DIR="${DATASET_DIR:-}"

mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/artifacts" "${PROJECT_DIR}/slurm"

# Stage dataset to $TMPDIR (local compute node disk, not counted against home quota)
if [[ -n "${DATASET_SRC_DIR}" && -z "${DATASET_DIR}" ]]; then
    STAGE_DIR="${TMPDIR:-/tmp}/dr-dataset-${SLURM_JOB_ID:-$$}"
    echo "Staging dataset from ${DATASET_SRC_DIR} to ${STAGE_DIR} ..."
    mkdir -p "${STAGE_DIR}"
    rsync -a --info=progress2 "${DATASET_SRC_DIR}/" "${STAGE_DIR}/"
    DATASET_DIR="${STAGE_DIR}"
    echo "Staging complete."
elif [[ -z "${DATASET_SRC_DIR}" && -z "${DATASET_DIR}" ]]; then
    echo "WARNING: Neither DATASET_SRC_DIR nor DATASET_DIR is set. The container will use relative data/ path." >&2
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
echo "Dataset source: ${DATASET_SRC_DIR:-none}"
echo "Dataset dir (staged): ${DATASET_DIR:-none}"
echo "Bind paths: ${BIND_PATHS}"
echo "Train overrides: $*"

apptainer exec \
    --nv \
    --bind "${BIND_PATHS}" \
    --pwd "${CONTAINER_WORKDIR}" \
    "${APPTAINER_IMAGE}" \
    python train.py "$@"