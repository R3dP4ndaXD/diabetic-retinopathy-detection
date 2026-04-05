#!/usr/bin/env bash
# Slurm batch job script — dual-stream training.

set -euo pipefail

if [[ -z "${APPTAINER_IMAGE:-}" ]]; then
    echo "APPTAINER_IMAGE must be set." >&2
    exit 1
fi

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-/workspace}"
APPTAINER_EXTRA_BINDS="${APPTAINER_EXTRA_BINDS:-}"
DATASET_DIR="${DATASET_DIR:-}"
NUM_GPUS="${NUM_GPUS:-1}"

mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/artifacts" "${PROJECT_DIR}/slurm"

BIND_PATHS="${PROJECT_DIR}:${CONTAINER_WORKDIR}"
if [[ -n "${APPTAINER_EXTRA_BINDS}" ]]; then
    BIND_PATHS="${BIND_PATHS},${APPTAINER_EXTRA_BINDS}"
fi
if [[ -n "${DATASET_DIR}" ]]; then
    BIND_PATHS="${BIND_PATHS},${DATASET_DIR}:/data"
fi

echo "Running on host: $(hostname)"
echo "Apptainer image: ${APPTAINER_IMAGE}"
echo "Num GPUs       : ${NUM_GPUS}"
echo "Train overrides: $*"

apptainer exec \
    --nv \
    --bind "${BIND_PATHS}" \
    --pwd "${CONTAINER_WORKDIR}" \
    "${APPTAINER_IMAGE}" \
    python train_dual_stream.py num_gpus="${NUM_GPUS}" "$@"
