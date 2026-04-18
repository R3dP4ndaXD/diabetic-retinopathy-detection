#!/usr/bin/env bash
# Slurm batch job script — XGBoost meta-learner training.
# Executed inside the allocation by submit_slurm_train_meta_xgb.sh.

set -euo pipefail

if [[ -z "${APPTAINER_IMAGE:-}" ]]; then
    echo "APPTAINER_IMAGE must be set." >&2
    exit 1
fi

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-/workspace}"
APPTAINER_EXTRA_BINDS="${APPTAINER_EXTRA_BINDS:-}"
DATASET_DIR="${DATASET_DIR:-}"
GRES="${GRES:-gpu:0}"

mkdir -p "${PROJECT_DIR}/artifacts/meta_xgb" "${PROJECT_DIR}/slurm"

BIND_PATHS="${PROJECT_DIR}:${CONTAINER_WORKDIR}"
if [[ -n "${APPTAINER_EXTRA_BINDS}" ]]; then
    BIND_PATHS="${BIND_PATHS},${APPTAINER_EXTRA_BINDS}"
fi
if [[ -n "${DATASET_DIR}" ]]; then
    BIND_PATHS="${BIND_PATHS},${DATASET_DIR}:/data"
fi

echo "Running on host : $(hostname)"
echo "Project dir     : ${PROJECT_DIR}"
echo "Apptainer image : ${APPTAINER_IMAGE}"
echo "Dataset dir     : ${DATASET_DIR:-none}"
echo "Bind paths      : ${BIND_PATHS}"
echo "GRES            : ${GRES}"
echo "Meta XGB args   : $*"

APPTAINER_CMD=(apptainer exec)
if [[ -n "${GRES}" && "${GRES}" != "gpu:0" ]]; then
    APPTAINER_CMD+=(--nv)
fi

APPTAINER_CMD+=(
    --bind "${BIND_PATHS}"
    --pwd "${CONTAINER_WORKDIR}"
    "${APPTAINER_IMAGE}"
    python train_meta_xgb.py
)

"${APPTAINER_CMD[@]}" "$@"
