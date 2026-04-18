#!/usr/bin/env bash
# Submit an XGBoost meta-learner training job.
# All arguments are passed directly to train_meta_xgb.py.
#
# Usage:
#   APPTAINER_IMAGE=~/apptainer-images/dr-detection-cu128.sif \
#   DATASET_DIR=~/diabetic-retinopathy-detection/data/... \
#   ./scripts/submit_slurm_train_meta_xgb.sh \
#       --oof-csvs /data/oof_convnext.csv /data/oof_swin.csv /data/oof_effnet.csv \
#       --test-prob-csvs /data/test_convnext.csv /data/test_swin.csv /data/test_effnet.csv \
#       --output-dir artifacts/meta_xgb

set -euo pipefail

if [[ -z "${APPTAINER_IMAGE:-}" ]]; then
    echo "APPTAINER_IMAGE must point to your cluster .sif image." >&2
    exit 1
fi

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
JOB_SCRIPT="${JOB_SCRIPT:-${PROJECT_DIR}/scripts/slurm_train_meta_xgb_apptainer.sh}"
SLURM_OUT_DIR="${SLURM_OUT_DIR:-${PROJECT_DIR}/slurm}"

JOB_NAME="${JOB_NAME:-dr-meta-xgb}"
PARTITION="${PARTITION:-dgxh100}"
# train_meta_xgb.py defaults to tree_method=hist (CPU). Set GRES=gpu:1 only if needed.
GRES="${GRES:-gpu:0}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEMORY="${MEMORY:-64G}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
ACCOUNT="${ACCOUNT:-}"
QOS="${QOS:-}"
CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-/workspace}"
APPTAINER_EXTRA_BINDS="${APPTAINER_EXTRA_BINDS:-}"
DATASET_DIR="${DATASET_DIR:-}"

mkdir -p "${SLURM_OUT_DIR}"

SBATCH_CMD=(
    sbatch
    --job-name="${JOB_NAME}"
    --partition="${PARTITION}"
    --cpus-per-task="${CPUS_PER_TASK}"
    --mem="${MEMORY}"
    --time="${TIME_LIMIT}"
    --chdir="${PROJECT_DIR}"
    --output="${SLURM_OUT_DIR}/%x-%j.out"
)

if [[ -n "${GRES}" && "${GRES}" != "gpu:0" ]]; then
    SBATCH_CMD+=(--gres="${GRES}")
fi

if [[ -n "${ACCOUNT}" ]]; then
    SBATCH_CMD+=(--account="${ACCOUNT}")
fi

if [[ -n "${QOS}" ]]; then
    SBATCH_CMD+=(--qos="${QOS}")
fi

SBATCH_CMD+=(
    --export="ALL,APPTAINER_IMAGE=${APPTAINER_IMAGE},PROJECT_DIR=${PROJECT_DIR},CONTAINER_WORKDIR=${CONTAINER_WORKDIR},APPTAINER_EXTRA_BINDS=${APPTAINER_EXTRA_BINDS},DATASET_DIR=${DATASET_DIR},GRES=${GRES}"
    "${JOB_SCRIPT}"
)

echo "Submitting DR XGBoost meta-learner job"
echo "  Image     : ${APPTAINER_IMAGE}"
echo "  GRES      : ${GRES}"
echo "  CPUs/task : ${CPUS_PER_TASK}"
echo "  Memory    : ${MEMORY}"
echo "  Time limit: ${TIME_LIMIT}"
echo "  Partition : ${PARTITION}"
echo "  Slurm logs: ${SLURM_OUT_DIR}"

"${SBATCH_CMD[@]}" "$@"
