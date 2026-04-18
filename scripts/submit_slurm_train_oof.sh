#!/usr/bin/env bash
# Submit an OOF training job.
# Usage:
#   NUM_GPUS=2 APPTAINER_IMAGE=... DATASET_DIR=... \
#   ./scripts/submit_slurm_train_oof.sh \
#       model_name=convnext_base \
#       oof_num_folds=5 \
#       oof_source_csv_path=/data/train.csv \
#       test_csv_path=/data/test.csv

set -euo pipefail

if [[ -z "${APPTAINER_IMAGE:-}" ]]; then
    echo "APPTAINER_IMAGE must point to your cluster image, e.g. /path/to/dr-detection.sif." >&2
    exit 1
fi

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
JOB_SCRIPT="${JOB_SCRIPT:-${PROJECT_DIR}/scripts/slurm_train_oof_apptainer.sh}"
SLURM_OUT_DIR="${SLURM_OUT_DIR:-${PROJECT_DIR}/slurm}"

JOB_NAME="${JOB_NAME:-dr-train-oof}"
PARTITION="${PARTITION:-dgxh100}"
NUM_GPUS="${NUM_GPUS:-1}"
GRES="${GRES:-gpu:${NUM_GPUS}}"
CPUS_PER_TASK="${CPUS_PER_TASK:-$((8 * NUM_GPUS))}"
MEMORY="${MEMORY:-$((32 * NUM_GPUS))G}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
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
    --gres="${GRES}"
    --cpus-per-task="${CPUS_PER_TASK}"
    --mem="${MEMORY}"
    --time="${TIME_LIMIT}"
    --chdir="${PROJECT_DIR}"
    --output="${SLURM_OUT_DIR}/%x-%j.out"
)

if [[ -n "${ACCOUNT}" ]]; then
    SBATCH_CMD+=(--account="${ACCOUNT}")
fi

if [[ -n "${QOS}" ]]; then
    SBATCH_CMD+=(--qos="${QOS}")
fi

SBATCH_CMD+=(
    --export="ALL,APPTAINER_IMAGE=${APPTAINER_IMAGE},PROJECT_DIR=${PROJECT_DIR},CONTAINER_WORKDIR=${CONTAINER_WORKDIR},APPTAINER_EXTRA_BINDS=${APPTAINER_EXTRA_BINDS},DATASET_DIR=${DATASET_DIR},NUM_GPUS=${NUM_GPUS}"
    "${JOB_SCRIPT}"
)

echo "Submitting DR OOF training job"
echo "  Image     : ${APPTAINER_IMAGE}"
echo "  GPUs      : ${NUM_GPUS} (${GRES})"
echo "  CPUs/task : ${CPUS_PER_TASK}"
echo "  Memory    : ${MEMORY}"
echo "  Partition : ${PARTITION}"
echo "  Slurm logs: ${SLURM_OUT_DIR}"

"${SBATCH_CMD[@]}" "$@"