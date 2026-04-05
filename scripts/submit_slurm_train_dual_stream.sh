#!/usr/bin/env bash
# Submit a dual-stream training job.
# Usage:
#   NUM_GPUS=2 APPTAINER_IMAGE=... DATASET_DIR=... \
#   ./scripts/submit_slurm_train_dual_stream.sh \
#       rgb_model_name=convnext_base wav_model_name=efficientnet_b0 \
#       fusion_type=mlp image_size=224 batch_size=32

set -euo pipefail

if [[ -z "${APPTAINER_IMAGE:-}" ]]; then
    echo "APPTAINER_IMAGE must be set." >&2
    exit 1
fi

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
JOB_SCRIPT="${JOB_SCRIPT:-${PROJECT_DIR}/scripts/slurm_train_dual_stream_apptainer.sh}"
SLURM_OUT_DIR="${SLURM_OUT_DIR:-${PROJECT_DIR}/slurm}"

JOB_NAME="${JOB_NAME:-dr-dual-stream}"
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

if [[ -n "${ACCOUNT}" ]]; then SBATCH_CMD+=(--account="${ACCOUNT}"); fi
if [[ -n "${QOS}" ]];     then SBATCH_CMD+=(--qos="${QOS}"); fi

SBATCH_CMD+=(
    --export="ALL,APPTAINER_IMAGE=${APPTAINER_IMAGE},PROJECT_DIR=${PROJECT_DIR},CONTAINER_WORKDIR=${CONTAINER_WORKDIR},APPTAINER_EXTRA_BINDS=${APPTAINER_EXTRA_BINDS},DATASET_DIR=${DATASET_DIR},NUM_GPUS=${NUM_GPUS}"
    "${JOB_SCRIPT}"
)

echo "Submitting DR dual-stream training job"
echo "  Image     : ${APPTAINER_IMAGE}"
echo "  GPUs      : ${NUM_GPUS}"
echo "  Partition : ${PARTITION}"

"${SBATCH_CMD[@]}" "$@"
