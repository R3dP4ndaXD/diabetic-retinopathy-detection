#!/usr/bin/env bash
# Submit a SMOTE image-generation job.
#
# Usage:
#   APPTAINER_IMAGE=~/apptainer-images/dr-detection-cu124.sif \
#   DATASET_DIR=~/diabetic-retinopathy-detection/data/... \
#   ./scripts/submit_slurm_smote.sh \
#       --checkpoint artifacts/checkpoints/<run_id>/epoch=N-....ckpt \
#       --train-csv  /data/train.csv \
#       --output-dir /data/smote_synthetic \
#       --method borderline \
#       --target-classes 3 4 \
#       --target-count 2000
#
# All arguments after the script name are forwarded verbatim to
# scripts/generate_smote_images.py.

set -euo pipefail

if [[ -z "${APPTAINER_IMAGE:-}" ]]; then
    echo "APPTAINER_IMAGE must point to your cluster .sif image." >&2
    exit 1
fi

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
JOB_SCRIPT="${JOB_SCRIPT:-${PROJECT_DIR}/scripts/slurm_smote.sh}"
SLURM_OUT_DIR="${SLURM_OUT_DIR:-${PROJECT_DIR}/slurm}"

JOB_NAME="${JOB_NAME:-dr-smote}"
PARTITION="${PARTITION:-dgxh100}"
GRES="${GRES:-gpu:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"   # generous: image I/O is the bottleneck
MEMORY="${MEMORY:-64G}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"   # feature extraction + blending: ~1-2h
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
    --export="ALL,APPTAINER_IMAGE=${APPTAINER_IMAGE},PROJECT_DIR=${PROJECT_DIR},CONTAINER_WORKDIR=${CONTAINER_WORKDIR},APPTAINER_EXTRA_BINDS=${APPTAINER_EXTRA_BINDS},DATASET_DIR=${DATASET_DIR}"
    "${JOB_SCRIPT}"
)

echo "Submitting SMOTE generation job"
echo "  Image     : ${APPTAINER_IMAGE}"
echo "  GPUs      : 1 (${GRES})"
echo "  CPUs/task : ${CPUS_PER_TASK}"
echo "  Memory    : ${MEMORY}"
echo "  Time limit: ${TIME_LIMIT}"
echo "  Partition : ${PARTITION}"
echo "  Slurm logs: ${SLURM_OUT_DIR}"
echo "  Script args: $*"

"${SBATCH_CMD[@]}" "$@"
