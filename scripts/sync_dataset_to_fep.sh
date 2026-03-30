#!/usr/bin/env bash

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-fep}"
REMOTE_DATASET_DIR="${REMOTE_DATASET_DIR:-~/diabetic-retinopathy-detection/data/diabetic-retinopathy-dataset_ben_260}"
LOCAL_DATASET_DIR="${LOCAL_DATASET_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/diabetic-retinopathy-dataset}"

RESIZED_DIR="${LOCAL_DATASET_DIR}/resized"
LABELS_CSV="${LOCAL_DATASET_DIR}/trainLabels.csv"

if [[ ! -d "${RESIZED_DIR}" ]]; then
    echo "Preprocessed images not found at: ${RESIZED_DIR}" >&2
    echo "Run the local preprocessing steps first:" >&2
    echo "  ./scripts/download-dr-dataset.sh" >&2
    echo "  python scripts/crop_and_resize.py --src data/diabetic-retinopathy-dataset/train --dest data/diabetic-retinopathy-dataset/resized/train" >&2
    exit 1
fi

if [[ ! -f "${LABELS_CSV}" ]]; then
    echo "trainLabels.csv not found at: ${LABELS_CSV}" >&2
    echo "Ensure the raw dataset was extracted: ./scripts/download-dr-dataset.sh" >&2
    exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
    echo "rsync is required but was not found in PATH." >&2
    exit 1
fi

echo "Syncing preprocessed images and labels to ${REMOTE_HOST}:${REMOTE_DATASET_DIR}"
echo "This may take several minutes on first transfer..."
echo ""

ssh "${REMOTE_HOST}" "mkdir -p ${REMOTE_DATASET_DIR}"

# Sync preprocessed images only (skip raw train/ and test/ which are large and unneeded)
rsync -avP --delete \
    -e ssh \
    "${RESIZED_DIR}/" "${REMOTE_HOST}:${REMOTE_DATASET_DIR}/resized/"

# Sync the labels CSV needed to generate train/val splits on FEP
rsync -avP \
    -e ssh \
    "${LABELS_CSV}" "${REMOTE_HOST}:${REMOTE_DATASET_DIR}/trainLabels.csv"

echo ""
echo "Dataset sync complete."
echo "Remote path: ${REMOTE_HOST}:${REMOTE_DATASET_DIR}"
echo ""
echo "Next: generate train/val CSV splits on FEP (run once):"
echo "  ssh ${REMOTE_HOST}"
echo "  cd ~/diabetic-retinopathy-detection"
echo "  apptainer exec --nv \\"
echo "    --bind ~/diabetic-retinopathy-detection/data/diabetic-retinopathy-dataset:/data \\"
echo "    --bind ~/diabetic-retinopathy-detection:/workspace \\"
echo "    --pwd /workspace \\"
echo "    ~/apptainer-images/dr-detection-cu121.sif \\"
echo "    python scripts/split_dataset.py \\"
echo "      --data_dir /data/resized/train \\"
echo "      --csv_path /data/trainLabels.csv \\"
echo "      --train_csv_path data/diabetic-retinopathy-dataset/train.csv \\"
echo "      --val_csv_path data/diabetic-retinopathy-dataset/val.csv"
echo ""
echo "Then submit jobs with:"
echo "  export DATASET_DIR=~/diabetic-retinopathy-detection/data/diabetic-retinopathy-dataset"
