#!/usr/bin/env bash

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-fep}"
REMOTE_DATASET_DIR="${REMOTE_DATASET_DIR:-~/diabetic-retinopathy-detection/data/diabetic-retinopathy-dataset}"
LOCAL_DATASET_DIR="${LOCAL_DATASET_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/diabetic-retinopathy-dataset}"

RESIZED_DIR="${LOCAL_DATASET_DIR}/resized"
TRAIN_LABELS_CSV="${LOCAL_DATASET_DIR}/trainLabels.csv"
TEST_LABELS_CSV="${LOCAL_DATASET_DIR}/testLabels.csv"

if [[ ! -d "${RESIZED_DIR}/train" ]]; then
    echo "Preprocessed train images not found at: ${RESIZED_DIR}/train" >&2
    echo "Run the local preprocessing steps first:" >&2
    echo "  ./scripts/download-dr-dataset.sh" >&2
    echo "  python scripts/crop_and_resize.py --src data/diabetic-retinopathy-dataset/train --dest data/diabetic-retinopathy-dataset/resized/train --workers 4 --skip-existing --sigmaX 10" >&2
    echo "  python scripts/crop_and_resize.py --src data/diabetic-retinopathy-dataset/test  --dest data/diabetic-retinopathy-dataset/resized/test  --workers 4 --skip-existing --sigmaX 10" >&2
    exit 1
fi

if [[ ! -f "${TRAIN_LABELS_CSV}" ]]; then
    echo "trainLabels.csv not found at: ${TRAIN_LABELS_CSV}" >&2
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

# Sync resized train + test images
rsync -avP --delete \
    -e ssh \
    "${RESIZED_DIR}/" "${REMOTE_HOST}:${REMOTE_DATASET_DIR}/resized/"

# Sync label CSVs
rsync -avP \
    -e ssh \
    "${TRAIN_LABELS_CSV}" "${REMOTE_HOST}:${REMOTE_DATASET_DIR}/trainLabels.csv"

if [[ -f "${TEST_LABELS_CSV}" ]]; then
    rsync -avP \
        -e ssh \
        "${TEST_LABELS_CSV}" "${REMOTE_HOST}:${REMOTE_DATASET_DIR}/testLabels.csv"
fi

echo ""
echo "Dataset sync complete."
echo "Remote path: ${REMOTE_HOST}:${REMOTE_DATASET_DIR}"
echo ""
echo "Next: generate train/val/test CSV splits on FEP (run once):"
echo "  ssh ${REMOTE_HOST}"
echo "  cd ~/diabetic-retinopathy-detection"
echo "  apptainer exec --nv \\"
echo "    --bind ~/diabetic-retinopathy-detection/data/diabetic-retinopathy-dataset:/data \\"
echo "    --bind ~/diabetic-retinopathy-detection:/workspace \\"
echo "    --pwd /workspace \\"
echo "    ~/apptainer-images/dr-detection-cu128.sif \\"
echo "    python scripts/split_dataset.py \\"
echo "      --data-dir /data/resized/train \\"
echo "      --csv-path /data/trainLabels.csv \\"
echo "      --test-labels-csv /data/testLabels.csv \\"
echo "      --test-data-dir /data/resized/test \\"
echo "      --train-csv-path /data/train.csv \\"
echo "      --val-csv-path /data/val.csv \\"
echo "      --test-csv-path /data/test.csv"
echo ""
echo "Then submit jobs with:"
echo "  export DATASET_DIR=~/diabetic-retinopathy-detection/data/diabetic-retinopathy-dataset"
