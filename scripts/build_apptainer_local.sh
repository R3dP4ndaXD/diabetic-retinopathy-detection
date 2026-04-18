#!/usr/bin/env bash

set -euo pipefail

if ! command -v apptainer >/dev/null 2>&1; then
    echo "apptainer is required but was not found in PATH." >&2
    exit 1
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APPTAINER_DEF="${APPTAINER_DEF:-${PROJECT_DIR}/container/apptainer.def}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/container/build}"
IMAGE_NAME="${IMAGE_NAME:-dr-detection-cu128}"
OUTPUT_IMAGE="${OUTPUT_IMAGE:-${OUTPUT_DIR}/${IMAGE_NAME}.sif}"
BUILD_MODE="${BUILD_MODE:-fakeroot}"

mkdir -p "${OUTPUT_DIR}"

BUILD_ARGS=()
if [[ -n "${APPTAINER_TMPDIR:-}" ]]; then
    export APPTAINER_TMPDIR
fi

case "${BUILD_MODE}" in
    fakeroot)
        BUILD_ARGS+=(--fakeroot)
        ;;
    sudo)
        ;;
    *)
        echo "Unsupported BUILD_MODE: ${BUILD_MODE}. Use 'fakeroot' or 'sudo'." >&2
        exit 1
        ;;
esac

echo "Definition file: ${APPTAINER_DEF}"
echo "Output image: ${OUTPUT_IMAGE}"
echo "Build mode: ${BUILD_MODE}"

if [[ "${BUILD_MODE}" == "sudo" ]]; then
    sudo apptainer build "${BUILD_ARGS[@]}" "${OUTPUT_IMAGE}" "${APPTAINER_DEF}"
else
    apptainer build "${BUILD_ARGS[@]}" "${OUTPUT_IMAGE}" "${APPTAINER_DEF}"
fi

echo "Build complete: ${OUTPUT_IMAGE}"