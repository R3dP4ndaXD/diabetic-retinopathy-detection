#!/usr/bin/env bash

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-fep}"
REMOTE_PATH="${REMOTE_PATH:-.}"
MOUNT_DIR="${1:-$HOME/mnt/fep-home}"

if ! command -v sshfs >/dev/null 2>&1; then
    echo "sshfs is required but was not found in PATH." >&2
    echo "Install it with your package manager, for example: sudo apt install sshfs" >&2
    exit 1
fi

mkdir -p "${MOUNT_DIR}"

if mountpoint -q "${MOUNT_DIR}"; then
    echo "Mount point is already active: ${MOUNT_DIR}"
    exit 0
fi

sshfs "${REMOTE_HOST}:${REMOTE_PATH}" "${MOUNT_DIR}" \
    -o reconnect \
    -o ServerAliveInterval=15 \
    -o ServerAliveCountMax=3

echo "Mounted ${REMOTE_HOST}:${REMOTE_PATH} at ${MOUNT_DIR}"
echo "Unmount with: fusermount -u ${MOUNT_DIR}"