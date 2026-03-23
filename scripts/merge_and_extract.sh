#!/bin/bash

set -euo pipefail

# Set dataset directory
DATASET_DIR="data/diabetic-retinopathy-dataset"

# Function to log messages
log() {
    echo "$(date +"%Y-%m-%d %H:%M:%S") $1"
}

# Function to merge and extract zip files
merge_and_extract_zip() {
    local zip_name="$1"
    local merged_zip="$DATASET_DIR/$zip_name.zip"

    if ! compgen -G "$DATASET_DIR/$zip_name.zip.*" > /dev/null; then
        log "No parts found for $zip_name in $DATASET_DIR"
        return 1
    fi

    log "Merging $zip_name parts into a single zip file..."
    cat "$DATASET_DIR/$zip_name".zip.* > "$merged_zip"
    log "Merged $zip_name.zip created at $DATASET_DIR"

    # Remove partition files
    rm "$DATASET_DIR/$zip_name".zip.*
    log "Removing $zip_name parts"

    # Extract the merged file
    log "Extracting $zip_name.zip..."
    unzip -o "$merged_zip" -d "$DATASET_DIR"
    log "Extracted $zip_name.zip at $DATASET_DIR"

    rm -f "$merged_zip"
    log "Removed merged archive $merged_zip"
}

# Merge and extract train.zip parts;
merge_and_extract_zip "train" &

# Merge and extract test.zip parts
merge_and_extract_zip "test" &

# Wait for all background processes to finish
wait

# End of script
log "Script execution completed."
