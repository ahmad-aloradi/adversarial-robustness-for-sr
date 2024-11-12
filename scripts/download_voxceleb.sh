#!/bin/bash

# VoxCeleb Download and Processing Script with Authentication
# Usage: scripts/download_voxceleb.sh <target_directory> <username> <password>

set -e  # Exit on error

# Check arguments
if [ $# -ne 3 ]; then
    echo "Usage: $0 <target_directory> <username> <password>"
    echo "Example: $0 /data/voxceleb your_username your_password"
    exit 1
fi

# Configuration
TARGET_DIR="$1"
USERNAME="$2"
PASSWORD="$3"
TEMP_DIR="${TARGET_DIR}/temp"
METADATA_DIR="${TARGET_DIR}/metadata"
VOX1_DIR="${TARGET_DIR}/voxceleb1"
VOX2_DIR="${TARGET_DIR}/voxceleb2"
LOG_FILE="${TARGET_DIR}/download_log.txt"

# URLs
VOX1_DEV_URL="https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partaa"
VOX1_TEST_URL="https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_test_wav.zip"
VOX2_DEV_URL="https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox2_dev_aac_partaa"

# Create required directories
mkdir -p "$TARGET_DIR" "$TEMP_DIR" "$METADATA_DIR" "$VOX1_DIR" "$VOX2_DIR"

# Initialize log
exec 1> >(tee -a "$LOG_FILE")
exec 2>&1

echo "Starting VoxCeleb download and processing at $(date)"
echo "Target directory: $TARGET_DIR"

# Function to download with authentication and retry
download_with_retry() {
    local url="$1"
    local output="$2"
    local max_retries=3
    local retry=0
    local wget_opts=(
        --continue
        --progress=bar:force:noscroll
        --no-check-certificate
        --user "$USERNAME"
        --password "$PASSWORD"
    )
    
    while [ $retry -lt $max_retries ]; do
        echo "Downloading $url (attempt $((retry + 1))/$max_retries)"
        if wget "${wget_opts[@]}" "$url" -O "$output"; then
            return 0
        fi
        retry=$((retry + 1))
        if [ $retry -lt $max_retries ]; then
            echo "Download failed, retrying in 5 seconds..."
            sleep 5
        fi
    done
    
    echo "Failed to download $url after $max_retries retries"
    return 1
}

# Download VoxCeleb1
download_voxceleb1() {
    echo "Downloading VoxCeleb1..."
    
    # Download dev set
    echo "Downloading VoxCeleb1 dev set..."
    download_with_retry "$VOX1_DEV_URL" "${TEMP_DIR}/vox1_dev.tar"
    
    if [ -f "${TEMP_DIR}/vox1_dev.tar" ]; then
        echo "Extracting VoxCeleb1 dev set..."
        tar -xf "${TEMP_DIR}/vox1_dev.tar" -C "$VOX1_DIR"
    else
        echo "Error: VoxCeleb1 dev set download failed"
        exit 1
    fi
    
    # Download test set
    echo "Downloading VoxCeleb1 test set..."
    download_with_retry "$VOX1_TEST_URL" "${TEMP_DIR}/vox1_test.zip"
    
    if [ -f "${TEMP_DIR}/vox1_test.zip" ]; then
        echo "Extracting VoxCeleb1 test set..."
        unzip -q "${TEMP_DIR}/vox1_test.zip" -d "$VOX1_DIR"
    else
        echo "Error: VoxCeleb1 test set download failed"
        exit 1
    fi
}

# Download VoxCeleb2
download_voxceleb2() {
    echo "Downloading VoxCeleb2..."
    
    # Download dev set
    echo "Downloading VoxCeleb2 dev set..."
    download_with_retry "$VOX2_DEV_URL" "${TEMP_DIR}/vox2_dev.tar"
    
    if [ -f "${TEMP_DIR}/vox2_dev.tar" ]; then
        echo "Extracting VoxCeleb2 dev set..."
        tar -xf "${TEMP_DIR}/vox2_dev.tar" -C "$VOX2_DIR"
    else
        echo "Error: VoxCeleb2 dev set download failed"
        exit 1
    fi
}

# Main execution
echo "Testing authentication..."
# Test auth with a small request first
if ! wget --spider --user "$USERNAME" --password "$PASSWORD" --no-check-certificate "$VOX1_TEST_URL" 2>/dev/null; then
    echo "Authentication failed. Please check your credentials."
    exit 1
fi

echo "Authentication successful. Starting downloads..."

# Download datasets
download_voxceleb1
download_voxceleb2

# Cleanup
echo "Cleaning up temporary files..."
rm -rf "$TEMP_DIR"

echo "Process completed at $(date)"
echo "VoxCeleb1 location: $VOX1_DIR"
echo "VoxCeleb2 location: $VOX2_DIR"
echo "Metadata location: $METADATA_DIR"

# Final check
if [ -d "$VOX1_DIR" ] && [ -d "$VOX2_DIR" ]; then
    echo "Successfully downloaded and processed VoxCeleb datasets"
else
    echo "Error: Something went wrong during processing"
    exit 1
fi