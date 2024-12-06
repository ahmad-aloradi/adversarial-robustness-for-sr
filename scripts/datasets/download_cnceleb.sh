#!/bin/bash

# Variables
DOWNLOAD_DIR="/home.local/aloradi/cnceleb_data"
LOG_FILE=".download.log"
EXTRACT_DIR="$DOWNLOAD_DIR/extracted"


declare -a URLS=(
    # CN-Celeb v2
    "https://openslr.elda.org/resources/82/cn-celeb_v2.tar.gz"
    "https://openslr.elda.org/resources/82/cn-celeb2_v2.tar.gzaa"
    "https://openslr.elda.org/resources/82/cn-celeb2_v2.tar.gzab"
    "https://openslr.elda.org/resources/82/cn-celeb2_v2.tar.gzac"
)

declare -a FILENAMES=(
    # CN-Celeb v2 (contains CN-Celeb_flac)
    "cn-celeb_v2.tar.gz"
    
    # CN-Celeb2 v2 parts (contains CN-Celeb2_flac)
    "cn-celeb2_v2.tar.gz.aa"
    "cn-celeb2_v2.tar.gz.ab"
    "cn-celeb2_v2.tar.gz.ac"
)

# Additional step needed after download to combine CN-Celeb2 parts
combine_cnceleb2() {
    echo "Combining CN-Celeb2 parts..."
    cat cn-celeb2_v2.tar.gz.* > cn-celeb2_v2.tar.gz
    if [ $? -eq 0 ]; then
        echo "✓ Successfully combined CN-Celeb2 parts"
        return 0
    else
        echo "❌ Failed to combine CN-Celeb2 parts"
        return 1
    fi
}


# Create directories
mkdir -p "$DOWNLOAD_DIR" "$EXTRACT_DIR"
cd "$DOWNLOAD_DIR"

# Initialize log
echo "=== Download started at $(date) ===" | tee -a "$LOG_FILE"

# Download function with progress
download_and_verify() {
    local url=$1
    local filename=$2
    local max_retries=3
    local retry_count=0

    # Skip if file exists and is not empty
    if [ -s "$filename" ]; then
        echo "✓ $filename already exists, skipping download" | tee -a "$LOG_FILE"
        return 0
    fi

    while [ $retry_count -lt $max_retries ]; do
        echo "⚡ Downloading $filename (Attempt $((retry_count + 1))/$max_retries)"
        wget -c --progress=bar:force "$url" -O "$filename" 2>&1 | tee -a "$LOG_FILE"
        
        if [ $? -eq 0 ] && [ -s "$filename" ]; then
            echo "✓ Successfully downloaded $filename" | tee -a "$LOG_FILE"
            return 0
        else
            echo "⚠ Failed attempt $((retry_count + 1)) for $filename" | tee -a "$LOG_FILE"
            ((retry_count++))
            sleep 5
        fi
    done
    
    echo "❌ Failed to download $filename after $max_retries attempts" | tee -a "$LOG_FILE"
    return 1
}

# Extract function
extract_file() {
    local filename=$1
    local marker_file="$EXTRACT_DIR/.${filename}_extracted"

    # Skip if already extracted
    if [ -f "$marker_file" ]; then
        echo "✓ $filename already extracted, skipping" | tee -a "$LOG_FILE"
        return 0
    fi

    echo "📦 Extracting $filename..." | tee -a "$LOG_FILE"
    tar -xzf "$filename" -C "$EXTRACT_DIR" 2>&1 | tee -a "$LOG_FILE"
    
    if [ $? -eq 0 ]; then
        touch "$marker_file"
        echo "✓ Successfully extracted $filename" | tee -a "$LOG_FILE"
        return 0
    else
        echo "❌ Failed to extract $filename" | tee -a "$LOG_FILE"
        return 1
    fi
}

# Main process
echo "🚀 Starting CNCeleb dataset download..." | tee -a "$LOG_FILE"

# Download files
for i in "${!URLS[@]}"; do
    if ! download_and_verify "${URLS[$i]}" "${FILENAMES[$i]}"; then
        exit 1
    fi
done

# Extract files
for filename in "${FILENAMES[@]}"; do
    if ! extract_file "$filename"; then
        exit 1
    fi
done

# Print summary
echo -e "\n📊 Download Summary:" | tee -a "$LOG_FILE"
echo "===================="
echo "📂 Download location: $(pwd)"
echo "📝 Log file: $LOG_FILE"
echo "📦 Downloaded files:"
ls -lh

echo "✨ Process completed successfully at $(date)" | tee -a "$LOG_FILE"
exit 0