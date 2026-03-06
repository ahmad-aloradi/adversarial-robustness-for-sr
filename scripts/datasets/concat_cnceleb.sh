#!/bin/bash
# Concatenate short utterances for CNCeleb dataset
# Can be run standalone or automatically via: CONCAT=1 ./scripts/datasets/prep_cnceleb.sh
#
# Concatenated files are saved as WAV (not FLAC) to avoid seek table issues.

set -euo pipefail

# Default ROOT_DIR
default_path="data/cnceleb"
if [ -d "data/datasets/cnceleb" ]; then
    default_path="data/datasets/cnceleb"
fi

ROOT_DIR="${1:-$default_path}"
OUTPUT_DIR="${2:-$ROOT_DIR/concatenated}"
MAPPING_FILE="${3:-$ROOT_DIR/metadata/concat_mapping.map}"

# Dataset subdirectory names
CNCELEB1="${4:-CN-Celeb_flac}"
CNCELEB2="${5:-CN-Celeb2_flac}"  # Set to empty string to skip

# Concatenation parameters
TARGET_DURATION="${6:-6.0}"
MIN_THRESHOLD="${7:-2.0}"

echo "🔗 Concatenating short utterances for CNCeleb"
echo "   Root directory: $ROOT_DIR"
echo "   Output directory: $OUTPUT_DIR"
echo "   Mapping file: $MAPPING_FILE"
echo "   Target duration: ${TARGET_DURATION}s"
echo "   Min threshold: ${MIN_THRESHOLD}s"
echo ""

# Run the concatenation script directly (no eval)
PYTHONPATH=$(pwd) python scripts/datasets/concat_short_utterances.py \
    --root_dir "$ROOT_DIR" \
    --cnceleb1 "$CNCELEB1" \
    --target_duration "$TARGET_DURATION" \
    --output_dir "$OUTPUT_DIR" \
    --mapping_file "$MAPPING_FILE" \
    --min_threshold "$MIN_THRESHOLD" \
    --cnceleb2 "$CNCELEB2"

echo ""
echo "✅ Concatenation completed successfully"
echo ""