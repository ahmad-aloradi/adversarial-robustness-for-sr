#!/bin/bash
# Concatenate short utterances for CNCeleb dataset
# This script should be run BEFORE prep_cnceleb.sh

set -euo pipefail

# Default paths
ROOT_DIR="${1:-data/cnceleb}"
OUTPUT_DIR="${2:-data/cnceleb/concatenated}"
MAPPING_FILE="${3:-data/cnceleb/metadata/concat_mapping.map}"

# Dataset subdirectory names
CNCELEB1="${CNCELEB1:-CN-Celeb_flac}"
CNCELEB2="${CNCELEB2:-CN-Celeb2_flac}"  # Set to empty string to skip

# Concatenation parameters
TARGET_DURATION="${TARGET_DURATION:-6.0}"
MIN_THRESHOLD="${MIN_THRESHOLD:-2.0}"

echo "🔗 Concatenating short utterances for CNCeleb"
echo "   Root directory: $ROOT_DIR"
echo "   Output directory: $OUTPUT_DIR"
echo "   Mapping file: $MAPPING_FILE"
echo "   Target duration: ${TARGET_DURATION}s"
echo "   Min threshold: ${MIN_THRESHOLD}s"
echo ""

# Build command
CMD="PYTHONPATH=$(pwd) python scripts/datasets/concat_short_utterances.py \
    --root_dir \"$ROOT_DIR\" \
    --cnceleb1 \"$CNCELEB1\" \
    --target_duration $TARGET_DURATION \
    --output_dir \"$OUTPUT_DIR\" \
    --mapping_file \"$MAPPING_FILE\" \
    --min_threshold $MIN_THRESHOLD"

# Add cnceleb2 if specified
if [ -n "${CNCELEB2}" ]; then
    CMD="$CMD --cnceleb2 \"$CNCELEB2\""
fi

# Run the concatenation script
eval "$CMD"

echo ""
echo "✅ Concatenation completed successfully"
echo ""
echo "Next steps:"
echo "   1. Run prep_cnceleb.sh with the concat_mapping parameter"
echo "   2. Example: CONCAT_MAPPING=$MAPPING_FILE ./scripts/datasets/prep_cnceleb.sh"
