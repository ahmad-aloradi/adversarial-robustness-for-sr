#!/bin/bash
# Clean up concatenated utterances and mapping file for CNCeleb
# Run this before re-running concat_cnceleb.sh with new parameters

set -euo pipefail

# Default ROOT_DIR
default_path="data/cnceleb"
if [ -d "data/datasets/cnceleb" ]; then
    default_path="data/datasets/cnceleb"
fi

ROOT_DIR="${1:-$default_path}"
OUTPUT_DIR="${2:-$ROOT_DIR/concatenated}"
MAPPING_FILE="${3:-$ROOT_DIR/metadata/concat_mapping.map}"

echo "🧹 Cleaning up CNCeleb concatenation artifacts"
echo "   Output directory: $OUTPUT_DIR"
echo "   Mapping file: $MAPPING_FILE"
echo ""

# Remove mapping file
if [ -f "$MAPPING_FILE" ]; then
    rm -v "$MAPPING_FILE"
    echo "   ✓ Removed mapping file"
else
    echo "   ⚠ Mapping file not found: $MAPPING_FILE"
fi

# Remove concatenated files directory
if [ -d "$OUTPUT_DIR" ]; then
    # Count files before deletion (WAV files - concatenated files are saved as WAV)
    FILE_COUNT=$(find "$OUTPUT_DIR" -type f -name "*.wav" 2>/dev/null | wc -l)
    rm -rf "$OUTPUT_DIR"
    echo "   ✓ Removed $FILE_COUNT concatenated audio files from $OUTPUT_DIR"
else
    echo "   ⚠ Output directory not found: $OUTPUT_DIR"
fi

echo ""
echo "✅ Cleanup complete. Ready for new concatenation run."
