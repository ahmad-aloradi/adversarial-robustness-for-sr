#!/bin/bash
# Simplified CNCeleb dataset preparation script
# Assumes all audio files are in FLAC format

set -euo pipefail

# Default paths
ROOT_DIR="${1:-data/cnceleb}"
ARTIFACTS_DIR="${2:-data/cnceleb/metadata}"

echo "🔧 Preparing CNCeleb dataset"
echo "   Root directory: $ROOT_DIR"
echo "   Artifacts directory: $ARTIFACTS_DIR"

# Run the Python preparation script
PYTHONPATH=$(pwd) python src/datamodules/components/cnceleb/cnceleb_prep.py \
    --root_dir "$ROOT_DIR" \
    --artifacts_dir "$ARTIFACTS_DIR"

echo "✅ CNCeleb dataset preparation completed successfully"
