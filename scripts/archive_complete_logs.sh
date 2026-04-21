#!/bin/bash
# Move *.o* log files that contain the job-completion marker to an archive folder.
#
# "Starting saving state dicts!" is written when a training job reaches the
# checkpoint-save phase, indicating the job ran to a meaningful completion point.
#
# Usage:
#   bash archive_complete_logs.sh [DIR]
#
# DIR defaults to the current working directory when omitted.
# Completed files are moved to o_files/complete/<timestamp> relative to DIR.

set -euo pipefail

SEARCH_DIR="${1:-.}"
MARKER="Starting saving state dicts!"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
DEST_DIR="$SEARCH_DIR/o_files/complete/$TIMESTAMP"

mapfile -t candidates < <(find "$SEARCH_DIR" -maxdepth 1 -type f -name "*.o*" 2>/dev/null | sort)

if [[ ${#candidates[@]} -eq 0 ]]; then
    echo "No *.o* files found in: $SEARCH_DIR"
    exit 0
fi

to_move=()
for f in "${candidates[@]}"; do
    if grep -qF "$MARKER" "$f" 2>/dev/null; then
        to_move+=("$f")
    fi
done

if [[ ${#to_move[@]} -eq 0 ]]; then
    echo "No completed log files found (checked ${#candidates[@]} file(s))."
    exit 0
fi

echo "Found ${#to_move[@]} completed log file(s) to archive:"
for f in "${to_move[@]}"; do
    echo "  $f"
done

read -rp "Move all ${#to_move[@]} file(s) to $DEST_DIR? [Y/n] " ans
if [[ ! "${ans:-Y}" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

mkdir -p "$DEST_DIR"

for f in "${to_move[@]}"; do
    mv "$f" "$DEST_DIR/"
    echo "Archived: $f"
done

echo "Done. Archived ${#to_move[@]} file(s) to $DEST_DIR"
