#!/bin/bash
# Delete *.o* log files that contain a SLURM time-limit termination marker.
#
# SLURM writes "DUE TO TIME LIMIT ***" near the end of output files when a job
# is killed for exceeding its walltime.  These files are safe to remove because
# the corresponding job must be re-submitted anyway.
#
# Usage:
#   bash cleanup_timelimit_logs.sh [DIR]
#
# DIR defaults to the current working directory when omitted.

set -euo pipefail

SEARCH_DIR="${1:-.}"
MARKER="DUE TO TIME LIMIT \*\*\*"

mapfile -t candidates < <(find "$SEARCH_DIR" -maxdepth 1 -type f -name "*.o*" 2>/dev/null | sort)

if [[ ${#candidates[@]} -eq 0 ]]; then
    echo "No *.o* files found in: $SEARCH_DIR"
    exit 0
fi

to_delete=()
for f in "${candidates[@]}"; do
    if grep -qE "$MARKER" "$f" 2>/dev/null; then
        to_delete+=("$f")
    fi
done

if [[ ${#to_delete[@]} -eq 0 ]]; then
    echo "No time-limit log files found (checked ${#candidates[@]} file(s))."
    exit 0
fi

echo "Found ${#to_delete[@]} time-limit log file(s) to delete:"
for f in "${to_delete[@]}"; do
    echo "  $f"
done

read -rp "Delete all ${#to_delete[@]} file(s)? [Y/n] " ans
if [[ ! "${ans:-Y}" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

for f in "${to_delete[@]}"; do
    rm -f "$f"
    echo "Deleted: $f"
done

echo "Done. Removed ${#to_delete[@]} file(s)."
