#!/bin/bash
# Fix a broken last.ckpt (or best.ckpt) symlink by pointing it to an available checkpoint.
# Usage: bash fix_ckpt_symlink.sh <checkpoints_dir_or_run_dir> [--auto]
#
# --auto: non-interactive; picks the checkpoint with highest metric_valid

set -euo pipefail

TARGET="${1:?Usage: $0 <checkpoints_dir_or_run_dir> [--auto]}"
AUTO=false
[[ "${2:-}" == "--auto" ]] && AUTO=true

# Resolve to the checkpoints dir
if [[ -d "$TARGET/checkpoints" ]]; then
    CKPT_DIR="$TARGET/checkpoints"
else
    CKPT_DIR="$TARGET"
fi

[[ -d "$CKPT_DIR" ]] || { echo "ERROR: directory not found: $CKPT_DIR"; exit 1; }

echo "=== Checkpoint directory: $CKPT_DIR ==="

# Show current state of last.ckpt
LAST="$CKPT_DIR/last.ckpt"
if [[ -L "$LAST" ]]; then
    resolved=$(readlink "$LAST")
    if [[ -f "$LAST" ]]; then
        echo "last.ckpt -> $resolved  [OK]"
    else
        echo "last.ckpt -> $resolved  [BROKEN]"
    fi
elif [[ -f "$LAST" ]]; then
    echo "last.ckpt is a regular file (not a symlink)"
else
    echo "last.ckpt does not exist"
fi

# Collect real (non-symlink) checkpoint files
mapfile -t real_ckpts < <(find "$CKPT_DIR" -maxdepth 1 -name '*.ckpt' ! -type l | sort)

if [[ ${#real_ckpts[@]} -eq 0 ]]; then
    echo "ERROR: no real checkpoint files found in $CKPT_DIR"
    exit 1
fi

echo ""
echo "Available checkpoints:"
for i in "${!real_ckpts[@]}"; do
    echo "  [$((i+1))] $(basename "${real_ckpts[$i]}")"
done

# Pick best by metric_valid (highest), break ties by epoch number (highest), fallback to last sorted
best_file=""
best_metric=-1
best_epoch=-1
for f in "${real_ckpts[@]}"; do
    base=$(basename "$f")
    metric=$(echo "$base" | grep -oP 'metric_valid\K[0-9.]+' || true)
    epoch=$(echo "$base"  | grep -oP '^epoch\K[0-9]+'         || true)
    [[ -z "$metric" ]] && metric="-1"
    [[ -z "$epoch"  ]] && epoch="-1"
    if awk "BEGIN{exit !(($metric > $best_metric) || ($metric == $best_metric && $epoch > $best_epoch))}"; then
        best_metric=$metric
        best_epoch=$epoch
        best_file="$f"
    fi
done
[[ -z "$best_file" ]] && best_file="${real_ckpts[-1]}"

if $AUTO; then
    chosen="$best_file"
    echo ""
    echo "Auto-selected: $(basename "$chosen")"
else
    echo ""
    echo "Suggested (highest metric_valid): $(basename "$best_file")"
    read -rp "Enter number to select [default: suggestion]: " pick
    if [[ -z "$pick" ]]; then
        chosen="$best_file"
    else
        idx=$((pick - 1))
        if [[ $idx -lt 0 || $idx -ge ${#real_ckpts[@]} ]]; then
            echo "ERROR: invalid selection"
            exit 1
        fi
        chosen="${real_ckpts[$idx]}"
    fi
fi

echo ""
echo "Will set last.ckpt -> $(basename "$chosen")"
if ! $AUTO; then
    read -rp "Confirm? [Y/n] " confirm
    [[ "${confirm:-Y}" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }
fi

# Backup old symlink if it exists
if [[ -L "$LAST" || -f "$LAST" ]]; then
    mv -v "$LAST" "${LAST}.bak"
fi

ln -s "$(basename "$chosen")" "$LAST"
echo "Done: last.ckpt -> $(basename "$chosen")"
