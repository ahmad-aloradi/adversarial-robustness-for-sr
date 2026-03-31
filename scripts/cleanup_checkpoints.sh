#!/bin/bash
# Keep the 2 best (by metric_valid) + last* checkpoints per run, delete the rest.
# Usage: bash cleanup_checkpoints.sh /path/to/runs/dir

set -euo pipefail

DIR="${1:?Usage: $0 <runs_dir>}"

to_delete=()
to_keep=()

for ckpt_dir in "$DIR"/*/checkpoints; do
    [[ -d "$ckpt_dir" ]] || continue
    echo "=== $(basename "$(dirname "$ckpt_dir")") ==="

    # Separate last* and epoch* files
    lasts=() epochs=()
    for f in "$ckpt_dir"/*.ckpt; do
        [[ -f "$f" ]] || continue
        base=$(basename "$f")
        if [[ "$base" == last*.ckpt ]]; then
            lasts+=("$f")
        else
            epochs+=("$f")
        fi
    done

    # Keep all last* checkpoints
    for f in "${lasts[@]+"${lasts[@]}"}"; do
        to_keep+=("$f")
        echo "  KEEP (last): $(basename "$f")"
    done

    # Sort epoch checkpoints by metric_valid descending, keep top 2
    if [[ ${#epochs[@]} -gt 0 ]]; then
        sorted=($(for f in "${epochs[@]}"; do
            metric=$(basename "$f" | grep -oP 'metric_valid\K[0-9.]+')
            echo "$metric $f"
        done | sort -t' ' -k1 -rn | awk '{print $2}'))

        for i in "${!sorted[@]}"; do
            if [[ $i -lt 2 ]]; then
                to_keep+=("${sorted[$i]}")
                echo "  KEEP (top$((i+1))): $(basename "${sorted[$i]}")"
            else
                to_delete+=("${sorted[$i]}")
                echo "  DEL:  $(basename "${sorted[$i]}")"
            fi
        done
    fi
    echo
done

if [[ ${#to_delete[@]} -eq 0 ]]; then
    echo "Nothing to delete."
    exit 0
fi

echo "--- Summary: keeping ${#to_keep[@]}, deleting ${#to_delete[@]} ---"
read -rp "Proceed? [Y]es / [n]o / [m]odify (exclude experiments): " ans
ans="${ans:-Y}"

if [[ "$ans" =~ ^[Mm]$ ]]; then
    # Build unique list of experiments that have deletions
    declare -A exp_map  # exp_name -> list of files (newline-separated)
    exp_names=()
    for f in "${to_delete[@]}"; do
        exp=$(basename "$(dirname "$(dirname "$f")")")
        if [[ -z "${exp_map[$exp]+x}" ]]; then
            exp_names+=("$exp")
            exp_map[$exp]="$f"
        else
            exp_map[$exp]+=$'\n'"$f"
        fi
    done

    echo ""
    echo "Experiments with pending deletions:"
    for i in "${!exp_names[@]}"; do
        exp="${exp_names[$i]}"
        count=$(echo "${exp_map[$exp]}" | wc -l)
        echo "  [$((i+1))] $exp  ($count ckpt(s) to delete)"
    done

    echo ""
    echo "Enter experiment numbers to EXCLUDE (comma/space-separated), or 'none':"
    read -rp "> " exclude_input

    if [[ "${exclude_input,,}" == "none" || -z "$exclude_input" ]]; then
        exclude_indices=()
    else
        IFS=', ' read -ra exclude_indices <<< "$exclude_input"
    fi

    # Build set of excluded experiment names
    declare -A excluded
    for idx in "${exclude_indices[@]}"; do
        pos=$((idx - 1))
        if [[ $pos -ge 0 && $pos -lt ${#exp_names[@]} ]]; then
            excluded[${exp_names[$pos]}]=1
        else
            echo "Warning: ignoring invalid index $idx"
        fi
    done

    # Show exclusions
    if [[ ${#excluded[@]} -gt 0 ]]; then
        echo ""
        echo "Excluded experiments (will NOT delete their checkpoints):"
        for exp in "${!excluded[@]}"; do
            echo "  - $exp"
            while IFS= read -r f; do
                echo "      $(basename "$f")"
            done <<< "${exp_map[$exp]}"
        done
    fi

    # Filter to_delete
    filtered=()
    for f in "${to_delete[@]}"; do
        exp=$(basename "$(dirname "$(dirname "$f")")")
        if [[ -z "${excluded[$exp]+x}" ]]; then
            filtered+=("$f")
        fi
    done

    if [[ ${#filtered[@]} -eq 0 ]]; then
        echo ""
        echo "Nothing left to delete after exclusions."
        exit 0
    fi

    echo ""
    echo "Will delete ${#filtered[@]} checkpoint(s) (excluded ${#excluded[@]} experiment(s))."
    read -rp "Confirm? [Y/n] " confirm
    if [[ "${confirm:-Y}" =~ ^[Yy]$ ]]; then
        rm -v "${filtered[@]}"
        echo "Done."
    else
        echo "Aborted."
    fi
elif [[ "$ans" =~ ^[Yy]$ ]]; then
    rm -v "${to_delete[@]}"
    echo "Done."
else
    echo "Aborted."
fi
