#!/bin/bash
# Delete the oldest timestamped test artifact run per test set (dataset) in
# each experiment, while keeping newer runs.
# Usage: bash cleanup_test_artifacts.sh /path/to/runs/dir

set -euo pipefail

DIR="${1:?Usage: $0 <runs_dir>}"
TIMESTAMP_RE='^[0-9]{8}_[0-9]{6}$'

to_delete=()
to_keep=()

declare -A exp_map                 # exp_rel -> list of directories to delete (newline-separated)
declare -A last_run_file_for_del   # delete_dir -> LAST_RUN file path
declare -A last_run_new_for_del    # delete_dir -> replacement timestamp
exp_names=()

found_test_artifacts=0

while IFS= read -r test_root; do
    found_test_artifacts=1
    exp_dir=$(dirname "$test_root")
    exp_rel="${exp_dir#$DIR/}"
    [[ "$exp_rel" == "$exp_dir" ]] && exp_rel=$(basename "$exp_dir")

    echo "=== $exp_rel ==="
    exp_has_deletions=0

    for test_set_dir in "$test_root"/*; do
        [[ -d "$test_set_dir" ]] || continue
        test_set=$(basename "$test_set_dir")

        # Skip internal/cache folders such as _cohort_cache
        if [[ "$test_set" == _* || "$test_set" == .* ]]; then
            continue
        fi

        mapfile -t sorted_runs < <(
            for p in "$test_set_dir"/*; do
                [[ -d "$p" ]] || continue
                ts=$(basename "$p")
                if [[ "$ts" =~ $TIMESTAMP_RE ]]; then
                    echo "$p"
                fi
            done | sort
        )

        run_count=${#sorted_runs[@]}
        if [[ $run_count -eq 0 ]]; then
            echo "  $test_set: no timestamped runs"
            continue
        fi

        if [[ $run_count -eq 1 ]]; then
            to_keep+=("${sorted_runs[0]}")
            echo "  KEEP (only run) $test_set: $(basename "${sorted_runs[0]}")"
            continue
        fi

        oldest="${sorted_runs[0]}"
        to_delete+=("$oldest")
        exp_has_deletions=1

        if [[ -z "${exp_map[$exp_rel]+x}" ]]; then
            exp_names+=("$exp_rel")
            exp_map[$exp_rel]="$oldest"
        else
            exp_map[$exp_rel]+=$'\n'"$oldest"
        fi

        echo "  $test_set ($run_count runs):"
        for i in "${!sorted_runs[@]}"; do
            ts=$(basename "${sorted_runs[$i]}")
            if [[ $i -eq 0 ]]; then
                echo "    DEL (oldest): $ts"
            else
                to_keep+=("${sorted_runs[$i]}")
                echo "    KEEP:         $ts"
            fi
        done

        # If LAST_RUN points to the removed run, rewrite it to newest remaining.
        last_run_file="$test_set_dir/LAST_RUN"
        if [[ -f "$last_run_file" ]]; then
            current_last=$(<"$last_run_file")
            if [[ "$current_last" == "$(basename "$oldest")" ]]; then
                newest_index=$((run_count - 1))
                newest_ts=$(basename "${sorted_runs[$newest_index]}")
                last_run_file_for_del[$oldest]="$last_run_file"
                last_run_new_for_del[$oldest]="$newest_ts"
            fi
        fi
    done

    if [[ $exp_has_deletions -eq 0 ]]; then
        echo "  No deletions queued in this experiment."
    fi
    echo
done < <(find "$DIR" -type d -name test_artifacts | sort)

if [[ $found_test_artifacts -eq 0 ]]; then
    echo "No test_artifacts directories found under: $DIR"
    exit 0
fi

if [[ ${#to_delete[@]} -eq 0 ]]; then
    echo "Nothing to delete."
    exit 0
fi

echo "--- Summary: keeping ${#to_keep[@]}, deleting ${#to_delete[@]} oldest run(s) ---"
read -rp "Proceed? [Y]es / [n]o / [m]odify (exclude experiments): " ans
ans="${ans:-Y}"

filtered=("${to_delete[@]}")
declare -A excluded

if [[ "$ans" =~ ^[Mm]$ ]]; then
    echo ""
    echo "Experiments with pending deletions:"
    for i in "${!exp_names[@]}"; do
        exp="${exp_names[$i]}"
        count=$(printf '%s\n' "${exp_map[$exp]}" | sed '/^$/d' | wc -l)
        count=$(echo "$count" | tr -d ' ')
        echo "  [$((i + 1))] $exp  ($count oldest run(s) to delete)"
    done

    echo ""
    echo "Enter experiment numbers to EXCLUDE (comma/space-separated), or 'none':"
    read -rp "> " exclude_input

    if [[ "${exclude_input,,}" != "none" && -n "$exclude_input" ]]; then
        IFS=', ' read -ra exclude_indices <<< "$exclude_input"
        for idx in "${exclude_indices[@]}"; do
            pos=$((idx - 1))
            if [[ $pos -ge 0 && $pos -lt ${#exp_names[@]} ]]; then
                excluded[${exp_names[$pos]}]=1
            else
                echo "Warning: ignoring invalid index $idx"
            fi
        done
    fi

    filtered=()
    for d in "${to_delete[@]}"; do
        exp_dir=$(dirname "$(dirname "$(dirname "$d")")")
        exp_rel="${exp_dir#$DIR/}"
        [[ "$exp_rel" == "$exp_dir" ]] && exp_rel=$(basename "$exp_dir")
        if [[ -z "${excluded[$exp_rel]+x}" ]]; then
            filtered+=("$d")
        fi
    done

    if [[ ${#excluded[@]} -gt 0 ]]; then
        echo ""
        echo "Excluded experiments (will NOT delete their oldest runs):"
        for exp in "${!excluded[@]}"; do
            echo "  - $exp"
            while IFS= read -r d; do
                [[ -n "$d" ]] || continue
                test_set=$(basename "$(dirname "$d")")
                ts=$(basename "$d")
                echo "      $test_set/$ts"
            done <<< "${exp_map[$exp]}"
        done
    fi

    if [[ ${#filtered[@]} -eq 0 ]]; then
        echo ""
        echo "Nothing left to delete after exclusions."
        exit 0
    fi

    echo ""
    echo "Will delete ${#filtered[@]} oldest run(s) (excluded ${#excluded[@]} experiment(s))."
    read -rp "Confirm? [Y/n] " confirm
    if [[ ! "${confirm:-Y}" =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
elif [[ ! "$ans" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

for d in "${filtered[@]}"; do
    rm -rf "$d"
    echo "Deleted: $d"
done

for d in "${filtered[@]}"; do
    if [[ -n "${last_run_file_for_del[$d]+x}" ]]; then
        lr_file="${last_run_file_for_del[$d]}"
        lr_new="${last_run_new_for_del[$d]}"
        if [[ -f "$lr_file" ]]; then
            printf '%s\n' "$lr_new" > "$lr_file"
            echo "Updated LAST_RUN: $(dirname "$lr_file") -> $lr_new"
        fi
    fi
done

echo "Done."