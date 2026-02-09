#!/usr/bin/env bash

set -euo pipefail

# Default values
TARGETS="0.5 0.7 0.9"
EPOCHS=25
SEED=42
DRY_RUN=false

# Usage function
usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Run Bregman pruning experiments systematically across multiple configurations.

OPTIONS:
    --targets TARGETS      Space-separated sparsity targets (default: "0.5 0.7 0.9")
    --epochs EPOCHS        Maximum training epochs (default: 25)
    --seed SEED            Random seed (default: 42)
    --dry-run              Print commands without executing them
    -h, --help             Show this help message

EXAMPLES:
    # Run default configuration (3 inverse-scale + 1 scheduled + 1 EMA)
    $0

    # Run only at 0.9 sparsity
    $0 --targets "0.9"

    # Preview commands without execution
    $0 --dry-run

    # Custom epochs and seed
    $0 --epochs 30 --seed 123

EOF
    exit 0
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --targets)
            TARGETS="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Enable full Hydra errors
export HYDRA_FULL_ERROR=1

# Initialize tracking arrays
declare -a RUN_NAMES
declare -a RUN_COMMANDS
declare -a RUN_RESULTS
declare -a RUN_DIRS

# Counter for run numbering
RUN_ID=0

echo "======================================"
echo "Bregman Experiment Runner"
echo "======================================"
echo "Targets: $TARGETS"
echo "Epochs: $EPOCHS"
echo "Seed: $SEED"
echo "Dry run: $DRY_RUN"
echo "======================================"
echo ""

# Wave 1: Inverse-scale experiments (fixed target, initial sparsity 0.99)
echo "Wave 1: Inverse-scale experiments"
echo "-----------------------------------"
for target in $TARGETS; do
    RUN_ID=$((RUN_ID + 1))
    RUN_NAME="inverse_scale_${target}"
    RUN_CMD="python src/train.py experiment=sv/sv_pruning_bregman callbacks.model_pruning.lambda_scheduler.target_sparsity=${target} trainer.max_epochs=${EPOCHS} seed=${SEED} tags=[bregman_verify,inverse_scale,sparsity_${target}]"

    RUN_NAMES+=("$RUN_NAME")
    RUN_COMMANDS+=("$RUN_CMD")

    echo "[$RUN_ID] $RUN_NAME"
    echo "    $RUN_CMD"
    echo ""

    if [ "$DRY_RUN" = false ]; then
        if eval "$RUN_CMD"; then
            RUN_RESULTS+=("SUCCESS")
            # Capture output directory from Hydra (last line typically contains path)
            RUN_DIRS+=("check_logs")
        else
            EXIT_CODE=$?
            RUN_RESULTS+=("FAILED (exit code: $EXIT_CODE)")
            RUN_DIRS+=("N/A")
        fi
    fi
done

# Wave 2: Scheduled experiment (ramps 0.0 → 0.9)
echo "Wave 2: Scheduled experiment"
echo "-----------------------------------"
RUN_ID=$((RUN_ID + 1))
RUN_NAME="scheduled_0.9"
RUN_CMD="python src/train.py experiment=sv/sv_pruning_bregman_scheduled trainer.max_epochs=${EPOCHS} seed=${SEED} tags=[bregman_verify,scheduled,sparsity_0.9]"

RUN_NAMES+=("$RUN_NAME")
RUN_COMMANDS+=("$RUN_CMD")

echo "[$RUN_ID] $RUN_NAME"
echo "    $RUN_CMD"
echo ""

if [ "$DRY_RUN" = false ]; then
    if eval "$RUN_CMD"; then
        RUN_RESULTS+=("SUCCESS")
        RUN_DIRS+=("check_logs")
    else
        EXIT_CODE=$?
        RUN_RESULTS+=("FAILED (exit code: $EXIT_CODE)")
        RUN_DIRS+=("N/A")
    fi
fi

# Wave 3: EMA experiment (fixed target 0.9 with EMA smoothing)
echo "Wave 3: EMA experiment"
echo "-----------------------------------"
RUN_ID=$((RUN_ID + 1))
RUN_NAME="ema_0.9"
RUN_CMD="python src/train.py experiment=sv/sv_pruning_bregman_ema trainer.max_epochs=${EPOCHS} seed=${SEED} tags=[bregman_verify,ema,sparsity_0.9]"

RUN_NAMES+=("$RUN_NAME")
RUN_COMMANDS+=("$RUN_CMD")

echo "[$RUN_ID] $RUN_NAME"
echo "    $RUN_CMD"
echo ""

if [ "$DRY_RUN" = false ]; then
    if eval "$RUN_CMD"; then
        RUN_RESULTS+=("SUCCESS")
        RUN_DIRS+=("check_logs")
    else
        EXIT_CODE=$?
        RUN_RESULTS+=("FAILED (exit code: $EXIT_CODE)")
        RUN_DIRS+=("N/A")
    fi
fi

# Print summary
echo ""
echo "======================================"
echo "Experiment Summary"
echo "======================================"

if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN MODE - No experiments were executed"
    echo "Total commands generated: ${#RUN_NAMES[@]}"
else
    echo "Run | Name                      | Status"
    echo "----|---------------------------|------------------"
    for i in "${!RUN_NAMES[@]}"; do
        printf "%3d | %-25s | %s\n" "$((i + 1))" "${RUN_NAMES[$i]}" "${RUN_RESULTS[$i]}"
    done

    # Count successes and failures
    SUCCESS_COUNT=$(printf '%s\n' "${RUN_RESULTS[@]}" | grep -c "SUCCESS" || true)
    FAIL_COUNT=$(printf '%s\n' "${RUN_RESULTS[@]}" | grep -c "FAILED" || true)

    echo ""
    echo "Total: ${#RUN_NAMES[@]} | Success: $SUCCESS_COUNT | Failed: $FAIL_COUNT"

    if [ "$FAIL_COUNT" -gt 0 ]; then
        echo ""
        echo "Some experiments failed. Check logs for details."
        exit 1
    fi
fi

echo "======================================"
