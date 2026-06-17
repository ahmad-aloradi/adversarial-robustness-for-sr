#!/bin/bash
# Launch the three AdaBreg sparsity-redistribution variants (progressive,
# wpanneal, trainable_scales) at 99% target sparsity on multi_sv, for two
# backbones.
# Six runs total. ALL SIX run concurrently -- three pinned to each GPU:
#
#   GPU 0 -> ECAPA-TDNN (wespeaker_ecapa_tdnn, batch size 64) x3 variants
#   GPU 1 -> ResNet34   (wespeaker_resnet34,   batch size 32) x3 variants
#
# Settings mirror the main Bregman runs in scripts/fabfile.py (no augmentation,
# enroll/test batch 8, lambda acceleration_factor pinned to 1) adapted for a
# local two-GPU box instead of SLURM.
#
# NOTE: three training jobs share a single GPU. Make sure the per-GPU memory
# budget holds for 3x concurrent ECAPA (GPU 0) and 3x concurrent ResNet34
# (GPU 1) at these batch sizes; drop a batch size or move a job to the other
# GPU if you hit OOM.
#
# Usage:
#   bash scripts/run_bregman_adabreg_variants.sh
#
# Each run's console output is tee'd to logs/train/runs/<name>/launch.log, and
# the run itself lands in logs/train/runs/<name>/ so the scripts/visualize_*.py
# globs pick it up.
set -uo pipefail

cd "$HOME/adversarial-robustness-for-sr"
export PYTHONPATH="$HOME/adversarial-robustness-for-sr"

MAX_EPOCHS=30
TARGET_SPARSITY=0.99
DATASET=multi_sv
LOG_ROOT=logs/train/runs
# Brief gap between launches so concurrent jobs don't race on the shared
# HuggingFace cache / CUDA init; training still overlaps.
STAGGER_SECONDS=5

# One row per run: "<gpu> <experiment> <sv_model> <batch_size>".
# progressive is epoch-based by default (_bregman_ramp_granularity: epoch).
JOBS=(
    "0 sv_bregman_adabreg_progressive      wespeaker_ecapa_tdnn 64"
    "0 sv_bregman_adabreg_wpanneal         wespeaker_ecapa_tdnn 64"
    "0 sv_bregman_adabreg_trainable_scales wespeaker_ecapa_tdnn 64"
    "1 sv_bregman_adabreg_progressive      wespeaker_resnet34   32"
    "1 sv_bregman_adabreg_wpanneal         wespeaker_resnet34   32"
    "1 sv_bregman_adabreg_trainable_scales wespeaker_resnet34   32"
)

pids=()
labels=()

for job in "${JOBS[@]}"; do
    read -r gpu experiment sv_model batch_size <<< "${job}"
    name="${experiment}-${sv_model}-${DATASET}-bs${batch_size}-ep${MAX_EPOCHS}-augFalse-sr99"
    run_dir="${LOG_ROOT}/${name}"
    mkdir -p "${run_dir}"
    echo "[GPU ${gpu}] START ${name}  (log: ${run_dir}/launch.log)"

    CUDA_VISIBLE_DEVICES="${gpu}" python src/train.py \
        experiment=sv/"${experiment}" \
        module/sv_model="${sv_model}" \
        datamodule="${DATASET}" \
        trainer.max_epochs="${MAX_EPOCHS}" \
        datamodule.loaders.train.batch_size="${batch_size}" \
        datamodule.loaders.valid.batch_size="${batch_size}" \
        datamodule.loaders.enrollment.batch_size=8 \
        datamodule.loaders.test.batch_size=8 \
        module.data_augmentation=null \
        callbacks.model_pruning.lambda_scheduler.acceleration_factor=1 \
        _bregman_target_sparsity="${TARGET_SPARSITY}" \
        name="${name}" \
        hydra.run.dir="${run_dir}" \
        > "${run_dir}/launch.log" 2>&1 &

    pids+=("$!")
    labels+=("${name}")
    sleep "${STAGGER_SECONDS}"
done

echo "All six runs launched; waiting for completion..."

status=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "DONE  ${labels[$i]}"
    else
        echo "FAIL  ${labels[$i]} (exit $?)"
        status=1
    fi
done

if [ "${status}" -eq 0 ]; then
    echo "All six runs finished successfully."
else
    echo "One or more runs failed -- see the FAIL lines above and each run's launch.log."
fi
exit "${status}"
