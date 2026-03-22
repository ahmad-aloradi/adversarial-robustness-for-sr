#!/bin/bash
# Available sources: train_log --> epoch-mean, csv --> batch-step

export PYTHONPATH="$HOME/adversarial-robustness-for-sr"

stage_visualize=true
stage_aggregate_and_vis=false
stage_test_viz=false
stage_weight_norms=false

eva_model='ecapa_tdnn'
force_recompute=true

base_dirs=(
    '/dataHDD/ahmad/21_03_2026'
    '/dataHDD/ahmad/comfort26_sem'
)
experiments=(
    "sv_bregman_*${eva_model}*"
    "sv_vanilla_*${eva_model}*"
    "sv_wespeaker*${eva_model}*"
    )

# Build list of dataset subdirs across all base_dirs
exp_dirs=()
for bd in "${base_dirs[@]}"; do
    exp_dirs+=("$bd/cnceleb" "$bd/multi_sv")
done

############################
# Exps Visualization
############################
if [ "$stage_visualize" = true ]; then
    metrics=(
        "EER" "minDCF" \
        "train_loss" "train/MulticlassAccuracy" "valid/MulticlassAccuracy" \
        "bregman/global_lambda" "bregman/sparsity" #"sparsity"
        )

    python scripts/visualize.py \
        --base_dirs "${exp_dirs[@]}" \
        --experiments "sv_bregman_*" \
        --source "csv" \
        --output results/cross_exp_comparison/convergence_curves \
        --metrics "${metrics[@]}"
fi

########################################################
# Exps metrics aggregation + Test Metrics Bar Charts
########################################################
if [ "$stage_aggregate_and_vis" = true ]; then
    python scripts/aggregate_json_scores.py \
         --base_dirs "${exp_dirs[@]}" \
         --output_dir results/cross_exp_comparison/test_metrics

    python scripts/visualize_test_metrics.py \
        --input_dir results/cross_exp_comparison/test_metrics
fi

############################
# Test Artifacts Visualization
############################
if [ "$stage_test_viz" = true ]; then
    force_flag=""
    if [ "$force_recompute" = true ]; then
        force_flag="--force_recompute"
    fi

    python scripts/visualize_test_artifacts.py \
        --base_dirs "${exp_dirs[@]}" \
        --experiments "sv_bregman_*" \
        --plots all \
        --embed_method tsne \
        --score_col both \
        --output results/test_artifacts_visualizations/ \
        $force_flag
fi

############################
# Weight Norms Visualization
############################
if [ "$stage_weight_norms" = true ]; then
    python scripts/visualize_weight_norms.py \
        --base_dirs "${exp_dirs[@]}" \
        --experiments "sv_bregman_*" \
        --output results/weight_norms_visualizations/
fi
