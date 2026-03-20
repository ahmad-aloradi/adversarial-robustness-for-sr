#!/bin/bash
# Available sources: train_log --> epoch-mean, csv --> batch-step

export PYTHONPATH="$HOME/adversarial-robustness-for-sr"

stage_visualize=false
stage_aggregate=false
stage_test_viz=true
stage_weight_norms=true

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
        --base_dir /dataHDD/ahmad/comfort26_sem/cnceleb \
        --experiments "sv_vanilla_*" "sv_wespeaker_*" "sv_bregman_*-sr90" "sv_bregman_*-sr95" \
        --source "csv" \
        --output results/cross_exp_comparison/convergence_curves \
        --metrics "${metrics[@]}"
fi

############################
# Exps metrics aggregation
############################
if [ "$stage_aggregate" = true ]; then
    base_dirs=(
        "/dataHDD/ahmad/comfort26_sem/cnceleb" \
        "/dataHDD/ahmad/comfort26_sem/multi_sv"
        )

    python scripts/aggregate_json_scores.py \
         --base_dirs "${base_dirs[@]}" \
         --output_dir results/cross_exp_comparison/test_metrics
fi

############################
# Test Artifacts Visualization
############################
if [ "$stage_test_viz" = true ]; then
    python scripts/visualize_test_artifacts.py \
        --base_dir /dataHDD/ahmad/comfort26_sem/cnceleb \
        --experiments "sv_vanilla_*" "sv_wespeaker_*" "sv_bregman_*-sr90" "sv_bregman_*-sr95" \
        --plots all \
        --embed_method umap \
        --score_col both \
        --output results/test_artifacts_vis/
fi

############################
# Weight Norms Visualization
############################
if [ "$stage_weight_norms" = true ]; then
    python scripts/visualize_weight_norms.py \
        --base_dir logs/train/runs \
        --experiments "2026-03-14_14-19-55" \
        --output results/weight_norms_vis/
fi
