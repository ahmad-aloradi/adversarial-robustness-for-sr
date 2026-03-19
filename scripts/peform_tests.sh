#!/bin/bash
# Available sources: train_log --> epoch-mean, csv --> batch-step

export PYTHONPATH="~/adversarial-robustness-for-sr"

stage_visualize=true
stage_aggregate=true

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
        --output results/figures/ \
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
         --output_dir results/test_metrics
fi
