#!/bin/bash

export PYTHONPATH="$HOME/adversarial-robustness-for-sr"
source=csv # # Sources: train_log --> epoch-mean, csv --> batch-step

stage_visualize=true
stage_aggregate_and_vis=true
stage_test_viz=false
stage_weight_norms=true
force_recompute=true

# Experiment selection criteria
eval_model='ecapa_tdnn' # 'resnet34' 'ecapa_tdnn'
eval_data='multi_sv'
sparsity_rate='sr90' # 'sr90', 'sr95', 'sr99'
experiment='sv_bregman*' # e.g. 'sv_bregman', 'sv_pruning', 'sv_vanilla', 'sv_wespeaker'

base_dirs=(
    '/data/ahmad/results'
)
experiments=(
    "${experiment}*${eval_model}*${eval_data}*-subgrad_corr_v2"
    "*fixed*${eval_model}*${eval_data}*"
    "${experiment}*${eval_model}*${eval_data}*${sparsity_rate}"
    "sv_vanilla_*${eval_model}*"
    )

# Build list of dataset subdirs across all base_dirs
exp_dirs=()
for bd in "${base_dirs[@]}"; do
    exp_dirs+=(
        "$bd/cnceleb"
        "$bd/multi_sv"
        )
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
        --experiments ${experiments[@]} \
        --source ${source} \
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

    python scripts/visualize_sparsity_trend.py \
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
        --experiments ${experiments[@]} \
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
        --experiments ${experiments[@]} \
        --output results/weight_norms_visualizations/
fi
