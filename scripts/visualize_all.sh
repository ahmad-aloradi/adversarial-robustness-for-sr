#!/bin/bash

export PYTHONPATH="$HOME/adversarial-robustness-for-sr"
source=csv # # Sources: train_log --> epoch-mean, csv --> batch-step

stage_visualize=true
stage_aggregate_and_vis=true
stage_test_viz=false
stage_weight_norms=false
force_recompute=true

# Experiment selection criteria
eval_model='resnet34' # 'resnet34' 'ecapa_tdnn'
eval_data='multi_sv' # 'cnceleb', 'multi_sv'
sparsity_rate='sr[7-9][0-9]' #  'sr75' 'sr90', 'sr95', 'sr99'
experiment='sv_bregman_*breg-wespeaker' # e.g. 'sv_bregman', 'sv_pruning', 'sv_vanilla', 'sv_wespeaker'
experiment2='sv_pruning_mag_unstruct' # e.g. 'sv_bregman', 'sv_pruning', 'sv_vanilla', 'sv_wespeaker'

if [ "$eval_model" = 'resnet34' ]; then
    suffix=*'regl1_conv'    # '*regl1_conv'  regl1_conv-alpha0.5-f50
    echo "Using suffix: $suffix for experiment filtering"
else
    suffix=''
    echo "No suffix for experiment filtering"
fi 

base_dirs=(
    '/data/aloradad/results'
)
experiments=(
    # sv_bregman_linbreg_fixed-wespeaker_ecapa_tdnn-cnceleb-virt-False-bs256-vadFalse-ep40-augFalse-sr90
    "${experiment}*${eval_model}*${eval_data}*${sparsity_rate}*${suffix}"
    "${experiment2}*${eval_model}*${eval_data}*${sparsity_rate}*"
    "sv_vanilla-*${eval_model}*${eval_data}*"
    "sv_wespeaker-*${eval_model}*${eval_data}*"
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
        "lr" "train/margin"
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
        --input_dir results/cross_exp_comparison/test_metrics \
        --base_dirs "${exp_dirs[@]}" \
        --experiments ${experiments[@]}

    python scripts/visualize_sparsity_trend.py \
        --input_dir results/cross_exp_comparison/test_metrics \
        --base_dirs "${exp_dirs[@]}" \
        --experiments ${experiments[@]}
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
