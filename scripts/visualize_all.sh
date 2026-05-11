#!/bin/bash

export PYTHONPATH="$HOME/adversarial-robustness-for-sr"
source=csv # # Sources: train_log --> epoch-mean, csv --> batch-step

stage_visualize=false
stage_aggregate_and_vis=false
stage_test_viz=false
stage_weight_norms=false
stage_struct_vs_unstruct=true
stage_cross_model_weight_norms=false
force_recompute=true

# Cross-model weight norms: single sparsity rate compared across architectures
cross_model_sparsity='sr90'

# Experiment selection criteria
experiment_fixed='sv_bregman_*breg_fixed-wespeaker'

# for eval_data in 'cnceleb' 'multi_sv'; do
#     for eval_model in 'resnet34' 'ecapa_tdnn'; do
for eval_data in 'multi_sv'; do
    for eval_model in resnet34; do

        sparsity_rate_test='sr[7-9][0-9]' #  'sr75' 'sr90', 'sr95', 'sr99'
        sparsity_rate='sr[7-9][0-9]' #  'sr75' 'sr90', 'sr95', 'sr99'

        if [ "$eval_model" = 'resnet34' ]; then
            suffix_adabreg='regl1_conv-alpha0.25-f50'
            suffix_linbreg='regl1_conv'
            suffix_fixed='regl1_conv'
        else
            suffix_adabreg=''
            suffix_linbreg=''
            suffix_fixed=''
        fi

        echo "=============================================================================================="
        echo "Showing results for ${eval_data} and ${eval_model} (adabreg suffix: '${suffix_adabreg}', linbreg suffix: '${suffix_linbreg}', fixed suffix: '${suffix_fixed}')"
        echo "=============================================================================================="

        base_dirs=(
            '/data/aloradad/results'
        )

        # Convergence curves
        experiments_vis=(
            "sv_bregman_*adabreg-wespeaker*${eval_model}*${eval_data}*${sparsity_rate}*${suffix_adabreg}"
            "sv_bregman_*linbreg-wespeaker*${eval_model}*${eval_data}*${sparsity_rate}*${suffix_linbreg}"
            "${experiment_fixed}*${eval_model}*${eval_data}*${sparsity_rate_test}*${suffix_fixed}"
            "sv_vanilla-*${eval_model}*${eval_data}*"
            "sv_wespeaker-*${eval_model}*${eval_data}*"
            )
        # For sparsity trends visualization
        experiments_test=(
            "sv_bregman_adabreg-wespeaker*${eval_model}*${eval_data}*${sparsity_rate_test}*${suffix_adabreg}"
            "sv_bregman_linbreg-wespeaker*${eval_model}*${eval_data}*${sparsity_rate_test}*${suffix_linbreg}"
            "${experiment_fixed}*${eval_model}*${eval_data}*${sparsity_rate_test}*${suffix_fixed}"
            # "sv_pruning_mag_struct*${eval_model}*${eval_data}*${sparsity_rate_test}*"
            "sv_pruning_mag_unstruct*${eval_model}*${eval_data}*${sparsity_rate_test}*"
            "sv_vanilla-*${eval_model}*${eval_data}*"
            "sv_wespeaker-*${eval_model}*${eval_data}*"
            )
        # For sparsity trends visualization
        experiments_sp_trends=(
            "sv_bregman_adabreg-wespeaker*${eval_model}*${eval_data}*${sparsity_rate_test}*${suffix_adabreg}"
            "sv_bregman_linbreg-wespeaker*${eval_model}*${eval_data}*${sparsity_rate_test}*${suffix_linbreg}"
            # "sv_pruning_mag_struct*${eval_model}*${eval_data}*${sparsity_rate_test}*"
            "sv_pruning_mag_unstruct*${eval_model}*${eval_data}*${sparsity_rate_test}*"
            "${experiment_fixed}*${eval_model}*${eval_data}*${sparsity_rate_test}*${suffix_fixed}"
            "sv_vanilla-*${eval_model}*${eval_data}*"
            "sv_wespeaker-*${eval_model}*${eval_data}*"
            )
        # Weight norms curves
        experiments_weight_norms=(
            "sv_bregman_*adabreg-wespeaker*${eval_model}*${eval_data}*sr90*${suffix_adabreg}"
            "sv_bregman_*linbreg-wespeaker*${eval_model}*${eval_data}*sr90*${suffix_linbreg}"
            # "sv_bregman_*adabreg-wespeaker*${eval_model}*${eval_data}*sr75*${suffix_adabreg}"
            # "sv_bregman_*linbreg-wespeaker*${eval_model}*${eval_data}*sr75*${suffix_linbreg}"
            "${experiment_fixed}*${eval_model}*${eval_data}*sr90*${suffix_fixed}"
            "sv_vanilla-*${eval_model}*${eval_data}*"
            "sv_wespeaker-*${eval_model}*${eval_data}*"
            )
        # For struct/unstruct comparison — Bregman at sr75/sr90/sr99; unstruct pinned to sr90 only
        experiments_struct_vs_unstruct_linbreg=(
            "sv_bregman_linbreg-wespeaker*${eval_model}*${eval_data}*sr75*${suffix_linbreg}"
            "sv_bregman_linbreg-wespeaker*${eval_model}*${eval_data}*sr90*${suffix_linbreg}"
            "sv_bregman_linbreg-wespeaker*${eval_model}*${eval_data}*sr99*${suffix_linbreg}"
            "sv_bregman_linbreg_fixed-wespeaker*${eval_model}*${eval_data}*sr90*${suffix_fixed}"
            "sv_pruning_mag_unstruct*${eval_model}*${eval_data}*sr90*"
            )
        experiments_struct_vs_unstruct_adabreg=(
            "sv_bregman_adabreg-wespeaker*${eval_model}*${eval_data}*sr75*${suffix_adabreg}"
            "sv_bregman_adabreg-wespeaker*${eval_model}*${eval_data}*sr90*${suffix_adabreg}"
            "sv_bregman_adabreg-wespeaker*${eval_model}*${eval_data}*sr99*${suffix_adabreg}"
            "sv_bregman_adabreg_fixed*${eval_model}*${eval_data}*sr75*${suffix_fixed}"
            "sv_bregman_adabreg_fixed*${eval_model}*${eval_data}*sr90*${suffix_fixed}"
            "sv_bregman_adabreg_fixed*${eval_model}*${eval_data}*sr99*${suffix_fixed}"
            "sv_pruning_mag_unstruct*${eval_model}*${eval_data}*sr90*"
            )
        experiments_struct_vs_unstruct_mixed=(
            "sv_bregman_linbreg-wespeaker*${eval_model}*${eval_data}*sr99*${suffix_linbreg}"
            "sv_bregman_adabreg-wespeaker*${eval_model}*${eval_data}*sr99*${suffix_adabreg}"
            "sv_pruning_mag_unstruct*${eval_model}*${eval_data}*sr99*"
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
                "EER" \
                # "train_loss" 
                "train/MulticlassAccuracy" "valid/MulticlassAccuracy" \
                "bregman/global_lambda" "bregman/sparsity" #"sparsity"
                # "lr" "train/margin"
                )

            python scripts/visualize.py \
                --base_dirs "${exp_dirs[@]}" \
                --experiments ${experiments_vis[@]} \
                --source ${source} \
                --output results/cross_exp_comparison/convergence_curves/${eval_model}/${eval_data} \
                --metrics "${metrics[@]}" \
                --legend-mode split
        fi

        ########################################################
        # Exps metrics aggregation + Test Metrics Bar Charts
        ########################################################
        if [ "$stage_aggregate_and_vis" = true ]; then
            python scripts/aggregate_json_scores.py \
                --base_dirs "${exp_dirs[@]}" \
                --output_dir results/test_eval/metrics/${eval_model}

            python scripts/visualize_test_metrics.py \
                --input_dir results/test_eval/metrics/${eval_model} \
                --base_dirs "${exp_dirs[@]}" \
                --experiments ${experiments_test[@]} \
                --output_dir results/test_eval/visualizations/${eval_model}

            python scripts/visualize_sparsity_trend.py \
                --input_dir results/test_eval/metrics/${eval_model} \
                --base_dirs "${exp_dirs[@]}" \
                --experiments ${experiments_sp_trends[@]} \
                --output_dir results/test_eval/visualizations/${eval_model} \
                --legend-mode split
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
                --experiments ${experiments_weight_norms[@]} \
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
                --experiments ${experiments_weight_norms[@]} \
                --output results/weight_norms_visualizations/${eval_model}/${eval_data}
        fi

        ############################
        # Structured vs Unstructured Sparsity Visualization
        ############################
        if [ "$stage_struct_vs_unstruct" = true ]; then
            # python scripts/visualize_structured_vs_unstructured.py \
            #     --base_dirs "${exp_dirs[@]}" \
            #     --experiments ${experiments_struct_vs_unstruct_adabreg[@]} \
            #     --output results/struct_vs_unstruct/${eval_data}/${eval_model}/adabreg \
            #     --legend-mode split
            #     # --rtf

            python scripts/visualize_structured_vs_unstructured.py \
                --base_dirs "${exp_dirs[@]}" \
                --experiments ${experiments_struct_vs_unstruct_mixed[@]} \
                --output results/struct_vs_unstruct/${eval_data}/${eval_model}/mixed \
                --legend-mode split

        fi
    done

    ##################################################################
    # Cross-Model Weight Norms (single sparsity rate across backbones)
    ##################################################################
    # Compares ResNet34 vs ECAPA-TDNN at a single target sparsity (set via
    # $cross_model_sparsity at the top of this script). Runs once per dataset.
    # Reuses $base_dirs / $exp_dirs from the inner loop — both are independent
    # of $eval_model, so the last-iteration values are correct.
    if [ "$stage_cross_model_weight_norms" = true ]; then
        echo "=============================================================================================="
        echo "Cross-model weight norms: ${eval_data} @ ${cross_model_sparsity}"
        echo "=============================================================================================="

        # experiments_cross_model=(
        #     # "sv_bregman_*adabreg-wespeaker*resnet34*${eval_data}*${cross_model_sparsity}*regl1_conv-alpha0.25-f50"
        #     # "sv_bregman_*linbreg-wespeaker*resnet34*${eval_data}*${cross_model_sparsity}*regl1_conv"
        #     # "${experiment_fixed}*resnet34*${eval_data}*${cross_model_sparsity}*regl1_conv"
        #     "sv_bregman_*adabreg-wespeaker*ecapa_tdnn*${eval_data}*${cross_model_sparsity}*"
        #     "sv_bregman_*linbreg-wespeaker*ecapa_tdnn*${eval_data}*${cross_model_sparsity}*"
        #     "${experiment_fixed}*ecapa_tdnn*${eval_data}*${cross_model_sparsity}*"
        #     # "sv_vanilla-*resnet34*${eval_data}*"
        #     # "sv_wespeaker-*resnet34*${eval_data}*"
        #     "sv_vanilla-*ecapa_tdnn*${eval_data}*"
        #     "sv_wespeaker-*ecapa_tdnn*${eval_data}*"
        #     )
        experiments_cross_model=(
            "sv_bregman_*adabreg-wespeaker*resnet34*${eval_data}*${cross_model_sparsity}*regl1_conv-alpha0.25-f50"
            "sv_bregman_*linbreg-wespeaker*resnet34*${eval_data}*${cross_model_sparsity}*regl1_conv"
            "${experiment_fixed}*resnet34*${eval_data}*${cross_model_sparsity}*regl1_conv"
            # "sv_bregman_*adabreg-wespeaker*ecapa_tdnn*${eval_data}*${cross_model_sparsity}*"
            # "sv_bregman_*linbreg-wespeaker*ecapa_tdnn*${eval_data}*${cross_model_sparsity}*"
            # "${experiment_fixed}*ecapa_tdnn*${eval_data}*${cross_model_sparsity}*"
            "sv_vanilla-*resnet34*${eval_data}*"
            "sv_wespeaker-*resnet34*${eval_data}*"
            # "sv_vanilla-*ecapa_tdnn*${eval_data}*"
            # "sv_wespeaker-*ecapa_tdnn*${eval_data}*"
            )

        python scripts/visualize_weight_norms.py \
            --base_dirs "${exp_dirs[@]}" \
            --experiments ${experiments_cross_model[@]} \
            --output results/weight_norms_visualizations/cross_model/${eval_data}/${cross_model_sparsity} \
            --legend-mode split
    fi
done