#!/bin/bash

export PYTHONPATH="$HOME/adversarial-robustness-for-sr"
source=csv # # Sources: train_log --> epoch-mean, csv --> batch-step

stage_visualize=true
stage_aggregate_and_vis=true
stage_test_viz=false
stage_weight_norms=false
stage_struct_vs_unstruct=false
force_recompute=true

# Experiment selection criteria
experiment_fixed='sv_bregman_*breg_fixed-wespeaker'

for eval_data in 'cnceleb' 'multi_sv'; do
    for eval_model in 'resnet34' 'ecapa_tdnn'; do
# for eval_data in 'cnceleb'; do
    # for eval_model in 'ecapa_tdnn'; do

        sparsity_rate_test='sr[0-9][0-9]' #  'sr75' 'sr90', 'sr95', 'sr99'

        if [ "${eval_data}" = 'cnceleb' ] && [ "${eval_model}" = 'ecapa_tdnn' ]; then
            sparsity_rate='sr[7-9][0-9]' #  'sr90', 'sr95', 'sr99'
        else
            sparsity_rate='sr[7-9][0-9]' #  'sr75' 'sr90', 'sr95', 'sr99'
            echo Using sparsity pattern ${sparsity_rate}
        fi

        if [ "$eval_model" = 'resnet34' ]; then
            suffix_adabreg='regl1_conv-alpha0.25-f50'
            suffix_linbreg='regl1_conv'
        else
            suffix_adabreg=''
            suffix_linbreg=''
        fi

        echo "=============================================================================================="
        echo "Showing results for ${eval_data} and ${eval_model} (adabreg suffix: '${suffix_adabreg}', linbreg suffix: '${suffix_linbreg}')"
        echo "=============================================================================================="

        base_dirs=(
            '/data/aloradad/results'
        )

        # Convergence curves
        experiments_vis=(
            "sv_bregman_*adabreg-wespeaker*${eval_model}*${eval_data}*${sparsity_rate}*${suffix_adabreg}"
            "sv_bregman_*linbreg-wespeaker*${eval_model}*${eval_data}*${sparsity_rate}*${suffix_linbreg}"
            # "${experiment_fixed}*${eval_model}*${eval_data}*${sparsity_rate_test}*${suffix_linbreg}"
            "sv_vanilla-*${eval_model}*${eval_data}*"
            "sv_wespeaker-*${eval_model}*${eval_data}*"
            )
        # For sparsity trends visualization
        experiments_test=(
            "sv_bregman_adabreg-wespeaker*${eval_model}*${eval_data}*${sparsity_rate_test}*${suffix_adabreg}"
            "sv_bregman_linbreg-wespeaker*${eval_model}*${eval_data}*${sparsity_rate_test}*${suffix_linbreg}"
            "sv_pruning_mag_struct*${eval_model}*${eval_data}*${sparsity_rate_test}*"
            "sv_pruning_mag_unstruct*${eval_model}*${eval_data}*${sparsity_rate_test}*"
            "sv_vanilla-*${eval_model}*${eval_data}*"
            "sv_wespeaker-*${eval_model}*${eval_data}*"
            )
        # For sparsity trends visualization
        experiments_sp_trends=(
            "sv_bregman_adabreg-wespeaker*${eval_model}*${eval_data}*${sparsity_rate_test}*${suffix_adabreg}"
            "sv_bregman_linbreg-wespeaker*${eval_model}*${eval_data}*${sparsity_rate_test}*${suffix_linbreg}"
            "sv_pruning_mag_unstruct*${eval_model}*${eval_data}*${sparsity_rate_test}*"
            "sv_vanilla-*${eval_model}*${eval_data}*"
            "sv_wespeaker-*${eval_model}*${eval_data}*"
            )
        # For struct/unstruct comparison
        experiments_sparsity=(
            "sv_bregman_adabreg-wespeaker*${eval_model}*${eval_data}*${sparsity_rate_test}*${suffix_adabreg}"
            "sv_bregman_linbreg-wespeaker*${eval_model}*${eval_data}*${sparsity_rate_test}*${suffix_linbreg}"
            "sv_pruning_mag_unstruct*${eval_model}*${eval_data}*${sparsity_rate_test}*" 
            "sv_pruning_mag_struct*${eval_model}*${eval_data}*${sparsity_rate_test}*"         
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
                "train_loss" "train/MulticlassAccuracy" "valid/MulticlassAccuracy" \
                "bregman/global_lambda" "bregman/sparsity" #"sparsity"
                "lr" "train/margin"
                )

            python scripts/visualize.py \
                --base_dirs "${exp_dirs[@]}" \
                --experiments ${experiments_vis[@]} \
                --source ${source} \
                --output results/cross_exp_comparison/convergence_curves/${eval_data}/${eval_model} \
                --metrics "${metrics[@]}"
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
                --output_dir results/test_eval/visualizations/${eval_model}
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

        ############################
        # Structured vs Unstructured Sparsity Visualization
        ############################
        if [ "$stage_struct_vs_unstruct" = true ]; then
            python scripts/visualize_structured_vs_unstructured.py \
                --base_dirs "${exp_dirs[@]}" \
                --experiments ${experiments_sparsity[@]} \
                --output results/struct_vs_unstruct/${eval_data}/${eval_model} \
                # --rtf
        fi
    done
done