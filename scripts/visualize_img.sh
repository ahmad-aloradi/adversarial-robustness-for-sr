#!/bin/bash
# Image-classification visualization: accuracy curves + cross-seed accuracy
# summary (mean±std) + sparsity curves, per dataset/backbone/augmentation/target.
#
# Run layout: <base_root>/<dataset>/<model>/<augmentation>/<exp>/seed_<N>
# where <exp> = <method>[-isr<NN>][-sr<NN>|-lam<V>]-<scheduler>. Pinning the
# augmentation dir and the scheduler tag keeps each figure to one comparable set;
# visualize.py reads dataset/model/augmentation from the path, so patterns only
# match <exp>.

export PYTHONPATH="$HOME/adversarial-robustness-for-sr"

base_root='/data/aloradad/results'  # /data/aloradad/results logs/train/runs
source=csv # train_log --> epoch-mean (always present); csv --> batch-step
summary_reduce=max # per-seed accuracy = best epoch (matches selected ckpt)

datasets=('cifar100' 'tinyimagenet') #('cifar10' 'tinyimagenet')
models=('resnet50' 'wrn28_10') # resnet18 | wrn28_10
augmentation='augmentation' # 'augmentation' | 'no_augmentation'
scheduler='*' # one LR-scheduler tag: -CosineAnnealing | -ReduceLROnPlateau | -no_scheduler
sparsity_rates=('-sr90' '-sr95' '-sr99') # target tokens; the leading dash keeps -isr99 out
fixed_lambda='-lam*' # fixed-lambda runs carry their lambda, not a target
suffix="*" # *, classifier_10k

struct_unstruct=false
support_turnover=true # exploration rate; Bregman runs only, they alone log it

head_anatomy_data_dir=data # local dataset root for the pooled-feature forward pass
head_anatomy=false # explain the head stripe; inflated-head runs only

rule='=============================================================='

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do

        # The curated results tree has no augmentation level; train runs nest one.
        if [ "$base_root" = '/data/aloradad/results' ]; then
            base_dir="${base_root}/${dataset}/${model}"
            str="${rule}\nImage results: ${dataset} / ${model}\n${rule}"

        elif [ "$base_root" = 'logs/train/runs' ]; then
            base_dir="${base_root}/${dataset}/${model}/${augmentation}"
            str="${rule}\nImage results: ${dataset} / ${model} / ${augmentation}\n${rule}"

        else
            echo "[error] unknown base_root: ${base_root}" >&2
            exit 1
        fi

        if [ ! -d "$base_dir" ]; then
            echo -e "[skip] no runs at ${base_dir}\n"
            continue
        fi

        echo -e "$str"
        out_dir="results/img/${dataset}/${model}/${augmentation}"

        # One target per pass. Two targets in one figure would put a 90% run and
        # a 99% run on one axis, so each gets its own directory.
        for sparsity_rate in "${sparsity_rates[@]}"; do
            rate_tag="${sparsity_rate#-}" # the same token in a path
            echo -e "\n--- target ${rate_tag} ---"

            # Dense baseline + pruning + Bregman, all on one scheduler and target.
            # The fixed-lambda runs need their own patterns: they are named by
            # the lambda they hold, so no target token selects them.
            experiments=(
                "dense_sgd${scheduler}"
                # "pruning_mag_unstruct*${model}-${dataset}*${sparsity_rate}*${scheduler}*${suffix}"
                # "pruning_mag_struct*${model}-${dataset}*${sparsity_rate}*${scheduler}*${suffix}"
                "pruning_rigl*${model}-${dataset}*${sparsity_rate}*${scheduler}*${suffix}"
                "pruning_set*${model}-${dataset}*${sparsity_rate}*${scheduler}*${suffix}"
                "pruning_static*${model}-${dataset}*${sparsity_rate}*${scheduler}*${suffix}"
                "pruning_snip*${model}-${dataset}*${sparsity_rate}*${scheduler}*${suffix}"
                "pruning_granet*${model}-${dataset}*${sparsity_rate}*${scheduler}*${suffix}"
                "bregman_adabreg*${model}-${dataset}*${sparsity_rate}*${scheduler}*${suffix}"
                "bregman_linbreg*${model}-${dataset}*${sparsity_rate}*${scheduler}*${suffix}"
                "bregman_adabreg_fixed*${model}-${dataset}*${fixed_lambda}"
                "bregman_linbreg_fixed*${model}-${dataset}*${fixed_lambda}"
                # STR's sparsity is an outcome of its weight decay, so no target token selects it.
                "soft_threshold*${model}-${dataset}*${suffix}"
            )

            python scripts/visualize.py \
                --base_dirs "$base_dir" \
                --experiments "${experiments[@]}" \
                --metrics 'test/MulticlassAccuracy' 'valid/MulticlassAccuracy' 'train/MulticlassAccuracy' "pruning/sparsity" "bregman/pruned_sparsity" "bregman/global_lambda" \
                --source "$source" \
                --summary-reduce "$summary_reduce" \
                --output "${out_dir}/numerical_${rate_tag}"

            # Mask-structure diagnostics read from the checkpoints, not the metric logs.
            # Dense is the 0%-sparsity reference and lists the same layers as the rest.
            mask_experiments=(
                "pruning_mag_unstruct*${sparsity_rate}*${scheduler}*${suffix}"
                "pruning_mag_struct*${sparsity_rate}*${scheduler}*${suffix}"
                "pruning_rigl*${sparsity_rate}*${scheduler}*${suffix}"
                "pruning_set*${sparsity_rate}*${scheduler}*${suffix}"
                "pruning_static*${sparsity_rate}*${scheduler}*${suffix}"
                "pruning_snip*${sparsity_rate}*${scheduler}*${suffix}"
                "pruning_granet*${sparsity_rate}*${scheduler}*${suffix}"
                "bregman_adabreg*${sparsity_rate}*${scheduler}*${suffix}"
                "bregman_linbreg*${sparsity_rate}*${scheduler}*${suffix}"
                "dense_sgd*${scheduler}*${suffix}*"
            )
            if [ "$struct_unstruct" = true ]; then
                python scripts/visualize_structured_vs_unstructured.py \
                    --base_dirs "$base_dir" \
                    --experiments "${mask_experiments[@]}" \
                    --output "${out_dir}/struct_vs_unstruct_${rate_tag}"
            fi

            # Turnover comes from bregman/support_births and bregman/support_deaths,
            # so only the Bregman runs carry it; every other method is skipped.
            if [ "$support_turnover" = true ]; then
                turnover_experiments=(
                    "bregman_adabreg*${sparsity_rate}*${scheduler}*${suffix}"
                    "bregman_linbreg*${sparsity_rate}*${scheduler}*${suffix}"
                    "bregman_adabreg_fixed*${fixed_lambda}"
                    "bregman_linbreg_fixed*${fixed_lambda}"
                )

                python scripts/visualize_support_turnover.py \
                    --base_dirs "$base_dir" \
                    --experiments "${turnover_experiments[@]}" \
                    --output "${out_dir}/support_turnover_${rate_tag}"
            fi
        done

        # Head anatomy explains the stripe those heatmaps show, so it needs the
        # inflated head — magnitude is kept as the no-stripe control. Its pattern
        # carries no target token, so it runs once per model, not once per target.
        if [ "$head_anatomy" = true ]; then
            head_experiments=(
                "*classifier_10k*"
            )

            python scripts/visualize_structured_vs_unstructured.py \
                --base_dirs "$base_dir" \
                --experiments "${head_experiments[@]}" \
                --head_anatomy --activations --data_dir "$head_anatomy_data_dir" \
                --output "${out_dir}/head_anatomy"
        fi
    done
done
