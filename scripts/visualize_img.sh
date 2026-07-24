#!/bin/bash
# Image-classification visualization: accuracy curves + cross-seed accuracy
# summary (mean±std) + sparsity curves, per dataset/backbone/augmentation.
#
# Run layout: <base_root>/<dataset>/<model>/<augmentation>/<exp>/seed_<N>
# where <exp> = <method>[-sr<NN>]-<scheduler>. Pinning the augmentation dir and
# the scheduler tag keeps each figure to one comparable set; visualize.py reads
# dataset/model/augmentation from the path, so patterns only match <exp>.

export PYTHONPATH="$HOME/adversarial-robustness-for-sr"

base_root='/data/aloradad/results'  # /data/aloradad/results logs/train/runs
source=csv # train_log --> epoch-mean (always present); csv --> batch-step
summary_reduce=max # per-seed accuracy = best epoch (matches selected ckpt)

datasets=('tinyimagenet') #('cifar10' 'tinyimagenet')
models=('resnet18') # ('resnet18' 'wrn28_10') resnet18 | wrn28_10
augmentation='augmentation' # 'augmentation' | 'no_augmentation'
scheduler='*' # one LR-scheduler tag: -CosineAnnealing | -ReduceLROnPlateau | -no_scheduler
sparsity_rate='sr[8-9][8-9]' # sr75 sr90 sr95 sr99
suffix="10k-" # *, classifier_10k
head_anatomy_data_dir=data # local dataset root for the pooled-feature forward pass

struct_unstruct=false
head_anatomy=true # explain the head stripe; inflated-head runs only


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
        # Dense baseline + pruning + Bregman, all on one scheduler and sparsity
        # sweep. Exact method tokens exclude the fixed/progressive/wanda variants.
        experiments=(
            "dense_sgd${scheduler}"
            "pruning_mag_unstruct*${sparsity_rate}*${scheduler}*${suffix}"
            "pruning_mag_struct*${sparsity_rate}*${scheduler}*${suffix}"
            "bregman_adabreg*${sparsity_rate}*${scheduler}*${suffix}"
            "bregman_linbreg*${sparsity_rate}*${scheduler}*${suffix}"
        )

        python scripts/visualize.py \
            --base_dirs "$base_dir" \
            --experiments "${experiments[@]}" \
            --metrics 'test/MulticlassAccuracy' 'valid/MulticlassAccuracy' 'train/MulticlassAccuracy' "sparsity" "bregman/global_lambda" "bregman/sparsity" \
            --source "$source" \
            --summary-reduce "$summary_reduce" \
            --output "results/img/${dataset}/${model}/${augmentation}"/numerical

        # Mask-structure diagnostics read from the checkpoints, not the metric logs.
        # Dense is the 0%-sparsity reference and lists the same layers as the rest.
        mask_experiments=(
            "pruning_mag_unstruct*${sparsity_rate}*${scheduler}*${suffix}"
            "pruning_mag_struct*${sparsity_rate}*${scheduler}*${suffix}"
            "bregman_adabreg*${sparsity_rate}*${scheduler}*${suffix}"
            "bregman_linbreg*${sparsity_rate}*${scheduler}*${suffix}"
            "dense_sgd*${scheduler}*${suffix}*"
        )
         if [ "$struct_unstruct" = true ]; then
            python scripts/visualize_structured_vs_unstructured.py \
                --base_dirs "$base_dir" \
                --experiments "${mask_experiments[@]}" \
                --output "results/img/${dataset}/${model}/${augmentation}/struct_vs_unstruct"
        fi

        # Head anatomy explains the stripe those heatmaps show, so it needs the
        # inflated head — magnitude is kept as the no-stripe control.
        if [ "$head_anatomy" = true ]; then
            head_experiments=(
                "*classifier_10k*"
            )

            python scripts/visualize_structured_vs_unstructured.py \
                --base_dirs "$base_dir" \
                --experiments "${head_experiments[@]}" \
                --head_anatomy --activations --data_dir "$head_anatomy_data_dir" \
                --output "results/img/${dataset}/${model}/${augmentation}/head_anatomy"
        fi
    done
done
