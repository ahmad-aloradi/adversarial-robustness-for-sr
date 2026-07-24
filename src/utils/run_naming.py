"""Run naming: the output-directory stem and the logger run name.

``configs/hydra/default.yaml`` calls the ``run_subdir`` resolver so every
compression method lands under one ``<dataset>/<model>/<augmentation>`` parent
— comparing methods is then a plain ``ls``. ``configs/logger/wandb.yaml`` calls
``run_name`` to label the dashboard entry with that same run directory.
"""

import os


def _scheduler_tag(target):
    """Short tag for an LR-scheduler ``_target_``; ``no_scheduler`` when unset.

    Strips the module path and a trailing ``LR`` (``CosineAnnealingLR`` ->
    ``CosineAnnealing``), leaving names without that suffix untouched.

    >>> _scheduler_tag("torch.optim.lr_scheduler.CosineAnnealingLR")
    'CosineAnnealing'
    >>> _scheduler_tag("torch.optim.lr_scheduler.ReduceLROnPlateau")
    'ReduceLROnPlateau'
    >>> _scheduler_tag(None)
    'no_scheduler'
    """
    if not target or target in ("none", "None"):
        return "no_scheduler"
    name = target.rsplit(".", 1)[-1]
    return name[:-2] if name.endswith("LR") and len(name) > 2 else name


def run_subdir(dataset, model, augmentation, experiment, sparsity, scheduler):
    """Build ``<dataset>/<model>/<aug>/<method>[-srNN]-<scheduler>`` for the run dir.

    Each argument is a config interpolation; ``None`` (a field absent on a
    non-image task, or no target sparsity on a dense baseline) becomes a
    neutral tag or is dropped so the path never fails to resolve.

    >>> run_subdir("cifar10", "resnet18", True,
    ...            "img/pruning_mag_struct", 0.9,
    ...            "torch.optim.lr_scheduler.CosineAnnealingLR")
    'cifar10/resnet18/augmentation/pruning_mag_struct-sr90-CosineAnnealing'
    >>> run_subdir("mnist", "resnet18", False, "img/dense_sgd", None, None)
    'mnist/resnet18/no_augmentation/dense_sgd-no_scheduler'
    """

    aug = "augmentation" if augmentation else "no_augmentation"
    method = (experiment or "no_experiment").rsplit("/", 1)[-1].split(".")[0]
    sr = f"-sr{int(round(sparsity * 100))}" if sparsity is not None else ""
    return f"{dataset}/{model}/{aug}/{method}{sr}-{_scheduler_tag(scheduler)}"


def run_name(output_dir, log_dir):
    """Name the logger run after its run directory, minus the machine-specific root.

    ``output_dir`` is the resolved ``hydra.run.dir``; only its ``log_dir`` prefix
    and the constant ``<task>/runs`` head differ between a cluster and a laptop,
    so dropping both gives one dashboard name per experiment everywhere.

    >>> run_name("/vault/results/train/runs/cifar10/resnet18/no_augmentation/dense_sgd-no_scheduler/seed_42", "/vault/results")
    'cifar10/resnet18/no_augmentation/dense_sgd-no_scheduler/seed_42'
    >>> run_name("/home/logs/eval/runs/cnceleb/sv_vanilla-bs256", "/home/logs/eval/runs/cnceleb")
    'sv_vanilla-bs256'
    """
    relative = os.path.relpath(output_dir, log_dir)
    assert not relative.startswith(
        ".."
    ), f"run dir {output_dir!r} must live under log dir {log_dir!r}"
    parts = relative.split(os.sep)
    if len(parts) > 2 and parts[1] in ("runs", "multiruns"):
        parts = parts[2:]
    return "/".join(parts)


if __name__ == "__main__":
    print(
        run_subdir(
            "cifar10", "resnet18", False, "img/bregman_adabreg.yaml", 0.9, None
        )
    )
    print(
        run_subdir(
            "cifar100",
            "wrn28_10",
            True,
            "img/dense_sgd",
            None,
            "torch.optim.lr_scheduler.ReduceLROnPlateau",
        )
    )
    print(
        run_subdir(
            "cifar100",
            "wrn28_10",
            True,
            "img/pruning_mag_struct",
            0.8,
            "torch.optim.lr_scheduler.CosineAnnealingLR",
        )
    )
    print(run_name("/results/train/runs/cifar10/wrn28_10/dense_sgd/seed_42", "/results"))
