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


def sparsity_token(sparsity):
    """``-sr<NN>`` for the sparsity a run drives toward, empty when absent.

    >>> sparsity_token(0.9), sparsity_token(0.29), sparsity_token(None)
    ('-sr90', '-sr29', '')
    """
    return f"-sr{int(round(sparsity * 100))}" if sparsity is not None else ""


def initial_sparsity_token(initial_sparsity):
    """``-isr<NN>`` for the sparsity a Bregman run starts at.

    >>> initial_sparsity_token(0.99), initial_sparsity_token(None)
    ('-isr99', '')
    """
    return (
        f"-isr{int(round(initial_sparsity * 100))}"
        if initial_sparsity is not None
        else ""
    )


def lambda_token(lambda_value):
    """``-lam<value>`` for the static lambda a fixed-lambda run holds.

    >>> lambda_token(18.0), lambda_token(0.025), lambda_token(None)
    ('-lam18', '-lam0.025', '')
    """
    return f"-lam{lambda_value:g}" if lambda_value is not None else ""


def is_fixed_lambda(method):
    """Whether a method token names a fixed-lambda experiment.

    >>> is_fixed_lambda("bregman_adabreg_fixed"), is_fixed_lambda("proxsgd")
    (True, False)
    """
    return method.endswith("_fixed")


def run_subdir(
    dataset,
    model,
    augmentation,
    experiment,
    sparsity,
    initial_sparsity,
    lambda_value,
    scheduler,
):
    """Build the ``<dataset>/<model>/<aug>/<method>`` stem for the run dir.

    The method token carries ``[-isrNN]``, then either ``-srNN`` (the sparsity
    the run drives toward) or ``-lam<value>`` for a fixed-lambda run, whose
    sparsity is an outcome of that lambda rather than a target. The scheduler
    tag closes it. Each argument is a config interpolation; ``None`` (a field
    absent on a non-image task, no target sparsity on a dense baseline, no
    initial sparsity outside Bregman) becomes a neutral tag or is dropped so
    the path never fails to resolve.

    >>> run_subdir("cifar10", "resnet18", True,
    ...            "img/pruning_mag_struct", 0.9, None, None,
    ...            "torch.optim.lr_scheduler.CosineAnnealingLR")
    'cifar10/resnet18/augmentation/pruning_mag_struct-sr90-CosineAnnealing'
    >>> run_subdir("cifar10", "resnet18", False,
    ...            "img/bregman_adabreg", 0.9, 0.99, 0.01, None)
    'cifar10/resnet18/no_augmentation/bregman_adabreg-isr99-sr90-no_scheduler'
    >>> run_subdir("cifar10", "resnet18", False,
    ...            "img/bregman_adabreg_fixed", 0.9, 0.99, 10.0, None)
    'cifar10/resnet18/no_augmentation/bregman_adabreg_fixed-isr99-lam10-no_scheduler'
    >>> run_subdir("mnist", "resnet18", False, "img/dense_sgd", None, None, None, None)
    'mnist/resnet18/no_augmentation/dense_sgd-no_scheduler'
    """

    aug = "augmentation" if augmentation else "no_augmentation"
    method = (experiment or "no_experiment").rsplit("/", 1)[-1].split(".")[0]
    isr = initial_sparsity_token(initial_sparsity)
    if is_fixed_lambda(method) and lambda_value is not None:
        tag = lambda_token(lambda_value)
    else:
        tag = sparsity_token(sparsity)
    return f"{dataset}/{model}/{aug}/{method}{isr}{tag}-{_scheduler_tag(scheduler)}"


def run_name(output_dir, log_dir):
    """Name the logger run after its run directory, minus the machine root.

    ``output_dir`` is the resolved ``hydra.run.dir``; under the default layout only
    its ``log_dir`` prefix and the constant ``<task>/runs`` head differ between a
    cluster and a laptop, so dropping both gives one dashboard name everywhere. A
    ``hydra.run.dir`` pointed at another volume escapes ``log_dir``; there fall back
    to the tail after a ``runs``/``multiruns`` anchor, else the ``<name>/seed`` pair.

    >>> run_name("/vault/results/train/runs/cifar10/resnet18/no_augmentation/dense_sgd-no_scheduler/seed_42", "/vault/results")
    'cifar10/resnet18/no_augmentation/dense_sgd-no_scheduler/seed_42'
    >>> run_name("/home/logs/eval/runs/cnceleb/sv_vanilla-bs256", "/home/logs/eval/runs/cnceleb")
    'sv_vanilla-bs256'
    >>> run_name("/data/results/cifar10/resnet18/bregman_adabreg-sr99-LambdaDecay/seed_42", "/home/proj/logs")
    'bregman_adabreg-sr99-LambdaDecay/seed_42'
    >>> run_name("/runs/train/runs/mnist/resnet18/dense_sgd/seed_1", "/home/logs")
    'mnist/resnet18/dense_sgd/seed_1'
    """
    relative = os.path.relpath(output_dir, log_dir)
    if relative.startswith(".."):
        parts = output_dir.rstrip(os.sep).split(os.sep)
        for anchor in ("runs", "multiruns"):
            if anchor in parts:
                # Scan from the right: a results root may itself contain "runs".
                last = len(parts) - 1 - parts[::-1].index(anchor)
                return "/".join(parts[last + 1 :])
        return "/".join(parts[-2:])
    parts = relative.split(os.sep)
    if len(parts) > 2 and parts[1] in ("runs", "multiruns"):
        parts = parts[2:]
    return "/".join(parts)


if __name__ == "__main__":
    print(
        run_subdir(
            "cifar10",
            "resnet18",
            False,
            "img/bregman_adabreg.yaml",
            0.9,
            0.99,
            0.01,
            None,
        )
    )
    print(
        run_subdir(
            "cifar10",
            "resnet18",
            False,
            "img/bregman_adabreg_fixed",
            0.9,
            0.99,
            10.0,
            None,
        )
    )
    print(
        run_subdir(
            "cifar100",
            "wrn28_10",
            True,
            "img/dense_sgd",
            None,
            None,
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
            None,
            None,
            "torch.optim.lr_scheduler.CosineAnnealingLR",
        )
    )
    print(
        run_name(
            "/results/train/runs/cifar10/wrn28_10/dense_sgd/seed_42",
            "/results",
        )
    )
