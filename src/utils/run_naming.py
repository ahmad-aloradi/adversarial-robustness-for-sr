"""Run naming: the shared name tokens and the logger run name.

``scripts/fabfile.py`` builds every submitted run's name from the tokens here,
so the launcher and ``scripts/retag_fixed_lambda_runs.py`` spell a token the
same way. ``configs/logger/wandb.yaml`` calls ``run_name`` to label the
dashboard entry after the run directory.
"""

import os


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
    print(initial_sparsity_token(0.99) + sparsity_token(0.9))
    print(initial_sparsity_token(0.5) + lambda_token(18.0))
    print(is_fixed_lambda("bregman_adabreg_fixed"))
    print(
        run_name(
            "/results/train/runs/cifar10/wrn28_10/dense_sgd/seed_42",
            "/results",
        )
    )
