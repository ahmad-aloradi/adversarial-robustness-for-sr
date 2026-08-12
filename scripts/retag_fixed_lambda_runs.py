"""Rename finished fixed-lambda run dirs from ``-sr<NN>`` to ``-lam<value>``.

A ``*_fixed`` run holds a static lambda and lands at whatever sparsity that
lambda reaches, so its name now carries the lambda instead of a target it never
aimed at (``src/utils/run_naming.py``). This renames runs that finished under
the old spelling, reading each one's lambda from its own Hydra snapshot.

Runs already carrying ``-lam`` are skipped; non-fixed runs are never touched.
Dry run by default; ``--apply`` performs the renames.

    python scripts/retag_fixed_lambda_runs.py
    python scripts/retag_fixed_lambda_runs.py --apply
    python scripts/retag_fixed_lambda_runs.py --root /data/aloradad/results/cifar10/resnet18
"""
import argparse
import glob
import os
import re
import sys

from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.bregman_utils import get_bregman_lambda  # noqa: E402
from src.utils.run_naming import is_fixed_lambda, lambda_token  # noqa: E402

# DEFAULT_ROOT = "/home/vault/dsnf/dsnf101h/results/train/runs/cifar100/resnet18"
DEFAULT_ROOT = "/data/aloradad/results/cifar10/resnet18"


def register_resolvers():
    """Register the resolvers a snapshot's ``_bregman_lambda`` runs through.

    Hydra stores ``config.yaml`` unresolved, so the stored value is an
    interpolation (``${bregman_lambda:...}``, or an ``${oc.eval:...}`` product
    of it and a sweep factor) rather than a number.
    """
    if not OmegaConf.has_resolver("oc.eval"):
        OmegaConf.register_new_resolver("oc.eval", eval)
    if not OmegaConf.has_resolver("bregman_lambda"):
        OmegaConf.register_new_resolver("bregman_lambda", get_bregman_lambda)


def read_lambda(run_dir):
    """Return the ``_bregman_lambda`` every Hydra snapshot under ``run_dir``
    agrees on.

    Read from the run's own snapshot rather than recomputed, so the token in
    the new name states the lambda the run actually held. Looks one level down
    for the per-seed layout and at the run itself for the flat one.
    """
    cfg_paths = sorted(
        glob.glob(os.path.join(run_dir, "*", ".hydra", "config.yaml"))
    ) or sorted(glob.glob(os.path.join(run_dir, ".hydra", "config.yaml")))
    assert cfg_paths, f"no .hydra/config.yaml under {run_dir}"

    register_resolvers()
    values = set()
    for cfg_path in cfg_paths:
        cfg = OmegaConf.load(cfg_path)
        assert (
            "_bregman_lambda" in cfg
        ), f"_bregman_lambda absent from {cfg_path}"
        values.add(
            float(
                OmegaConf.select(cfg, "_bregman_lambda", throw_on_missing=True)
            )
        )

    assert (
        len(values) == 1
    ), f"seeds disagree on lambda in {run_dir}: {sorted(values)}"
    return values.pop()


def retagged(name, lambda_value):
    """Replace the ``-sr<NN>`` target tag with ``-lam<value>``.

    >>> retagged("bregman_adabreg_fixed-resnet18-cifar100-bs128-isr99-sr99", 18.0)
    'bregman_adabreg_fixed-resnet18-cifar100-bs128-isr99-lam18'
    >>> retagged("bregman_linbreg_fixed-isr50-sr95-CosineAnnealing", 0.025)
    'bregman_linbreg_fixed-isr50-lam0.025-CosineAnnealing'
    """
    assert "-lam" not in name, f"{name} already carries a -lam tag"
    new_name, hits = re.subn(
        r"-sr\d+", lambda_token(lambda_value), name, count=1
    )
    assert hits == 1, f"no -sr<NN> tag to replace in {name}"
    return new_name


def method_token(name):
    """The method part of a run name — everything before the first tag.

    >>> method_token("bregman_adabreg_fixed-resnet18-cifar100-bs128-sr99")
    'bregman_adabreg_fixed'
    """
    return name.split("-")[0].split(".")[0]


def plan(root):
    """Return the (action, src, dst) list for every fixed-lambda run under
    ``root``.

    Walks the tree so both the flat curated layout and the
    ``<augmentation>/<method>/seed_N`` run_subdir layout are covered.
    """
    assert os.path.isdir(root), f"no such directory: {root}"

    actions = []
    for dirpath, dirnames, _ in os.walk(root):
        matched = [
            d for d in sorted(dirnames) if is_fixed_lambda(method_token(d))
        ]
        # A run's own subtree (seed_*, checkpoints, .hydra) holds no further runs.
        dirnames[:] = [d for d in dirnames if d not in matched]
        for dirname in matched:
            run_dir = os.path.join(dirpath, dirname)
            if "-lam" in dirname:
                actions.append(("skip", run_dir, run_dir))
            elif re.search(r"-sr\d+", dirname):
                new_name = retagged(dirname, read_lambda(run_dir))
                actions.append(
                    ("rename", run_dir, os.path.join(dirpath, new_name))
                )
    return actions


def apply_actions(actions):
    """Rename every planned run, refusing to overwrite an existing target.

    Every destination is checked before the first rename, so a collision leaves
    the tree untouched rather than half-migrated.
    """
    renames = [
        (src, dst) for action, src, dst in actions if action == "rename"
    ]
    for _, dst in renames:
        assert not os.path.exists(dst), f"target already exists: {dst}"
    for src, dst in renames:
        os.rename(src, dst)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument(
        "--apply", action="store_true", help="perform the renames"
    )
    args = parser.parse_args()

    actions = plan(args.root)
    print(f"\n{args.root}  ({len(actions)} fixed-lambda runs)")
    total = {"rename": 0, "skip": 0}
    for action, src, dst in actions:
        total[action] += 1
        arrow = (
            ""
            if action == "skip"
            else f" -> {os.path.relpath(dst, args.root)}"
        )
        print(f"  {action:6s} {os.path.relpath(src, args.root)}{arrow}")

    if args.apply:
        apply_actions(actions)

    verb = "applied" if args.apply else "planned (dry run — pass --apply)"
    print(
        f"\n{verb}: {total['rename']} renamed, {total['skip']} already tagged"
    )


if __name__ == "__main__":
    main()
