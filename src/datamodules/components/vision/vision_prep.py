"""Pre-download torchvision image datasets (CIFAR-10/100, MNIST).

HPC compute nodes are usually offline, so datasets must be fetched on a login
node before training. Idempotent — torchvision skips archives already present.
``--data-dir`` must match ``paths.data_dir`` (the repo's ``data/``), since the
datamodule configs read ``${paths.data_dir}/<name>``.

    python src/datamodules/components/vision/vision_prep.py \
        --dataset cifar10 --data-dir data
"""

import argparse
from pathlib import Path

from torchvision import datasets

# name -> (torchvision class, subdir under data-dir matching the datamodule)
DATASETS = {
    "cifar10": (datasets.CIFAR10, "cifar10"),
    "cifar100": (datasets.CIFAR100, "cifar100"),
    "mnist": (datasets.MNIST, "mnist"),
}


def prepare(dataset: str, data_dir: str) -> Path:
    cls, subdir = DATASETS[dataset]
    root = Path(data_dir) / subdir
    for train in (True, False):
        cls(root=str(root), train=train, download=True)
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    root = prepare(args.dataset, args.data_dir)
    print(f"{args.dataset} ready at {root}")


if __name__ == "__main__":
    main()
