"""Loaders for the published corruption benchmarks ({dataset}-C).

CIFAR-10-C / CIFAR-100-C (Hendrycks & Dietterich, 2019) ship as one
``{corruption}.npy`` of shape (50000, 32, 32, 3) uint8 — the 10k test set
repeated for severities 1..5 in order — plus a ``labels.npy`` of shape
(50000,). Tiny-ImageNet-C ships as an ImageFolder tree
``{root}/{corruption}/{severity}/{wnid}/*``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

_CIFAR_TEST_SIZE = 10_000
_PREP_SCRIPTS = {
    "cifar10": "bash scripts/datasets/prep_cifar10_c.sh",
    "cifar100": "bash scripts/datasets/prep_cifar100_c.sh",
    "tinyimagenet": "bash scripts/datasets/prep_tiny_imagenet_c.sh",
}


class CorruptionDataset(Dataset):
    """One severity slice of a CIFAR-style ``{corruption}.npy`` file.

    Yields (image, label) with images as float32 [0,1] CHW tensors — the same
    contract as the ToTensor-only eval pipeline the robustness eval runs with
    (normalization lives inside the model).
    """

    def __init__(self, root: Path, corruption: str, severity: int):
        assert 1 <= severity <= 5, f"severity must be 1..5, got {severity}"
        images_path = root / f"{corruption}.npy"
        labels_path = root / "labels.npy"
        for p in (images_path, labels_path):
            assert p.exists(), f"{p} not found — corruption data missing."

        # mmap: only the requested severity slice is materialized.
        images = np.load(images_path, mmap_mode="r")
        labels = np.load(labels_path, mmap_mode="r")
        assert images.ndim == 4 and images.shape[1:] == (32, 32, 3), (
            f"unexpected {images_path.name} layout: {images.shape} "
            "(expected (N, 32, 32, 3) uint8 HWC)"
        )
        assert images.dtype == np.uint8, f"expected uint8, got {images.dtype}"
        assert images.shape[0] == 5 * _CIFAR_TEST_SIZE, (
            f"expected {5 * _CIFAR_TEST_SIZE} rows (5 severities x 10k), "
            f"got {images.shape[0]}"
        )
        assert (
            labels.shape[0] == images.shape[0]
        ), f"labels/images mismatch: {labels.shape[0]} vs {images.shape[0]}"

        lo, hi = (severity - 1) * _CIFAR_TEST_SIZE, severity * _CIFAR_TEST_SIZE
        self.images = np.array(images[lo:hi])
        self.labels = np.array(labels[lo:hi]).astype(np.int64)

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int):
        image = torch.from_numpy(self.images[idx]).permute(2, 0, 1)
        return image.float() / 255.0, int(self.labels[idx])


def list_corruptions(dataset_name: str, data_dir: str) -> list[str]:
    """The corruption types present on disk, sorted.

    CIFAR-10-C/CIFAR-100-C ship 19 (the 15 benchmark corruptions plus 4 extra
    ones); Tiny-ImageNet-C ships only the 15. Discovered rather than hardcoded
    so the reported per-severity mean always matches what exists.
    """
    root = Path(data_dir)
    if not root.is_dir():
        hint = _PREP_SCRIPTS.get(dataset_name, "the corruption prep script")
        raise FileNotFoundError(f"{root} not found. Run: {hint}")

    if dataset_name in ("cifar10", "cifar100"):
        types = [p.stem for p in root.glob("*.npy") if p.stem != "labels"]
    elif dataset_name == "tinyimagenet":
        types = [p.name for p in root.iterdir() if p.is_dir()]
    else:
        raise ValueError(
            f"No published corruption set for {dataset_name!r}; "
            f"expected one of {sorted(_PREP_SCRIPTS)}"
        )

    assert types, f"no corruption types found under {root}"
    return sorted(types)


def build_corruption_loader(
    dataset_name: str,
    corruption: str,
    severity: int,
    data_dir: str,
    loader_kwargs: dict,
    class_to_idx: dict[str, int] | None = None,
) -> DataLoader:
    """Dispatch on the training dataset name to its corruption counterpart.

    ``loader_kwargs`` should mirror the run's eval loader (batch_size,
    num_workers, ...); shuffle is forced off. For Tiny-ImageNet-C pass the
    training test set's ``class_to_idx`` so label indices are asserted to
    match the checkpoint's.
    """
    root = Path(data_dir)
    if not root.is_dir():
        hint = _PREP_SCRIPTS.get(dataset_name, "the corruption prep script")
        raise FileNotFoundError(f"{root} not found. Run: {hint}")

    if dataset_name in ("cifar10", "cifar100"):
        dataset = CorruptionDataset(root, corruption, severity)
    elif dataset_name == "tinyimagenet":
        from torchvision import transforms
        from torchvision.datasets import ImageFolder

        split_dir = root / corruption / str(severity)
        assert (
            split_dir.is_dir()
        ), f"{split_dir} not found. Run: {_PREP_SCRIPTS['tinyimagenet']}"
        dataset = ImageFolder(str(split_dir), transform=transforms.ToTensor())
        if class_to_idx is not None:
            assert dataset.class_to_idx == class_to_idx, (
                "Tiny-ImageNet-C class indexing differs from the training "
                "test set — labels would be permuted."
            )
    else:
        raise ValueError(
            f"No published corruption set for {dataset_name!r}; "
            f"expected one of {sorted(_PREP_SCRIPTS)}"
        )

    kwargs = dict(loader_kwargs)
    kwargs["shuffle"] = False
    kwargs.pop("drop_last", None)
    return DataLoader(dataset, drop_last=False, **kwargs)


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rng = np.random.default_rng(0)
        images = rng.integers(0, 256, (50000, 32, 32, 3), dtype=np.uint8)
        np.save(root / "gaussian_noise.npy", images)
        np.save(root / "fog.npy", images)
        np.save(root / "labels.npy", np.tile(np.arange(10), 5000))
        print(
            "types:", list_corruptions("cifar10", str(root))
        )  # fog, gaussian_noise
        ds = CorruptionDataset(root, "gaussian_noise", severity=3)
        x, y = ds[0]
        print(
            "len:",
            len(ds),
            "| shape:",
            tuple(x.shape),
            "| in[0,1]:",
            bool(x.min() >= 0 and x.max() <= 1),
            "| label:",
            y,
        )
