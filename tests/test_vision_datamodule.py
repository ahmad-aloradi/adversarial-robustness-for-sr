"""Tests for VisionDataModule.

Fast path uses a synthetic labeled dataset (``FakeImages``) to check transform
wiring, batch shapes, that eval loaders don't shuffle, and that the stratified
carve puts every class in val. A slow path downloads real MNIST to exercise the
download branch of ``prepare_data`` and the val split. The stratified split
needs ``.targets``, which torchvision ``FakeData`` lacks — hence ``FakeImages``.
"""

import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import RandomSampler, SequentialSampler

from src.datamodules.vision_datamodule import VisionDataModule

_TOTENSOR = {"_target_": "torchvision.transforms.ToTensor"}


class FakeImages(torch.utils.data.Dataset):
    """Synthetic PIL-image dataset with balanced ``.targets`` for split
    tests."""

    def __init__(
        self, size, num_classes, image_size=(3, 32, 32), transform=None
    ):
        self.transform = transform
        self.targets = [
            i % num_classes for i in range(size)
        ]  # balanced labels
        self._wh = (image_size[2], image_size[1])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        img = Image.new("RGB", self._wh)
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[index]


def _fake(size=40, num_classes=10):
    return {
        "_target_": "tests.test_vision_datamodule.FakeImages",
        "size": size,
        "num_classes": num_classes,
        "image_size": [3, 32, 32],
    }


def _loaders(num_workers=0):
    common = {"batch_size": 4, "num_workers": num_workers}
    return {
        "train": {**common, "shuffle": True},
        "valid": {**common, "shuffle": False},
        "test": {**common, "shuffle": False},
    }


def _dm(augmentation, transforms):
    fake = _fake()
    cfg = OmegaConf.create(
        {
            "dataset": {
                "train_dataset": fake,
                "valid_dataset": {"split": 0.25, "split_seed": 0},
                "test_dataset": fake,
            },
            "augmentation": augmentation,
            "transforms": transforms,
            "num_classes": 10,
            "loaders": _loaders(),
        }
    )
    dm = VisionDataModule(**cfg)
    dm.setup()  # synthetic data needs no prepare_data/download
    return dm


# ds is a Subset; its transforms live on the underlying view.
_types = lambda ds: [type(t) for t in ds.dataset.transform.transforms]


def test_fakedata_mechanics():
    dm = _dm(
        False,
        {
            "base": [_TOTENSOR],
            "augment": [],
            "augment_post": [],
            "eval": [_TOTENSOR],
        },
    )

    images, targets = next(iter(dm.train_dataloader()))
    assert images.shape == (4, 3, 32, 32)
    assert images.dtype == torch.float32
    assert targets.dtype == torch.long

    # Train shuffles, eval does not.
    assert isinstance(dm.train_dataloader().sampler, RandomSampler)
    assert isinstance(dm.val_dataloader().sampler, SequentialSampler)
    assert isinstance(dm.test_dataloader().sampler, SequentialSampler)

    # Stratified carve: every class appears in val, and train/val are disjoint.
    val_labels = {dm.val_data[i][1] for i in range(len(dm.val_data))}
    assert val_labels == set(range(10))
    assert set(dm.train_data.indices).isdisjoint(dm.val_data.indices)


def test_augmentation_flag_gates_train_transforms():
    from torchvision.transforms import RandomErasing, RandomHorizontalFlip

    flip = {"_target_": "torchvision.transforms.RandomHorizontalFlip"}
    erase = {"_target_": "torchvision.transforms.RandomErasing"}
    specs = {
        "base": [_TOTENSOR],
        "augment": [flip],  # PIL-space
        "augment_post": [erase],  # tensor-space
        "eval": [_TOTENSOR],
    }

    on, off = _dm(True, specs), _dm(False, specs)
    # Both augment lists (PIL-space and tensor-space) are gated by the flag.
    assert {RandomHorizontalFlip, RandomErasing} <= set(_types(on.train_data))
    assert not {RandomHorizontalFlip, RandomErasing} & set(
        _types(off.train_data)
    )
    # Val/eval is never augmented.
    assert not {RandomHorizontalFlip, RandomErasing} & set(_types(on.val_data))


def test_unaugmented_train_uses_eval_pipeline():
    """When base and eval differ (the ImageNet shape), un-augmented train must
    take eval's geometry so train and test see the same crop."""
    from torchvision.transforms import CenterCrop, Resize

    resize = {"_target_": "torchvision.transforms.Resize", "size": 24}
    crop = {"_target_": "torchvision.transforms.CenterCrop", "size": 16}
    rrc = {"_target_": "torchvision.transforms.RandomResizedCrop", "size": 16}
    specs = {
        "base": [_TOTENSOR],  # no geometry, unlike eval
        "augment": [rrc],
        "augment_post": [],
        "eval": [resize, crop, _TOTENSOR],
    }

    off = _dm(False, specs)
    assert _types(off.train_data) == _types(off.val_data)
    images, _ = next(iter(off.train_dataloader()))
    assert images.shape == (4, 3, 16, 16)  # eval crop, not the raw 32x32

    # With augmentation on, geometry comes from augment and eval is not reused.
    on = _dm(True, specs)
    assert {Resize, CenterCrop}.isdisjoint(_types(on.train_data))


@pytest.mark.slow
def test_mnist_real_download(tmp_path):
    root = str(tmp_path / "mnist")
    block = lambda train: {
        "_target_": "torchvision.datasets.MNIST",
        "root": root,
        "train": train,
        "download": True,
    }
    normalize = {
        "_target_": "torchvision.transforms.Normalize",
        "mean": [0.1307],
        "std": [0.3081],
    }
    cfg = OmegaConf.create(
        {
            "dataset": {
                "train_dataset": block(True),
                "valid_dataset": {"split": 0.1, "split_seed": 0},
                "test_dataset": block(False),
            },
            "augmentation": False,
            "transforms": {
                "base": [_TOTENSOR, normalize],
                "augment": [],
                "augment_post": [],
                "eval": [_TOTENSOR, normalize],
            },
            "num_classes": 10,
            "loaders": _loaders(),
        }
    )
    dm = VisionDataModule(**cfg)
    dm.prepare_data()
    dm.setup()

    # Validation is 10% of the 60k train set; the 10k test set stays separate.
    assert len(dm.train_data) == 54000
    assert len(dm.val_data) == 6000
    assert len(dm.test_data) == 10000
    # Stratified: all 10 digits appear in val.
    val_labels = {int(dm.val_data[i][1]) for i in range(len(dm.val_data))}
    assert val_labels == set(range(10))
    images, _ = next(iter(dm.val_dataloader()))
    assert images.shape[1] == 1
    # Normalized MNIST has roughly zero mean, unit-ish scale.
    assert -1.0 < images.mean().item() < 1.0
