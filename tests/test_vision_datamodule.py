"""Tests for VisionDataModule.

Fast path uses torchvision FakeData (no network) to check transform wiring,
batch shapes, and that eval loaders don't shuffle. A slow path downloads real
MNIST to exercise the download branch of ``prepare_data`` and the val split.
"""

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import RandomSampler, SequentialSampler

from src.datamodules.vision_datamodule import VisionDataModule

_TOTENSOR = {"_target_": "torchvision.transforms.ToTensor"}


def _loaders(num_workers=0):
    common = {"batch_size": 4, "num_workers": num_workers}
    return {
        "train": {**common, "shuffle": True},
        "valid": {**common, "shuffle": False},
        "test": {**common, "shuffle": False},
    }


def test_fakedata_mechanics():
    fake = {
        "_target_": "torchvision.datasets.FakeData",
        "size": 16,
        "image_size": [3, 32, 32],
        "num_classes": 10,
    }
    cfg = OmegaConf.create(
        {
            "dataset": {
                "train_dataset": fake,
                "valid_dataset": fake,
                "test_dataset": fake,
            },
            "transforms": {"train": [_TOTENSOR], "eval": [_TOTENSOR]},
            "num_classes": 10,
            "in_channels": 3,
            "loaders": _loaders(),
        }
    )
    dm = VisionDataModule(**cfg)
    dm.setup()  # FakeData needs no prepare_data/download

    images, targets = next(iter(dm.train_dataloader()))
    assert images.shape == (4, 3, 32, 32)
    assert images.dtype == torch.float32
    assert targets.dtype == torch.long

    # Train shuffles, eval does not.
    assert isinstance(dm.train_dataloader().sampler, RandomSampler)
    assert isinstance(dm.val_dataloader().sampler, SequentialSampler)
    assert isinstance(dm.test_dataloader().sampler, SequentialSampler)


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
                "valid_dataset": block(False),
                "test_dataset": block(False),
            },
            "transforms": {
                "train": [_TOTENSOR, normalize],
                "eval": [_TOTENSOR, normalize],
            },
            "num_classes": 10,
            "in_channels": 1,
            "loaders": _loaders(),
        }
    )
    dm = VisionDataModule(**cfg)
    dm.prepare_data()
    dm.setup()

    assert len(dm.val_data) == 10000
    images, _ = next(iter(dm.val_dataloader()))
    assert images.shape[1] == 1
    # Normalized MNIST has roughly zero mean, unit-ish scale.
    assert -1.0 < images.mean().item() < 1.0
