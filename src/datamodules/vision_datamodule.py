"""Generic datamodule for CIFAR-10/100, MNIST, and TinyImageNet.

Train pipeline when ``augmentation`` is true: ``transforms.augment`` (PIL-space)
then ``transforms.base`` (ToTensor + Normalize) then ``transforms.augment_post``
(tensor-space); when false, ``base`` alone. Eval uses ``transforms.eval``.
Validation is a class-stratified ``valid_dataset.split`` fraction of the train
set (seed ``valid_dataset.split_seed``); the test set is separate.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from hydra.utils import instantiate
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from src import utils

log = utils.get_pylogger(__name__)


class GaussianNoise:
    """Add Gaussian noise to a tensor."""

    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class VisionDataModule(LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters(logger=False)
        assert (
            "augmentation" in self.hparams
        ), "config must set `augmentation: true|false`"
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def prepare_data(self) -> None:
        for split in ("train_dataset", "test_dataset"):
            block = self.hparams.dataset[split]
            if "download" in block:
                instantiate(block)  # torchvision download (idempotent)
            else:
                root = block.root
                assert Path(
                    root
                ).is_dir(), (
                    f"{root} not found. Run: {self.hparams.dataset.prep_hint}"
                )

    def _build_split(self, split_key: str, specs):
        transform = Compose([instantiate(t) for t in specs])
        return instantiate(
            self.hparams.dataset[split_key], transform=transform
        )

    def _train_specs(self):
        t = self.hparams.transforms
        if self.hparams.augmentation:
            # augment (PIL-space) before ToTensor; augment_post (tensor-space) after
            return list(t.augment) + list(t.base) + list(t.augment_post)
        return list(t.base)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ("fit", None):
            train_view = self._build_split("train_dataset", self._train_specs())
            val_view = self._build_split("train_dataset", self.hparams.transforms.eval)

            targets = np.asarray(train_view.targets)
            assert len(targets) == len(train_view), "dataset .targets length mismatch"
            # stratified so every class appears in both train and val
            train_idx, valid_idx = train_test_split(
                np.arange(len(train_view)),
                test_size=self.hparams.dataset.valid_dataset.split,
                stratify=targets,
                random_state=self.hparams.dataset.valid_dataset.split_seed,
            )
            self.train_data = torch.utils.data.Subset(train_view, train_idx.tolist())
            self.val_data = torch.utils.data.Subset(val_view, valid_idx.tolist())

            log.info(f"train/val split: {len(self.train_data)}, {len(self.val_data)}")

        if stage in ("test", None):
            self.test_data = self._build_split("test_dataset", self.hparams.transforms.eval)
            log.info(f"test split: {len(self.test_data)}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data, **self.hparams.loaders.train)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_data, **self.hparams.loaders.valid)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_data, **self.hparams.loaders.test)


if __name__ == "__main__":
    import sys

    import hydra
    import pyrootutils
    from hydra import compose, initialize_config_dir

    root = pyrootutils.setup_root(
        search_from=__file__,
        indicator=[".git", "pyproject.toml"],
        pythonpath=True,
        dotenv=True,
    )
    cfgd = str(root / "configs")
    # Downloads CIFAR-10 (~170 MB) on first run; override with e.g.
    # datamodule=datasets/mnist.
    ov = ["datamodule=datasets/cifar10", "logger=[]"] + sys.argv[1:]
    with initialize_config_dir(version_base="1.3", config_dir=cfgd):
        cfg = compose(config_name="train.yaml", overrides=ov)
    dm = hydra.utils.instantiate(cfg.datamodule, _recursive_=False)
    dm.prepare_data()
    dm.setup()
    images, targets = next(iter(dm.train_dataloader()))
    print(
        f"train images={tuple(images.shape)} targets={tuple(targets.shape)}; "
        f"val={len(dm.val_data)} test={len(dm.test_data)}"
    )
