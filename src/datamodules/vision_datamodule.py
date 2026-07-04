"""Generic datamodule for torchvision image benchmarks.

One class covers CIFAR-10/100, MNIST, and TinyImageNet — the datasets differ
only in config-expressible ways (dataset class, transform list, class count).
The dataset blocks and transform lists are instantiated lazily in ``setup`` so
the module can be built with ``_recursive_=False`` (like the audio ones).

Each split's dataset block is a torchvision ``_target_`` (its ``transform`` is
supplied here from ``transforms.{train,eval}``). ``prepare_data`` downloads
blocks that carry a ``download`` key; blocks without one (``ImageFolder``) must
already exist on disk — run the dataset's prep script first.
"""

from pathlib import Path
from typing import Optional

from hydra.utils import instantiate
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from src import utils

log = utils.get_pylogger(__name__)


class VisionDataModule(LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def prepare_data(self) -> None:
        for split in ("train_dataset", "valid_dataset", "test_dataset"):
            block = self.hparams.dataset[split]
            if "download" in block:
                instantiate(block)  # torchvision download (idempotent)
            else:
                root = block.root
                assert Path(root).is_dir(), (f"{root} not found. Run: {self.hparams.dataset.prep_hint}")

    def _build_split(self, split_key: str, transform_key: str):
        transform = Compose(
            [instantiate(t) for t in self.hparams.transforms[transform_key]]
        )
        return instantiate(
            self.hparams.dataset[split_key], transform=transform
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ("fit", None):
            self.train_data = self._build_split("train_dataset", "train")
            self.val_data = self._build_split("valid_dataset", "eval")
        if stage in ("test", None):
            self.test_data = self._build_split("test_dataset", "eval")

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
