from typing import Dict, List, Optional
from pathlib import Path

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import hydra

from src import utils
from src.datamodules.components.librispeech.librispeech_dataset import LibrispeechDataset, Collate
from src.datamodules.components.utils import CsvProcessor
from src.datamodules.preparation.librispeech import LibrispeechMetadataPreparer

log = utils.get_pylogger(__name__)


class LibrispeechDataModule(LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.csv_processor = CsvProcessor(verbose=self.hparams.dataset.verbose, fill_value='N/A')

    def prepare_data(self):
        preparer = LibrispeechMetadataPreparer(self.hparams.dataset, self.csv_processor)
        preparer.prepare()

    def _artifacts_ready(self) -> bool:
        required_artifacts = [
            self.hparams.dataset.train_csv_exp_filepath,
            self.hparams.dataset.dev_csv_exp_filepath,
            self.hparams.dataset.test_csv_exp_filepath,
            self.hparams.dataset.speaker_csv_exp_filepath,
        ]
        return all(Path(path).exists() for path in required_artifacts)

    def setup(self, stage: Optional[str] = None):
        if not self._artifacts_ready():
            self.prepare_data()

        max_duration = -1 if self.hparams.dataset.use_pre_segmentation else self.hparams.dataset.max_duration

        if stage == 'fit' or stage is None:
            self.train_data = LibrispeechDataset(
                self.hparams.dataset.dataset_dir,
                self.hparams.dataset.train_csv_exp_filepath,
                self.hparams.dataset.sample_rate,
                max_duration
            )
            self.val_data = LibrispeechDataset(
                self.hparams.dataset.dataset_dir,
                self.hparams.dataset.dev_csv_exp_filepath,
                self.hparams.dataset.sample_rate,
                max_duration
            )
        if stage == 'test' or stage is None:
            self.test_data = LibrispeechDataset(
                self.hparams.dataset.dataset_dir,
                self.hparams.dataset.test_csv_exp_filepath,
                self.hparams.dataset.sample_rate,
                max_duration
            )

    def train_dataloader(self):
        assert self.hparams.get('loaders') is not None, "LibrispeechDataModule requires 'loaders' config"
        return DataLoader(
            self.train_data,
            **self.hparams.loaders.train,
            collate_fn=Collate(pad_value=0.0)
        )

    def val_dataloader(self):
        assert self.hparams.get('loaders') is not None, "LibrispeechDataModule requires 'loaders' config"
        return DataLoader(
            self.val_data,
            **self.hparams.loaders.valid,
            collate_fn=Collate(pad_value=0.0)
        )

    def test_dataloader(self):
        assert self.hparams.get('loaders') is not None, "LibrispeechDataModule requires 'loaders' config"
        return DataLoader(
            self.test_data,
            **self.hparams.loaders.test,
            collate_fn=Collate(pad_value=0.0)
        )


if __name__ == "__main__":
    import pyrootutils
    root = pyrootutils.setup_root(search_from=__file__, indicator=[".env"], pythonpath=True, dotenv=True,)
    _HYDRA_PARAMS = {"version_base": "1.3", "config_path": str(root / "configs"), "config_name": "train.yaml"}

    @hydra.main(**_HYDRA_PARAMS)
    def test_datamodule(cfg):

        print("Starting Librispeech DataModule test...")
        datamodule: LibrispeechDataModule = hydra.utils.instantiate(cfg.datamodule, _recursive_=False)
        datamodule.prepare_data()

        assert not datamodule.train_data
        assert not datamodule.val_data
        assert not datamodule.test_data
        # assert not datamodule.predict_set

        datamodule.setup()
        assert datamodule.train_data
        assert datamodule.val_data
        assert datamodule.test_data

        assert datamodule.train_dataloader()
        assert datamodule.val_dataloader()
        assert datamodule.test_dataloader()

        batch = next(iter(datamodule.train_dataloader()))
        print(batch)

    test_datamodule()