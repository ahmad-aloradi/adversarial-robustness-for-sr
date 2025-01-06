from typing import Dict, List, Optional
import os

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import pandas as pd
import hydra

from src import utils
from src.datamodules.components.librispeech.librispeech_dataset import LibrispeechDataset, Collate
from src.datamodules.components.librispeech.librispeech_prep import generate_csvs, write_dataset_csv
from src.datamodules.components.utils import CsvProcessor

log = utils.get_pylogger(__name__)


class LibrispeechDataModule(LightningDataModule):
    def __init__(self,
                 dataset: Dict[str, Dict[str, int]],
                 transforms: Optional[List[Dict]],
                 loaders: Dict[str, Dict[str, int]],
                 *args, **kwargs):
        super().__init__()
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.dataset = dataset
        self.transforms = transforms
        self.loaders = loaders
        self.csv_processor = CsvProcessor(verbose=self.dataset.verbose, fill_value='N/A')

    def prepare_data(self, config):
        dfs_data, df_speaker = generate_csvs(self.dataset, 
                                             delimiter=self.dataset.sep, 
                                             save_csv=self.dataset.save_csv)

        # save the updated csv
        os.makedirs(self.dataset.artifacts_dir, exist_ok=True)
        write_dataset_csv(df_speaker, self.dataset.speaker_csv_exp_filepath, sep=self.dataset.sep)
        for df_path, df in dfs_data.items():
            write_dataset_csv(df, df_path, sep=self.dataset.sep)

        # Get class id and speaker stats
        updated_dev_csv, speaker_lookup_csv = self.csv_processor.process(
            dataset_files=list(dfs_data.keys()),
            spks_metadata_paths=[self.dataset.speaker_csv_path],
            verbose=self.dataset.verbose)
        
        # save the updated csv
        write_dataset_csv(speaker_lookup_csv, self.dataset.speaker_csv_exp_filepath, sep=self.dataset.sep)
        for path in [
            self.dataset.train_csv_exp_filepath, self.dataset.dev_csv_exp_filepath,
            self.dataset.test_csv_exp_filepath]:
            write_dataset_csv(updated_dev_csv[updated_dev_csv.split == os.path.splitext(os.path.basename(path))[0]], 
                              path, sep=self.dataset.sep)

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_data = LibrispeechDataset(
                self.dataset.dataset_dir,
                self.dataset.train_csv_exp_filepath,
                self.dataset.sample_rate,
                self.dataset.max_duration
            )
            self.val_data = LibrispeechDataset(
                self.dataset.dataset_dir,
                self.dataset.dev_csv_exp_filepath,
                self.dataset.sample_rate,
                self.dataset.max_duration
            )
        if stage == 'test' or stage is None:
            self.test_data = LibrispeechDataset(
                self.dataset.dataset_dir,
                self.dataset.test_csv_exp_filepath,
                self.dataset.sample_rate,
                self.dataset.max_duration
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.loaders.train.batch_size,
            num_workers=self.loaders.train.num_workers,
            shuffle=self.loaders.train.shuffle,
            drop_last=self.loaders.train.drop_last,
            pin_memory=self.loaders.train.pin_memory,
            collate_fn=Collate(pad_value=0.0)
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.loaders.valid.batch_size,
            num_workers=self.loaders.valid.num_workers,
            shuffle=self.loaders.valid.shuffle,
            drop_last=self.loaders.valid.drop_last,
            pin_memory=self.loaders.valid.pin_memory,
            collate_fn=Collate(pad_value=0.0)
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.loaders.test.batch_size,
            num_workers=self.loaders.test.num_workers,
            shuffle=self.loaders.test.shuffle,
            pin_memory=self.loaders.test.pin_memory,
            drop_last=self.loaders.test.drop_last,
            collate_fn=Collate(pad_value=0.0)
        )


@hydra.main(config_path="../../configs", config_name="train", version_base="1.3")
def test_datamodule(cfg):

    print("Starting Librispeech DataModule test...")

    datamodule: LibrispeechDataModule = hydra.utils.instantiate(
        cfg.datamodule, _recursive_=False)

    datamodule.prepare_data(datamodule)

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


if __name__ == "__main__":
    test_datamodule()
