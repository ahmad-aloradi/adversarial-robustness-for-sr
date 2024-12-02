from typing import Dict, List, Optional
from dataclasses import dataclass
import pandas as pd

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

import hydra
import sys
sys.path.append(f"/home/aloradi/adversarial-robustness-for-sr")
from src.datamodules.components.voxceleb.voxceleb_dataset import (
    VoxCelebDataset, 
    VoxCelebVerificationDataset, 
    TrainCollate, 
    VerificationCollate)

from src.datamodules.components.voxceleb.voxceleb_prep import VoxCelebProcessor
from src import utils
from src.datamodules.components.utils import CsvProcessor

log = utils.get_pylogger(__name__)

@dataclass
class VoxCelebConfig:
    data_dir: str = "data/voxceleb"  # Change this to voxceleb path
    voxceleb_artifacts_dir: str = "/path/to/voxceleb_artifacts"  # Change this to your path
    veri_test_path: str = "/path/to/veri_test.txt"  # Change this to your path
    veri_test_csv_filename: str = "veri_test.csv" 
    verbose: bool = True
    batch_size: int = 32
    num_workers: int = 4
    sample_rate: int = 16000
    min_duration: float = 0.5  # minimum duration in seconds
    max_duration: float = 8.0  # maximum duration in seconds
    dev_csv_filename: str = "voxceleb_dev.csv"
    train_csv_filename: str = "train_csv.csv",
    val_csv_filename: str = "val_csv.csv",


class VoxCelebDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.csv_processor = CsvProcessor(verbose=self.cfg.dataset.verbose,
                                          fill_value='N/A')

    def prepare_data(self):
        voxceleb_processor = VoxCelebProcessor(root_dir=self.cfg.dataset.data_dir,  
                                               verbose=self.cfg.dataset.verbose, 
                                               artifcats_dir=self.cfg.dataset.voxceleb_artifacts_dir,
                                               sep=self.cfg.dataset.sep)
        
        _, _ = voxceleb_processor.generate_metadata(
            base_search_dir=self.cfg.dataset.base_search_dir,
            min_duration=self.cfg.dataset.min_duration,
            save_df=self.cfg.dataset.save_csv
            )

        # Get class id and speaker stats
        updated_dev_csv, speaker_lookup_csv = self.csv_processor.process(
            dataset_files=[self.cfg.dataset.dev_csv_file],
            spks_metadata_paths=[self.cfg.dataset.metadata_csv_file],
            verbose=self.cfg.dataset.verbose)
        
        # save the updated csv
        VoxCelebProcessor.save_csv(updated_dev_csv, self.cfg.dataset.dev_csv_file)
        VoxCelebProcessor.save_csv(speaker_lookup_csv, self.cfg.dataset.speaker_lookup)
        
        # split the dataset into train and validation
        CsvProcessor.split_dataset(
            df=updated_dev_csv,
            train_ratio = self.cfg.dataset.train_ratio,
            save_csv=self.cfg.dataset.save_csv,
            speaker_overlap=self.cfg.dataset.speaker_overlap,
            speaker_id_col='speaker_id',
            train_csv=self.cfg.dataset.train_csv_file,
            val_csv=self.cfg.dataset.val_csv_file,
            sep=self.cfg.dataset.sep,
            seed=self.cfg.dataset.seed
            )
        
        # enrich the verification file
        _ = voxceleb_processor.enrich_verification_file(
            veri_test_path=self.cfg.dataset.veri_test_path,
            metadata_path=self.cfg.dataset.metadata_csv_file,
            output_path=self.cfg.dataset.veri_test_output_path,
            sep=self.cfg.dataset.sep,
            )
        

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_data = VoxCelebDataset(
                self.cfg.dataset.wav_dir,
                self.cfg.dataset.train_csv_file,
                self.cfg.dataset.sample_rate,
                self.cfg.dataset.max_duration
            )
            self.val_data = VoxCelebDataset(
                self.cfg.dataset.wav_dir,
                self.cfg.dataset.val_csv_file,
                self.cfg.dataset.sample_rate,
                self.cfg.dataset.max_duration
            )
        
        if stage == 'test' or stage is None:
            self.test_data = VoxCelebVerificationDataset(
                self.cfg.dataset.wav_dir,
                self.cfg.dataset.veri_test_path,
                self.cfg.dataset.sample_rate,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.cfg.loaders.train.batch_size,
            num_workers=self.cfg.loaders.train.num_workers,
            shuffle=self.cfg.loaders.train.shuffle,
            drop_last=self.cfg.loaders.train.drop_last,
            pin_memory=self.cfg.loaders.train.pin_memory,
            collate_fn=TrainCollate()
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.cfg.loaders.valid.batch_size,
            num_workers=self.cfg.loaders.valid.num_workers,
            shuffle=self.cfg.loaders.valid.shuffle,
            drop_last=self.cfg.loaders.valid.drop_last,
            pin_memory=self.cfg.loaders.valid.pin_memory,
            collate_fn=TrainCollate()
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.cfg.loaders.test.batch_size,
            num_workers=self.cfg.loaders.test.num_workers,
            shuffle=self.cfg.loaders.test.shuffle,
            pin_memory=self.cfg.loaders.test.pin_memory,
            drop_last=self.cfg.loaders.test.drop_last,
            collate_fn=VerificationCollate()
        )


@hydra.main(config_path="../../configs", config_name="train")
def test_datamodule(cfg):

    print("Starting VoxCeleb DataModule test...")

    datamodule: VoxCelebDataModule = hydra.utils.instantiate(
        cfg.datamodule, _recursive_=False)

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


if __name__ == "__main__":
    test_datamodule()