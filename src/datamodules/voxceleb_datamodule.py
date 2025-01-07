from typing import Dict, List, Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import pandas as pd
import hydra

from src.datamodules.components.voxceleb.voxceleb_dataset import (
    VoxCelebDataset, 
    VoxCelebVerificationDataset, 
    TrainCollate, 
    VerificationCollate)
from src.datamodules.components.voxceleb.voxceleb_prep import VoxCelebProcessor
from src import utils
from src.datamodules.components.utils import CsvProcessor

log = utils.get_pylogger(__name__)

class VoxCelebDataModule(LightningDataModule):
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

    def prepare_data(self):
        voxceleb_processor = VoxCelebProcessor(root_dir=self.dataset.data_dir,
                                               verbose=self.dataset.verbose,
                                               artifcats_dir=self.dataset.voxceleb_artifacts_dir,
                                               sep=self.dataset.sep)
        
        _, _ = voxceleb_processor.generate_metadata(
            base_search_dir=self.dataset.base_search_dir,
            min_duration=self.dataset.min_duration,
            save_df=self.dataset.save_csv
            )

        # Get class id and speaker stats
        updated_dev_csv, speaker_lookup_csv = self.csv_processor.process(
            dataset_files=[self.dataset.dev_csv_file],
            spks_metadata_paths=[self.dataset.metadata_csv_file],
            verbose=self.dataset.verbose)
        
        # save the updated csv
        VoxCelebProcessor.save_csv(updated_dev_csv, self.dataset.dev_csv_file)
        VoxCelebProcessor.save_csv(speaker_lookup_csv, self.dataset.speaker_lookup)
        
        # split the dataset into train and validation
        CsvProcessor.split_dataset(
            df=updated_dev_csv,
            train_ratio = self.dataset.train_ratio,
            save_csv=self.dataset.save_csv,
            speaker_overlap=self.dataset.speaker_overlap,
            speaker_id_col='speaker_id',
            train_csv=self.dataset.train_csv_file,
            val_csv=self.dataset.val_csv_file,
            sep=self.dataset.sep,
            seed=self.dataset.seed
            )
        
        # enrich the verification file
        _ = voxceleb_processor.enrich_verification_file(
            veri_test_path=self.dataset.veri_test_path,
            metadata_path=self.dataset.metadata_csv_file,
            output_path=self.dataset.veri_test_output_path,
            sep=self.dataset.sep,
            )
        
    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_data = VoxCelebDataset(
                self.dataset.wav_dir,
                self.dataset.train_csv_file,
                self.dataset.sample_rate,
                self.dataset.max_duration
            )
            self.val_data = VoxCelebDataset(
                self.dataset.wav_dir,
                self.dataset.val_csv_file,
                self.dataset.sample_rate,
                self.dataset.max_duration
            )
        
        if stage == 'test' or stage is None:
            self.test_data = VoxCelebVerificationDataset(
                self.dataset.wav_dir,
                self.dataset.veri_test_path,
                self.dataset.sample_rate,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.loaders.train.batch_size,
            num_workers=self.loaders.train.num_workers,
            shuffle=self.loaders.train.shuffle,
            drop_last=self.loaders.train.drop_last,
            pin_memory=self.loaders.train.pin_memory,
            collate_fn=TrainCollate()
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.loaders.valid.batch_size,
            num_workers=self.loaders.valid.num_workers,
            shuffle=self.loaders.valid.shuffle,
            drop_last=self.loaders.valid.drop_last,
            pin_memory=self.loaders.valid.pin_memory,
            collate_fn=TrainCollate()
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.loaders.test.batch_size,
            num_workers=self.loaders.test.num_workers,
            shuffle=self.loaders.test.shuffle,
            pin_memory=self.loaders.test.pin_memory,
            drop_last=self.loaders.test.drop_last,
            collate_fn=VerificationCollate()
        )


if __name__ == "__main__":
    import pyrootutils
    root = pyrootutils.setup_root(search_from=__file__, indicator=[".env"], pythonpath=True, dotenv=True,)
    _HYDRA_PARAMS = {"version_base": "1.3", "config_path": str(root / "configs"), "config_name": "train.yaml"}

    @hydra.main(**_HYDRA_PARAMS)
    def test_datamodule(cfg):

        print("Starting VoxCeleb DataModule test...")

        datamodule: VoxCelebDataModule = hydra.utils.instantiate(cfg.datamodule, _recursive_=False)

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