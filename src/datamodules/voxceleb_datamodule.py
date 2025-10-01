from typing import Dict, List, Optional
from pathlib import Path

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import pandas as pd
import hydra

from src.datamodules.components.voxceleb.voxceleb_dataset import (
    VoxCelebDataset, 
    VoxCelebVerificationDataset, 
    TrainCollate, 
    VerificationCollate,
    VoxCelebEnroll,
    EnrollCoallate)
from src import utils
from src.datamodules.components.utils import CsvProcessor
from src.datamodules.components.common import VoxcelebDefaults, get_dataset_class
from src.datamodules.preparation.voxceleb import VoxCelebMetadataPreparer


log = utils.get_pylogger(__name__)
DATASET_DEFAULTS = VoxcelebDefaults()
DATASET_CLS, DF_COLS = get_dataset_class(DATASET_DEFAULTS.dataset_name)


class VoxCelebDataModule(LightningDataModule):
    def __init__(self,
                 dataset: Dict[str, Dict[str, int]],
                 transforms: Optional[List[Dict]],
                 loaders: Dict[str, Dict[str, int]],
                 *args, **kwargs):
        super().__init__()
        self.train_data = None
        self.val_data = None
        self.dataset = dataset
        self.transforms = transforms
        self.loaders = loaders
        self.csv_processor = CsvProcessor(verbose=self.dataset.verbose, fill_value='N/A')
        self.enroll_data_dict: Dict[str, pd.DataFrame] = {}
        self.unique_trial_data_dict: Dict[str, pd.DataFrame] = {}
        self.test_dataframes: Dict[str, pd.DataFrame] = {}

    def prepare_data(self):
        preparer = VoxCelebMetadataPreparer(self.dataset, self.csv_processor)
        result = preparer.prepare()

        self.enroll_data_dict = result.test.enroll_frames
        self.unique_trial_data_dict = result.test.unique_trial_frames
        self.test_dataframes = result.extras["test_dataframes"]

    def _artifacts_ready(self) -> bool:
        required_files = [
            self.dataset.train_csv_file,
            self.dataset.val_csv_file,
            self.dataset.metadata_csv_file,
            self.dataset.speaker_lookup,
            *self.dataset.veri_test_output_paths.values(),
        ]

        have_files = all(Path(path).exists() for path in required_files)
        expected_tests = list(self.dataset.veri_test_filenames)
        have_enroll = (
            bool(self.enroll_data_dict)
            and bool(self.unique_trial_data_dict)
            and all(name in self.enroll_data_dict for name in expected_tests)
            and all(name in self.unique_trial_data_dict for name in expected_tests)
        )
        return have_files and have_enroll

    def setup(self, stage: Optional[str] = None):
        if not self._artifacts_ready():
            self.prepare_data()

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
            # Setup test datasets for all configured test sets
            self.test_data_dict = {}
            self.enrollment_data_dict = {}
            self.test_unique_data_dict = {}
            
            for test_filename in self.dataset.veri_test_filenames:
                # Create test verification dataset for each test set
                self.test_data_dict[test_filename] = VoxCelebVerificationDataset(
                    self.dataset.wav_dir,
                    self.dataset.veri_test_output_paths[test_filename],
                    self.dataset.sample_rate,
                )
                
                # Create enrollment dataset for each test set
                self.enrollment_data_dict[test_filename] = VoxCelebEnroll(
                    data_dir=self.dataset.wav_dir,
                    phase='enrollment',
                    sample_rate=self.dataset.sample_rate,
                    dataset=self.enroll_data_dict[test_filename]
                )
                
                # Create test unique dataset for each test set  
                self.test_unique_data_dict[test_filename] = VoxCelebEnroll(
                    data_dir=self.dataset.wav_dir,
                    phase='test',
                    sample_rate=self.dataset.sample_rate,
                    dataset=self.unique_trial_data_dict[test_filename]
                )
                log.info(f"Test set '{test_filename}' has {len(self.test_data_dict[test_filename])} trials, "
                         f"enrollment set has {len(self.enrollment_data_dict[test_filename])} trials, "
                         f"unique set has {len(self.test_unique_data_dict[test_filename])} trials.")

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
        """Return dictionary of test dataloaders for all configured test sets."""
        test_dataloaders = {}
        for test_filename in self.dataset.veri_test_filenames:
            test_dataloaders[test_filename] = DataLoader(
                self.test_data_dict[test_filename],
                batch_size=self.loaders.test.batch_size,
                num_workers=self.loaders.test.num_workers,
                shuffle=self.loaders.test.shuffle,
                pin_memory=self.loaders.test.pin_memory,
                drop_last=self.loaders.test.drop_last,
                collate_fn=VerificationCollate()
            )
        return test_dataloaders

    def get_enroll_and_trial_dataloaders(self, test_filename: str) -> tuple:
        """Get dataloaders for a specific test set.

        Args:
            test_filename: Name of the test set (e.g., 'veri_test2', 'veri_test_extended2', 'veri_test_hard2')
        Returns:
            Tuple of (enroll_dataloader, trial_unique_dataloader)
        """
        if test_filename not in self.dataset.veri_test_filenames:
            raise ValueError(f"Unknown test set: {test_filename}. Available test sets: {self.dataset.veri_test_filenames}")

        # Enrollment dataloader
        enroll_dataloader = DataLoader(
            self.enrollment_data_dict[test_filename],
            batch_size=self.loaders.enrollment.batch_size,
            shuffle=self.loaders.enrollment.shuffle,
            num_workers=self.loaders.enrollment.num_workers,
            pin_memory=self.loaders.enrollment.pin_memory,
            collate_fn=EnrollCoallate()
        )

        # Trial unique dataloader (to compute test embeddings per unique utterance, not per trial)
        trial_unique_dataloader = DataLoader(
            self.test_unique_data_dict[test_filename],
            batch_size=self.loaders.enrollment.batch_size,
            shuffle=self.loaders.enrollment.shuffle,
            num_workers=self.loaders.enrollment.num_workers,
            pin_memory=self.loaders.enrollment.pin_memory,
            collate_fn=EnrollCoallate()
        )

        return enroll_dataloader, trial_unique_dataloader


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
        assert not hasattr(datamodule, 'test_data') or datamodule.test_data is None
        # assert not datamodule.predict_set

        datamodule.setup()
        assert datamodule.train_data
        assert datamodule.val_data
        assert datamodule.test_data_dict  # Now we have test_data_dict instead of test_data

        assert datamodule.train_dataloader()
        assert datamodule.val_dataloader()
        assert datamodule.test_dataloader()

        batch = next(iter(datamodule.train_dataloader()))
        print(batch)
    
    test_datamodule()
