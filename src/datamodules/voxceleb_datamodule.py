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
from src.datamodules.preparation.voxceleb import VoxCelebMetadataPreparer, _extract_enroll_test


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
        self.save_hyperparameters(logger=False)
        self.train_data = None
        self.val_data = None
        self.csv_processor = CsvProcessor(verbose=self.hparams.dataset.verbose, fill_value='N/A')
        self.enroll_data_dict: Dict[str, pd.DataFrame] = {}
        self.unique_trial_data_dict: Dict[str, pd.DataFrame] = {}
        self._prepared_test_artifacts = None

    def prepare_data(self):
        """Prepare all metadata artifacts."""
        if self._artifacts_ready():
            if self.hparams.dataset.verbose:
                log.info("Skipping VoxCeleb data preparation because all artifacts are already present.")
            return None

        # Output files don't exist - run preparation
        # (This will check base_search_dir internally for pre-generated files to copy)
        preparer = VoxCelebMetadataPreparer(self.hparams.dataset, self.csv_processor)
        result = preparer.prepare()
        self._prepared_test_artifacts = result.test
        return result

    def _load_test_artifacts(self):
        """Load test enrollment and trial data from disk into memory."""
        self.enroll_data_dict = {}
        self.unique_trial_data_dict = {}

        if self._prepared_test_artifacts is not None:
            enroll_frames = self._prepared_test_artifacts.enroll_frames
            unique_frames = self._prepared_test_artifacts.unique_trial_frames
            for test_filename in self.hparams.dataset.veri_test_filenames:
                if test_filename in enroll_frames and test_filename in unique_frames:
                    self.enroll_data_dict[test_filename] = enroll_frames[test_filename].copy()
                    self.unique_trial_data_dict[test_filename] = unique_frames[test_filename].copy()
                else:
                    self._load_test_artifact_from_disk(test_filename)
            return

        for test_filename in self.hparams.dataset.veri_test_filenames:
            output_path = Path(self.hparams.dataset.veri_test_output_paths[test_filename])
            test_df = pd.read_csv(output_path, sep=self.hparams.dataset.sep)
            self.enroll_data_dict[test_filename] = _extract_enroll_test(test_df, mode="enroll")
            self.unique_trial_data_dict[test_filename] = _extract_enroll_test(test_df, mode="test")

    def _load_test_artifact_from_disk(self, test_filename: str):
        output_path = Path(self.hparams.dataset.veri_test_output_paths[test_filename])
        test_df = pd.read_csv(output_path, sep=self.hparams.dataset.sep)
        self.enroll_data_dict[test_filename] = _extract_enroll_test(test_df, mode="enroll")
        self.unique_trial_data_dict[test_filename] = _extract_enroll_test(test_df, mode="test")

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for the given stage."""
        
        max_duration = -1 if self.hparams.dataset.use_pre_segmentation else self.hparams.dataset.max_duration

        if stage in ('fit', None):
            # Ensure metadata files exist (idempotent)
            if not self._artifacts_ready():
                self.prepare_data()
            
            # Create train/val datasets
            self.train_data = VoxCelebDataset(
                self.hparams.dataset.wav_dir,
                self.hparams.dataset.train_csv_file,
                self.hparams.dataset.sample_rate,
                max_duration
            )
            self.val_data = VoxCelebDataset(
                self.hparams.dataset.wav_dir,
                self.hparams.dataset.val_csv_file,
                self.hparams.dataset.sample_rate,
                max_duration
            )
        
        if stage in ('test', None):
            # Load test artifacts from disk if not already loaded
            if not self.enroll_data_dict or not self.unique_trial_data_dict:
                self._load_test_artifacts()
            
            # Setup test datasets for all configured test sets
            self.test_data_dict = {}
            self.enrollment_data_dict = {}
            self.test_unique_data_dict = {}
            
            for test_filename in self.hparams.dataset.veri_test_filenames:
                # Create test verification dataset for each test set
                self.test_data_dict[test_filename] = VoxCelebVerificationDataset(
                    self.hparams.dataset.wav_dir,
                    self.hparams.dataset.veri_test_output_paths[test_filename],
                    self.hparams.dataset.sample_rate,
                )
                
                # Create enrollment dataset for each test set
                self.enrollment_data_dict[test_filename] = VoxCelebEnroll(
                    data_dir=self.hparams.dataset.wav_dir,
                    phase='enrollment',
                    sample_rate=self.hparams.dataset.sample_rate,
                    dataset=self.enroll_data_dict[test_filename]
                )
                
                # Create test unique dataset for each test set  
                self.test_unique_data_dict[test_filename] = VoxCelebEnroll(
                    data_dir=self.hparams.dataset.wav_dir,
                    phase='test',
                    sample_rate=self.hparams.dataset.sample_rate,
                    dataset=self.unique_trial_data_dict[test_filename]
                )
                log.info(f"Test set '{test_filename}' has {len(self.test_data_dict[test_filename])} trials, "
                         f"enrollment set has {len(self.enrollment_data_dict[test_filename])} trials, "
                         f"unique set has {len(self.test_unique_data_dict[test_filename])} trials.")

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.hparams.loaders.train.batch_size,
            num_workers=self.hparams.loaders.train.num_workers,
            shuffle=self.hparams.loaders.train.shuffle,
            drop_last=self.hparams.loaders.train.drop_last,
            pin_memory=self.hparams.loaders.train.pin_memory,
            collate_fn=TrainCollate()
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.hparams.loaders.valid.batch_size,
            num_workers=self.hparams.loaders.valid.num_workers,
            shuffle=self.hparams.loaders.valid.shuffle,
            drop_last=self.hparams.loaders.valid.drop_last,
            pin_memory=self.hparams.loaders.valid.pin_memory,
            collate_fn=TrainCollate()
        )

    def test_dataloader(self):
        """Return dictionary of test dataloaders for all configured test sets."""
        test_dataloaders = {}
        for test_filename in self.hparams.dataset.veri_test_filenames:
            test_dataloaders[test_filename] = DataLoader(
                self.test_data_dict[test_filename],
                batch_size=self.hparams.loaders.test.batch_size,
                num_workers=self.hparams.loaders.test.num_workers,
                shuffle=self.hparams.loaders.test.shuffle,
                pin_memory=self.hparams.loaders.test.pin_memory,
                drop_last=self.hparams.loaders.test.drop_last,
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
        if test_filename not in self.hparams.dataset.veri_test_filenames:
            raise ValueError(f"Unknown test set: {test_filename}. Available test sets: {self.hparams.dataset.veri_test_filenames}")

        # Enrollment dataloader
        enroll_dataloader = DataLoader(
            self.enrollment_data_dict[test_filename],
            batch_size=self.hparams.loaders.enrollment.batch_size,
            shuffle=self.hparams.loaders.enrollment.shuffle,
            num_workers=self.hparams.loaders.enrollment.num_workers,
            pin_memory=self.hparams.loaders.enrollment.pin_memory,
            collate_fn=EnrollCoallate()
        )

        # Trial unique dataloader (to compute test embeddings per unique utterance, not per trial)
        trial_unique_dataloader = DataLoader(
            self.test_unique_data_dict[test_filename],
            batch_size=self.hparams.loaders.enrollment.batch_size,
            shuffle=self.hparams.loaders.enrollment.shuffle,
            num_workers=self.hparams.loaders.enrollment.num_workers,
            pin_memory=self.hparams.loaders.enrollment.pin_memory,
            collate_fn=EnrollCoallate()
        )

        return enroll_dataloader, trial_unique_dataloader

    def _artifacts_ready(self) -> bool:
        required_files = [
            self.hparams.dataset.train_csv_file,
            self.hparams.dataset.val_csv_file,
            self.hparams.dataset.metadata_csv_file,
            self.hparams.dataset.speaker_lookup,
            *self.hparams.dataset.veri_test_output_paths.values(),
        ]
        return all(Path(path).exists() for path in required_files)


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
