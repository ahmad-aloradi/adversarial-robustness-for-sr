from typing import Dict, List, Optional
from pathlib import Path

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import pandas as pd
import hydra


from src.datamodules.components.cnceleb.cnceleb_dataset import (
    CNCelebDataset, 
    CNCelebVerificationDataset, 
    CNCelebEnroll,
    CNCelebTest,
    TrainCollate, 
    VerificationCollate,
    EnrollCollate
)

from src import utils
from src.datamodules.components.utils import CsvProcessor
from src.datamodules.components.common import CNCelebDefaults, get_dataset_class
from src.datamodules.preparation.cnceleb import CNCelebMetadataPreparer

log = utils.get_pylogger(__name__)
DATASET_DEFAULTS = CNCelebDefaults()
DATASET_CLS, DF_COLS = get_dataset_class(DATASET_DEFAULTS.dataset_name)


class CNCelebDataModule(LightningDataModule):
    """
    CNCeleb DataModule with standardized interface for sv.py
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.enrollment_data = None
        self.test_unique_data = None
        
        self.csv_processor = CsvProcessor(
            verbose=self.hparams.dataset.get('verbose', True), 
            fill_value='N/A'
        )

    @property
    def enroll_csv_path(self) -> Path:
        """Path to the enrollment CSV file."""
        return Path(self.hparams.dataset.artifacts_dir) / "enroll.csv"

    @property
    def test_unique_csv_path(self) -> Path:
        """Path to the unique test items CSV file."""
        return Path(self.hparams.dataset.artifacts_dir) / "test_unique.csv"

    def prepare_data(self):
        """Prepares the CNCeleb dataset by copying from base_search_dir or generating metadata, splitting, and creating trial lists."""
        log.info("Preparing CNCeleb data...")
        preparer = CNCelebMetadataPreparer(self.hparams.dataset, self.csv_processor)
        result = preparer.prepare()
        if result.extras.get("core_files_copied"):
            log.info("Loaded CNCeleb metadata from pre-generated artifacts.")
        else:
            log.info("Generated CNCeleb metadata from raw sources.")

    def _artifacts_ready(self) -> bool:
        required_artifacts = [
            self.hparams.dataset.train_csv_file,
            self.hparams.dataset.val_csv_file,
            self.hparams.dataset.veri_test_output_path,
            self.hparams.dataset.speaker_lookup,
            str(self.enroll_csv_path),
            str(self.test_unique_csv_path),
        ]
        return all(Path(path).exists() for path in required_artifacts)
    
    def setup(self, stage: Optional[str] = None):
        """Instantiates the PyTorch Datasets."""
        if not self._artifacts_ready():
            self.prepare_data()

        if stage == 'fit' or stage is None:
            self.train_data = CNCelebDataset(
                data_dir=self.hparams.dataset.data_dir,
                data_filepath=self.hparams.dataset.train_csv_file,
                sample_rate=self.hparams.dataset.sample_rate,
                max_duration=self.hparams.dataset.max_duration,
                sep=self.hparams.dataset.get('sep', '|')
            )
            self.val_data = CNCelebDataset(
                data_dir=self.hparams.dataset.data_dir,
                data_filepath=self.hparams.dataset.val_csv_file,
                sample_rate=self.hparams.dataset.sample_rate,
                max_duration=self.hparams.dataset.max_duration,
                sep=self.hparams.dataset.get('sep', '|')
            )
        
        if stage == 'test' or stage is None:
            self.test_data = CNCelebVerificationDataset(
                data_dir=self.hparams.dataset.data_dir,
                data_filepath=self.hparams.dataset.veri_test_output_path,
                sample_rate=self.hparams.dataset.sample_rate,
                sep=self.hparams.dataset.get('sep', '|')
            )
            
            # Load unique enrollment and test data from CSV files
            enroll_df = pd.read_csv(self.enroll_csv_path, sep=self.hparams.dataset.get('sep', '|'))
            enroll_df['map_path'] = enroll_df['map_path'].apply(lambda x: x.split(';') if pd.notna(x) else [])
            test_unique_df = pd.read_csv(self.test_unique_csv_path, sep=self.hparams.dataset.get('sep', '|'))
            
            self.enrollment_data = CNCelebEnroll(
                data_dir=self.hparams.dataset.data_dir,
                sample_rate=self.hparams.dataset.sample_rate,
                df=enroll_df
            )
            self.test_unique_data = CNCelebTest(
                data_dir=self.hparams.dataset.data_dir,
                sample_rate=self.hparams.dataset.sample_rate,
                df=test_unique_df
            )

    def train_dataloader(self):
        return DataLoader(self.train_data, **self.hparams.loaders.train, collate_fn=TrainCollate())

    def val_dataloader(self):
        return DataLoader(self.val_data, **self.hparams.loaders.val, collate_fn=TrainCollate())

    def test_dataloader(self):
        return {
            'cnceleb': DataLoader(self.test_data, **self.hparams.loaders.test, collate_fn=VerificationCollate())
        }
    
    def get_enroll_and_trial_dataloaders(self, *args, **kwargs):
        """
        Return enrollment and test dataloaders for sv.py to process.
        This provides a standardized interface for embedding computation.
        
        Args:
            test_filename: Name of the test set (e.g., 'cnceleb')
            
        Returns:
            Tuple of (enrollment_dataloader, test_unique_dataloader)
        """
        if self.enrollment_data is None or self.test_unique_data is None:
            raise ValueError("Enrollment and test data not prepared. Call setup() first.")
            
        collate_fn_enroll = EnrollCollate()
        
        # Use enrollment loader config for consistency with VoxCeleb
        enrollment_dataloader = DataLoader(
            self.enrollment_data, 
            batch_size=self.hparams.loaders.get('enrollment', self.hparams.loaders.test).batch_size,
            shuffle=self.hparams.loaders.get('enrollment', self.hparams.loaders.test).shuffle,
            num_workers=self.hparams.loaders.get('enrollment', self.hparams.loaders.test).num_workers,
            pin_memory=self.hparams.loaders.get('enrollment', self.hparams.loaders.test).pin_memory,
            collate_fn=collate_fn_enroll
        )
        
        test_unique_dataloader = DataLoader(
            self.test_unique_data, 
            batch_size=self.hparams.loaders.get('enrollment', self.hparams.loaders.test).batch_size,
            shuffle=self.hparams.loaders.get('enrollment', self.hparams.loaders.test).shuffle,
            num_workers=self.hparams.loaders.get('enrollment', self.hparams.loaders.test).num_workers,
            pin_memory=self.hparams.loaders.get('enrollment', self.hparams.loaders.test).pin_memory,
            collate_fn=collate_fn_enroll
        )
        
        return enrollment_dataloader, test_unique_dataloader


if __name__ == "__main__":
    import pyrootutils
    import omegaconf
    root = pyrootutils.setup_root(search_from=__file__, indicator=[".env"], pythonpath=True, dotenv=True,)
    _HYDRA_PARAMS = {"version_base": "1.3",
                     "config_path": str(root / "configs"),
                     "config_name": "train.yaml"}

    @hydra.main(**_HYDRA_PARAMS)
    def test_datamodule(cfg):

        print("Running CNCelebDataModule test with the following config:")
        print(omegaconf.OmegaConf.to_yaml(cfg))

        # Instantiate the DataModule using the loaded config
        cnceleb_dm: "CNCelebDataModule" = hydra.utils.instantiate(cfg.datamodule)
        
        cnceleb_dm.prepare_data()
        cnceleb_dm.setup(stage='fit')
        
        train_loader = cnceleb_dm.train_dataloader()
        print(f"Train loader has {len(train_loader)} batches.")
        
        val_loader = cnceleb_dm.val_dataloader()
        print(f"Validation loader has {len(val_loader)} batches.")
        
        cnceleb_dm.setup(stage='test')
        test_loaders = cnceleb_dm.test_dataloader()
        print(f"Test returns {len(test_loaders)} loaders.")

        test_loaders['cnceleb'].dataset.__getitem__(0)

        print("Getting enrollment and unique test dataloaders...")
        enroll_loader, test_unique_loader = cnceleb_dm.get_enroll_and_trial_dataloaders()
        print(f"Enrollment loader has {len(enroll_loader)} batches.")
        print(f"Enrollment loader has {len(enroll_loader)} batches.")
    
    test_datamodule()