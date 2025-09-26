from typing import Dict, List, Optional
from pathlib import Path
import shutil

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

from src.datamodules.components.cnceleb.cnceleb_prep import CNCelebProcessor
from src import utils
from src.datamodules.components.utils import CsvProcessor
from src.datamodules.components.common import CNCelebDefaults, get_dataset_class

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
        artifacts_dir = Path(self.hparams.dataset.artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Check for base_search_dir and copy core files if available
        base_search_dir = Path(self.hparams.dataset.get('base_search_dir', ''))
        
        # Define core files that can be pre-generated and copied
        core_files_to_copy = [
            ('cnceleb_dev.csv', self.hparams.dataset.dev_metadata_file),
            ('enroll.csv', self.hparams.dataset.enroll_csv_path),
            ('test_unique.csv', self.hparams.dataset.test_unique_csv_path),
            ('verification_trials.csv', self.hparams.dataset.veri_test_output_path),
            ('dev_speakers.txt', self.hparams.dataset.dev_spk_file),
            ('test_speakers.txt', self.hparams.dataset.test_spk_file),
        ]
        
        # Check if core pre-generated files exist
        core_files_exist = base_search_dir.exists() and all(
            (base_search_dir / source_file).exists() for source_file, _ in core_files_to_copy
        )
        
        if core_files_exist:
            log.info(f"Found core pre-generated files in {base_search_dir}. Copying to experiment directory...")
            for source_file, target_path in core_files_to_copy:
                source_path = base_search_dir / source_file
                target_path = Path(target_path)
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, target_path)
                log.info(f"Copied {source_path} -> {target_path}")
            
            # Load the copied dev metadata for further processing
            dev_df = pd.read_csv(self.hparams.dataset.dev_metadata_file, sep=self.hparams.dataset.sep)
            log.info(f"Loaded pre-generated dev metadata with {len(dev_df)} rows")
        else:
            # Generate core metadata from scratch
            log.info("Pre-generated core files not found. Generating metadata from scratch...")
            
            # Step 1: Instantiate the processor with specific artifact paths
            processor = CNCelebProcessor(
                root_dir=self.hparams.dataset.data_dir,
                artifacts_dir=self.hparams.dataset.artifacts_dir,
                cnceleb1=self.hparams.dataset.cnceleb1,
                dev_metadata_file=self.hparams.dataset.dev_metadata_file,
                enroll_csv_path=self.hparams.dataset.enroll_csv_path,
                test_unique_csv_path=self.hparams.dataset.test_unique_csv_path,
                dev_spk_file=self.hparams.dataset.dev_spk_file,
                test_spk_file=self.hparams.dataset.test_spk_file,
                cnceleb2=self.hparams.dataset.get('cnceleb2', None),
                verbose=self.hparams.dataset.verbose,
                sep=self.hparams.dataset.sep,
            )

            # Step 2: Generate trial and enrollment lists to identify test/enroll speakers
            trials_df = processor.generate_trial_list()
            trials_df.to_csv(self.hparams.dataset.veri_test_output_path, sep=self.hparams.dataset.get('sep', '|'), index=False)
            log.info(f"Saved verification trials to {self.hparams.dataset.veri_test_output_path}")
            enroll_csv_path, test_unique_csv_path = processor.generate_unique_embeddings_csvs(trials_df)
            if not (Path(test_unique_csv_path).exists() and Path(enroll_csv_path).exists()):
                raise FileNotFoundError("Failed to generate unique test or enrollment CSV files.")

            # Step 3: Generate enrollment and development metadata for the entire dataset
            enroll_df = processor.generate_enrollment_embeddings_list()
            dev_df = processor.generate_metadata()

            # Exclude enrollment speakers from the development dataframe
            # TODO: Although we do not loop over the eval dir during preparation, the dev directory somehow still contains enrollment speakers
            # We need to investigate this further.
            enroll_speakers = set(enroll_df['enroll_id'].str.split('-').str[0])
            test_speakers = set(pd.read_csv(processor.test_spk_file, header=None)[0])
            dev_speakers = set(pd.read_csv(processor.dev_spk_file, header=None)[0])

            original_dev_rows = len(dev_df)
            dev_df = dev_df[~dev_df['speaker_id'].isin(test_speakers)]
            log.info(f"Excluded {original_dev_rows - len(dev_df)} utterances belonging to enrollment speakers.")

            assert not dev_speakers.intersection(enroll_speakers), "Dev and enrollment speakers should not overlap!"
            assert test_speakers.intersection(enroll_speakers), "Enrollment and test speakers should overlap!"
            assert not test_speakers.intersection(dev_speakers), "Test and dev speakers should not overlap!"

            # Save the metadata file with the filtered dataframe
            dev_df.to_csv(processor.dev_metadata_file, sep=self.hparams.dataset.sep, index=False)

        # Always generate derived files (speaker_lookup, train/val splits) from the dev metadata
        log.info("Generating derived files (speaker_lookup, train/val splits)...")
        
        # Step 4: Generate speaker metadata from the dev_df and order them by total duration
        # Note: We need a processor instance for this, so create one if we copied files
        if core_files_exist:
            processor = CNCelebProcessor(
                root_dir=self.hparams.dataset.data_dir,
                artifacts_dir=self.hparams.dataset.artifacts_dir,
                cnceleb1=self.hparams.dataset.cnceleb1,
                dev_metadata_file=self.hparams.dataset.dev_metadata_file,
                enroll_csv_path=self.hparams.dataset.enroll_csv_path,
                test_unique_csv_path=self.hparams.dataset.test_unique_csv_path,
                dev_spk_file=self.hparams.dataset.dev_spk_file,
                test_spk_file=self.hparams.dataset.test_spk_file,
                cnceleb2=self.hparams.dataset.get('cnceleb2', None),
                verbose=self.hparams.dataset.verbose,
                sep=self.hparams.dataset.sep,
            )
        
        speaker_lookup_df = processor.generate_speaker_metadata(dev_df)
        speaker_lookup_df.to_csv(self.hparams.dataset.speaker_lookup, sep=self.hparams.dataset.sep, index=False)

        # Step 5: Add metadata class IDs and speaker stats
        dev_df, speaker_lookup_df = self.csv_processor.process(
            dataset_files=[str(processor.dev_metadata_file)],
            spks_metadata_paths=[self.hparams.dataset.speaker_lookup],
            verbose=self.hparams.dataset.verbose
            )

        # Overwrite the metadata file with the filtered dataframe
        dev_df.to_csv(processor.dev_metadata_file, sep=self.hparams.dataset.sep, index=False)
        log.info(f"Saved metadata to {processor.dev_metadata_file}")        
        # Save speaker_lookup_df
        speaker_lookup_df.to_csv(self.hparams.dataset.speaker_lookup, sep=self.hparams.dataset.sep, index=False)
        log.info(f"Saved speaker lookup to {self.hparams.dataset.speaker_lookup}")        

        # Step 6: Split dataset into train and validation 
        CsvProcessor.split_dataset(
            df=dev_df,
            train_ratio=self.hparams.dataset.train_ratio,
            save_csv=True,
            speaker_overlap=self.hparams.dataset.speaker_overlap,
            speaker_id_col=DATASET_CLS.SPEAKER_ID,
            train_csv=self.hparams.dataset.train_csv_file,
            val_csv=self.hparams.dataset.val_csv_file,
            sep=self.hparams.dataset.sep,
            seed=self.hparams.dataset.seed
        )
        log.info(f"Saved train and val csvs to {self.hparams.dataset.train_csv_file} and {self.hparams.dataset.val_csv_file}")

        # Step 7: Validate the splits
        log.info("CNCeleb preparation summary:")
        log.info(f"  Final metadata rows: {len(dev_df)}")
        if not core_files_exist:
            trials_df = pd.read_csv(self.hparams.dataset.veri_test_output_path, sep=self.hparams.dataset.sep)
            enroll_df = pd.read_csv(self.hparams.dataset.enroll_csv_path, sep=self.hparams.dataset.sep)
            log.info(f"  Trials rows: {len(trials_df)}")
            log.info(f"  Enrollment rows: {len(enroll_df)}")
        else:
            log.info("  (Core files were copied from pre-generated location)")
    
    def setup(self, stage: Optional[str] = None):
        """Instantiates the PyTorch Datasets."""
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