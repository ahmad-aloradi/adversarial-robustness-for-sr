from typing import Dict, List, Optional, Literal

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
from src.datamodules.components.voxceleb.voxceleb_prep import VoxCelebProcessor, VoxCelebTestFilter
from src import utils
from src.datamodules.components.utils import CsvProcessor
from src.datamodules.components.common import VoxcelebDefaults, get_dataset_class
from src import utils


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

    def prepare_data(self):
        if self.train_data is None:
            # Step 1: Generate all VoxCeleb metadata (no test speaker exclusion yet)
            voxceleb_processor = VoxCelebProcessor(
                root_dir=self.dataset.data_dir,
                verbose=self.dataset.verbose,
                artifcats_dir=self.dataset.voxceleb_artifacts_dir,
                sep=self.dataset.sep)
            
            _, _ = voxceleb_processor.generate_metadata(
                base_search_dir=self.dataset.base_search_dir,
                min_duration=self.dataset.min_duration,
                save_df=self.dataset.save_csv
                )

            # Step 2: Handle test speaker exclusion for all test sets
            test_filter = VoxCelebTestFilter(root_dir=self.dataset.data_dir, verbose=self.dataset.verbose)
            
            # Process all test sets and collect all test speakers
            all_test_speakers = set()
            self.test_dataframes = {}
            
            for test_filename in self.dataset.veri_test_filenames:
                test_speakers, veri_df = test_filter.get_test_speakers(test_filename)
                all_test_speakers.update(test_speakers)
                self.test_dataframes[test_filename] = veri_df
            
            if self.dataset.verbose:
                log.info(f"Total unique test speakers across all test sets: {len(all_test_speakers)}")
            
            # Load the generated dev metadata and filter out all test speakers
            dev_metadata = pd.read_csv(str(voxceleb_processor.dev_metadata_file), sep=self.dataset.sep)
            filtered_dev_metadata = test_filter.filter_dev_metadata(dev_metadata, all_test_speakers)

            # This is temporary setting and will be overwritten by the next save command
            # This is step is needed for compliance csv_processor. So not resetting the indices is ok
            VoxCelebProcessor.save_csv(filtered_dev_metadata, str(voxceleb_processor.dev_metadata_file), sep=self.dataset.sep)

            # Step 3: Get class id and speaker stats (df is loaded from filtered_dev_metadata!)
            updated_filtered_dev_metadata, speaker_lookup_csv = self.csv_processor.process(
                dataset_files=[str(voxceleb_processor.dev_metadata_file)],
                spks_metadata_paths=[self.dataset.metadata_csv_file],
                verbose=self.dataset.verbose)

            # save the updated csv
            VoxCelebProcessor.save_csv(updated_filtered_dev_metadata, str(voxceleb_processor.dev_metadata_file))
            VoxCelebProcessor.save_csv(speaker_lookup_csv, self.dataset.speaker_lookup)
            
            # Step 4: split the dataset into train and validation
            CsvProcessor.split_dataset(
                df=updated_filtered_dev_metadata,
                train_ratio = self.dataset.train_ratio,
                save_csv=self.dataset.save_csv,
                speaker_overlap=self.dataset.speaker_overlap,
                speaker_id_col=DATASET_CLS.SPEAKER_ID,
                train_csv=self.dataset.train_csv_file,
                val_csv=self.dataset.val_csv_file,
                sep=self.dataset.sep,
                seed=self.dataset.seed
                )
            
            # Step 5: enrich verification files for all test sets
            self.enroll_data_dict = {}
            self.unique_trial_data_dict = {}
            
            for test_filename in self.dataset.veri_test_filenames:
                veri_test_path = test_filter.download_test_file(test_filename)
                test_df = VoxCelebProcessor.enrich_verification_file(
                    veri_test_path=veri_test_path,
                    metadata_path=self.dataset.metadata_csv_file,
                    output_path=self.dataset.veri_test_output_paths[test_filename],
                    sep=self.dataset.sep,
                    )
                
                self.enroll_data_dict[test_filename] = VoxCelebDataModule._extract_enroll_test(test_df, mode='enroll')
                self.unique_trial_data_dict[test_filename] = VoxCelebDataModule._extract_enroll_test(test_df, mode='test')

    @staticmethod
    def _extract_enroll_test(df: pd.DataFrame, mode: Literal['enroll', 'test']):
        """
        Vectorized pandas implementation with speaker consistency validation.
        """
        path_col = f'{mode}_path'
        enroll_columns = [col for col in df.columns if col.startswith(f'{mode}_') and col != path_col]

        if not enroll_columns:
            return df[[path_col]].drop_duplicates().reset_index(drop=True)

        grouped = df.groupby(path_col)
        
        # Check which columns are constant within each group (nunique == 1)
        nunique_per_group = grouped[enroll_columns].nunique()
        is_constant = nunique_per_group == 1
        
        # Find paths where any column is not constant
        non_constant_mask = ~is_constant.all(axis=1)
        if non_constant_mask.any():
            problematic_paths = non_constant_mask[non_constant_mask].index.tolist()
            # Get details about which columns are inconsistent for the first problematic path
            first_path = problematic_paths[0]
            inconsistent_cols = nunique_per_group.loc[first_path][nunique_per_group.loc[first_path] > 1].index.tolist()
            
            raise ValueError(
                f"Inconsistent data found for {mode}_path '{first_path}'. "
                f"Expected all rows with the same path to have identical values, "
                f"but found multiple values in columns: {inconsistent_cols}. "
                f"This violates the assumption that same path = same speaker."
            )
        
        # Get the first value for each column in each group (all values are the same due to validation)
        results_df = grouped[enroll_columns].first().reset_index()
        
        return results_df

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
