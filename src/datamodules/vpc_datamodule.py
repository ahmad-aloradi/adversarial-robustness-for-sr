from pathlib import Path
from typing import Optional, Dict, Union
import os
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import pandas as pd

from src.datamodules.components.common import LibriSpeechDefaults, get_dataset_class
from src.datamodules.components.vpc25.vpc_dataset import (VPC25Dataset, VPC25TestDataset, VPC25ClassCollate,
                                                          VPCTestCoallate,VPC25EnrollCollate, VPC25EnrollDataset)
from src.datamodules.components.librispeech.librispeech_prep import write_dataset_csv
from src.datamodules.components.utils import CsvProcessor #,  SpeakerBatchSampler
from src import utils

log = utils.get_pylogger(__name__)

DATASET_DEFAULTS = LibriSpeechDefaults()
DATASET_CLS, DF_COLS = get_dataset_class(DATASET_DEFAULTS.dataset_name)

class AnonLibriDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for anonymized LibriSpeech-like data."""

    def __init__(
        self,
        root_dir: Union[str, Path],
        dataset: Dict[str, Dict[str, str]],
        loaders: Dict[str, Dict[str, int]],
        models: Dict[str, Dict[str, str]],
        transform=None,
        trainer_config: Optional[str] = None,
        *args, **kwargs
    ):
        """
        Initialize the DataModule.
        
        Args:
            root_dir: Root directory containing all data
            subset_dirs: List of subset directories to include
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            transform: Optional transform to be applied to audio
        """
        super().__init__()
        self.train_data = None
        self.eval_data = None
        self.test_data = None
        verbose = True
        
        # Check if using DDP trainer
        self.is_ddp = trainer_config is not None and ("ddp" in trainer_config or "ddp_find_unused_parameters_true" in trainer_config)

        self.root_dir = Path(root_dir)
        self.dataset = dataset
        self.sample_rate = dataset.get('sample_rate', DATASET_DEFAULTS.sample_rate)
        self.loaders = loaders
        self.models = models
        self.transform = transform
        self.save_paths = kwargs.get('artifacts_paths', None)
        self.csv_processor = CsvProcessor(verbose=verbose, fill_value='N/A')

    def prepare_data(self):
        if self.train_data is None:
            df_train, df_speakers_train  = self.csv_processor.process(
                [self.models[dire].train for dire in self.models.keys()], [self.dataset.speakers_file])
            df_dev, df_speakers_dev  = self.csv_processor.process(
                [self.models[dire].dev for dire in self.models.keys()], [self.dataset.speakers_file])
            df_test, df_speakers_test  = self.csv_processor.process(
                [self.models[dire].test for dire in self.models.keys()], [self.dataset.speakers_file])

            # Handle enrolls and trials seprately
            df_dev_enrolls = AnonLibriDataModule.concatenate_dfs([self.models[dire].dev_enrolls for dire in self.models.keys()])
            df_dev_trials = AnonLibriDataModule.concatenate_dfs([self.models[dire].dev_trials for dire in self.models.keys()])
            test_enrolls = AnonLibriDataModule.concatenate_dfs([self.models[dire].test_enrolls for dire in self.models.keys()])
            df_test_trials = AnonLibriDataModule.concatenate_dfs([self.models[dire].test_trials for dire in self.models.keys()])
            
            # Post-process the dfs
            df_dev_trials.enrollment_id = df_dev_trials.enrollment_id.apply(lambda x: DATASET_DEFAULTS.dataset_name + '_' + str(x))
            df_test_trials.enrollment_id = df_test_trials.enrollment_id.apply(lambda x: DATASET_DEFAULTS.dataset_name + '_' + str(x))
            df_dev_enrolls = df_dev_enrolls.rename(columns={'enrollment_path': 'rel_filepath'})
            test_enrolls = test_enrolls.rename(columns={'enrollment_path': 'rel_filepath'})

            df_dev_enrolls = df_dev_enrolls.merge(
                df_dev[[col for col in df_dev.columns if col not in ['speaker_id', 'split']]], 
                on='rel_filepath', 
                how='left'
                )
            test_enrolls = test_enrolls.merge(
                df_test[[col for col in df_test.columns if col not in ['speaker_id', 'split']]], 
                on='rel_filepath', 
                how='left'
                )

            df_dev_trials = df_dev_trials.merge(
                df_dev[[col for col in df_dev.columns if col not in ['model',  'gender',  'class_id']]], 
                left_on='test_path',
                right_on='rel_filepath',
                how='left'
                ).drop('test_path', axis=1)

            df_test_trials = df_test_trials.merge(
                df_test[[col for col in df_test.columns if col not in ['model',  'gender',  'class_id']]],
                left_on='test_path', 
                right_on='rel_filepath',
                how='left'
                ).drop('test_path', axis=1)

            # exlcude df_test_enrolls from df_test
            df_test_unique = df_test[~df_test.rel_filepath.isin(test_enrolls.rel_filepath)]

            # save the updated csv
            os.makedirs(self.dataset.artifacts_dir, exist_ok=True)

            for df, path in zip(
                [df_train, df_dev, df_test, df_test_unique,
                df_dev_enrolls, test_enrolls,
                df_dev_trials, df_test_trials,
                df_speakers_train, df_speakers_dev, df_speakers_test],
                [self.save_paths.train, self.save_paths.dev, self.save_paths.test, self.save_paths.test_unique,
                self.save_paths.dev_enrolls, self.save_paths.test_enrolls, 
                self.save_paths.dev_trials, self.save_paths.test_trials, 
                self.save_paths.spks_train, self.save_paths.spks_dev, self.save_paths.spks_test]
            ):
                write_dataset_csv(df, path, sep=self.dataset.sep)

    @staticmethod
    def concatenate_dfs(csv_paths, fill_value='N/A', sep='|') -> pd.DataFrame:
        # Read all CSVs and store their DataFrames
        dfs = []
        for path in csv_paths:
            dfs.append(pd.read_csv(path, sep=sep))        
        combined_df = pd.concat(dfs, ignore_index=True)        
        return combined_df.fillna(fill_value)

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_data = VPC25Dataset(
                data_dir=self.root_dir,
                data_filepath=self.save_paths.train,
                sample_rate=self.sample_rate,
                max_duration=self.dataset.max_duration
            )
            self.eval_data = VPC25Dataset(
                data_dir=self.root_dir,
                data_filepath=self.save_paths.dev,
                sample_rate=self.sample_rate,
                max_duration=self.dataset.max_duration
            )
        if stage == 'test' or stage is None:
            self.test_data = VPC25TestDataset(data_dir=self.root_dir,
                                              test_trials_path=self.save_paths.test_trials,
                                              sample_rate=self.sample_rate,
                                              max_duration=self.dataset.max_duration)

            self.test_unique = VPC25Dataset(data_dir=self.root_dir,
                                            data_filepath=self.save_paths.test_unique,
                                            sample_rate=self.sample_rate,
                                            max_duration=self.dataset.max_duration)

            self.enroll_data = VPC25EnrollDataset(data_dir=self.root_dir,
                                                  data_filepath=self.save_paths.test_enrolls,
                                                  sample_rate=self.sample_rate,
                                                  max_duration=None)
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data,
            batch_size=self.loaders.train.batch_size,
            shuffle=self.loaders.train.shuffle,
            num_workers=self.loaders.train.num_workers,
            pin_memory=self.loaders.train.pin_memory,
            collate_fn=VPC25ClassCollate()
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.eval_data,
            batch_size=self.loaders.valid.batch_size,
            shuffle=self.loaders.valid.shuffle,
            num_workers=self.loaders.valid.num_workers,
            pin_memory=self.loaders.valid.pin_memory,
            collate_fn=VPC25ClassCollate()
        )
    
    def test_dataloader(self) -> DataLoader:
        trial_loader = DataLoader(
            self.test_data,
            batch_size=self.loaders.test.batch_size,
            shuffle=self.loaders.test.shuffle,
            num_workers=self.loaders.test.num_workers,
            pin_memory=self.loaders.test.pin_memory,
            collate_fn=VPCTestCoallate()
        )

        trial_unique = DataLoader(
            self.test_unique,
            batch_size=self.loaders.test.batch_size,
            shuffle=self.loaders.test.shuffle,
            num_workers=self.loaders.test.num_workers,
            pin_memory=self.loaders.test.pin_memory,
            collate_fn=VPC25ClassCollate()
            )

        enroll_loader = DataLoader(
            self.enroll_data,
            batch_size=self.loaders.enrollment.batch_size,
            shuffle=self.loaders.enrollment.shuffle,
            num_workers=self.loaders.enrollment.num_workers,
            pin_memory=self.loaders.enrollment.pin_memory,
            collate_fn=VPC25EnrollCollate()
            )
        return trial_loader, enroll_loader, trial_unique


if __name__ == "__main__":
    import pyrootutils
    import hydra
    root = pyrootutils.setup_root(search_from=__file__, indicator=[".env"], pythonpath=True, dotenv=True,)
    _HYDRA_PARAMS = {"version_base": "1.3", "config_path": str(root / "configs"), "config_name": "train.yaml"}

    @hydra.main(**_HYDRA_PARAMS)
    def test_datamodule(cfg):

        print("Starting Librispeech DataModule test...")
        datamodule: AnonLibriDataModule = hydra.utils.instantiate(cfg.datamodule, _recursive_=False)
        datamodule.prepare_data()

        assert not datamodule.train_data
        assert not datamodule.eval_data
        assert not datamodule.test_data

        datamodule.setup()
        assert datamodule.train_data
        assert datamodule.eval_data
        assert datamodule.test_data

        test_loader, enroll_loader = datamodule.test_dataloader()
        enroll_batch = next(iter(enroll_loader))
        test_batch = next(iter(test_loader))
        print(enroll_batch)
        print(test_batch)

        train_batch = next(iter(datamodule.train_dataloader()))
        dev_batch = next(iter(datamodule.val_dataloader()))
        # print(train_batch)
        print(dev_batch)

    test_datamodule()
