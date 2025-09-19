from typing import Union, Literal
from pathlib import Path
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd

from src.datamodules.components.utils import AudioProcessor, BaseCollate, BaseDataset
from src.datamodules.components.common import get_dataset_class, CNCelebDefaults, DatasetItem
from src import utils

log = utils.get_pylogger(__name__)

DATASET_DEFAULTS = CNCelebDefaults()
DATASET_CLS, DF_COLS = get_dataset_class(DATASET_DEFAULTS.dataset_name)


@dataclass
class CNCelebVerificationItem:
    """Single verification trial from CNCeleb dataset."""
    test_audio: torch.Tensor
    test_length: int
    trial_label: int
    enroll_id: str  # Changed from enroll_path to enroll_id
    test_path: str
    sample_rate: int

@dataclass
class CNCelebItem(DatasetItem):
    """Single item from dataset."""
    sample_rate: float = 16000.0


@dataclass
class CNCelebEnrollItem:
    """Single enrollment item from CNCeleb dataset."""
    audio: torch.Tensor
    audio_length: int
    enroll_id: str
    enroll_path: str
    sample_rate: int

####### Coallate functions #######
class TrainCollate(BaseCollate):
    pass


class VerificationCollate(BaseCollate):
    """Collate function for verification pairs"""
    def __call__(self, batch) -> dict:
        test_wavs, labels, enroll_ids, test_paths, sample_rates = zip(
            *[(item.test_audio, item.trial_label, item.enroll_id,
               item.test_path, item.sample_rate) for item in batch])
                
        test_lengths = torch.tensor([wav.size(0) for wav in test_wavs])
        padded_test_wavs = pad_sequence(test_wavs, batch_first=True, padding_value=self.pad_value)
        
        return {
            'test_audio': padded_test_wavs,
            'test_length': test_lengths,
            'trial_label': torch.tensor(labels),
            'enroll_id': enroll_ids,
            'test_path': test_paths,
            'sample_rate': sample_rates[0]
        }


class EnrollCollate(BaseCollate):
    """Collate function for enrollment data"""
    def __init__(self, pad_value=0):
        super().__init__(pad_value)

    def __call__(self, batch) -> dict:
        audios, lengths, enroll_ids, enroll_paths, sample_rates = zip(
            *[(item.audio, item.audio_length, item.enroll_id, 
               item.enroll_path, item.sample_rate) for item in batch])
        
        padded_audios = pad_sequence(audios, batch_first=True, padding_value=self.pad_value)
        
        return {
            'audio': padded_audios,
            'audio_length': torch.tensor(lengths),
            'enroll_id': enroll_ids,
            'enroll_path': enroll_paths,
            'sample_rate': sample_rates[0]
        }
    
####### Datasets #######

class CNCelebDataset(BaseDataset):
    def __init__(self, *args, transforms=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.transforms = transforms

    def __getitem__(self, index: int):
        item = super().__getitem__(index)
        if self.transforms:
            item.audio = self.transforms(item.audio.unsqueeze(0)).squeeze(0)
        return item


class CNCelebVerificationDataset(Dataset):
    """
    A PyTorch Dataset class for the CNCeleb verification task.
    
    This dataset loads pre-computed enrollment embeddings and test audio.
    For verification trials, it expects enrollment embeddings to be pre-computed
    and aggregated by enrollment ID.
    """
    
    def __init__(
        self,
        data_dir: str,
        data_filepath: str,
        sample_rate: int = 16000,
        max_duration: Union[None, float, int] = None,
        sep: str = "|",
    ):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.audio_processor = AudioProcessor(sample_rate)
        
        # Load verification trials
        self.trials_df = pd.read_csv(data_filepath, sep=sep)
        
        # Check if the trials file is empty
        if len(self.trials_df) == 0:
            raise ValueError(f"Verification trials file is empty: {data_filepath}")
        
        # Validate required columns exist
        required_cols = ['label', 'enroll_id', 'test_path']
        missing_cols = [col for col in required_cols if col not in self.trials_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in trials file: {missing_cols}. Found: {list(self.trials_df.columns)}")
        
        log.info(f"Loaded {len(self.trials_df)} verification trials")
        
        if isinstance(max_duration, (int, float)):
            self.max_samples = int(max_duration * sample_rate)
        elif max_duration is None:
            self.max_samples = -1
        else:
            raise ValueError("max_duration must be an int, float, or None")

    def __len__(self):
        return len(self.trials_df)

    def __getitem__(self, idx) -> CNCelebVerificationItem:
        trial = self.trials_df.iloc[idx]
        
        # Load test audio
        test_audio_path = self.data_dir / trial['test_path']
        test_audio, _ = self.audio_processor.process_audio(str(test_audio_path))        
        
        return CNCelebVerificationItem(
            test_audio=test_audio,
            test_length=len(test_audio),
            trial_label=trial['label'],
            enroll_id= trial['enroll_id'],
            test_path=trial['test_path'],
            sample_rate=self.sample_rate
        )


class CNCelebEnroll(Dataset):    
    """
    Dataset for loading enrollment files for embedding extraction.
    This is used to pre-compute enrollment embeddings.
    """
    def __init__(
        self,
        data_dir: str,
        df: pd.DataFrame,
        sample_rate: int = 16000,
        max_duration: Union[None, float, int] = None,
    ):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.audio_processor = AudioProcessor(sample_rate)
        
        # Load enrollment mappings
        self.enroll_df = df
        log.info(f"Loaded {len(self.enroll_df)} enrollment file mappings")
        
    def __len__(self):
        return len(self.enroll_df)

    def __getitem__(self, idx) -> CNCelebEnrollItem:
        row = self.enroll_df.iloc[idx]
        enroll_id = row['enroll_id']
        enroll_path = row['enroll_path']
        
        # Load enrollment audio
        audio_path = self.data_dir / enroll_path
        audio, _ = self.audio_processor.process_audio(str(audio_path))
        
        return CNCelebEnrollItem(
            audio=audio,
            audio_length=len(audio),
            enroll_id=enroll_id,
            enroll_path=enroll_path,
            sample_rate=self.sample_rate
        )


class CNCelebTest(Dataset):
    """
    Dataset for loading test files for embedding extraction.
    """
    def __init__(
        self,
        data_dir: str,
        df: pd.DataFrame,
        sample_rate: int = 16000,
        max_duration: Union[None, float, int] = None,
    ):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.audio_processor = AudioProcessor(sample_rate)
        
        # Load test file paths
        self.test_df = df
        log.info(f"Loaded {len(self.test_df)} test file paths")
        
    def __len__(self):
        return len(self.test_df)

    def __getitem__(self, idx) -> dict:
        row = self.test_df.iloc[idx]
        test_path = row['test_path']
        
        # Load test audio
        audio_path = self.data_dir / test_path
        audio, _ = self.audio_processor.process_audio(str(audio_path))
        
        return {
            'audio': audio,
            'audio_length': len(audio),
            'test_path': test_path,
            'sample_rate': self.sample_rate
        }
