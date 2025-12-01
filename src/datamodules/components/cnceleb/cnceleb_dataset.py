from typing import Dict, Literal, Optional, Tuple, Union
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
class CNCelebVerificationSample:
    """Single verification trial sample prior to collation."""
    enroll_audio: torch.Tensor
    test_audio: torch.Tensor
    enroll_length: int
    test_length: int
    trial_label: int
    same_gender_label: Optional[int]
    same_country_label: Optional[int]
    enroll_path: str
    test_path: str
    sample_rate: int
    enroll_id: Optional[str] = None


@dataclass
class CNCelebVerificationBatch:
    """Batched verification data aligned with VoxCeleb semantics."""
    enroll_audio: torch.Tensor
    test_audio: torch.Tensor
    enroll_length: torch.Tensor
    test_length: torch.Tensor
    trial_label: Tuple[int, ...]
    same_gender_label: Tuple[Optional[int], ...]
    same_country_label: Tuple[Optional[int], ...]
    enroll_path: Tuple[str, ...]
    test_path: Tuple[str, ...]
    sample_rate: int
    enroll_id: Tuple[Optional[str], ...]


@dataclass
class CNCelebItem(DatasetItem):
    """Single item from dataset."""
    sample_rate: float = 16000.0


@dataclass
class CNCelebEnrollSample:
    """Single enrollment sample before collation."""
    audio: torch.Tensor
    audio_length: int
    enroll_id: str
    enroll_path: str
    sample_rate: int


@dataclass
class CNCelebEnrollBatch:
    """Batched enrollment data consumed by the model."""
    audio: torch.Tensor
    audio_length: torch.Tensor
    audio_path: Tuple[str, ...]
    enroll_id: Tuple[str, ...]
    sample_rate: int


@dataclass
class CNCelebTestSample:
    """Single test item from CNCeleb dataset."""
    audio: torch.Tensor
    audio_length: int
    audio_path: str
    sample_rate: int


@dataclass
class CNCelebTestBatch:
    """Batched unique test utterances for embedding computation."""
    audio: torch.Tensor
    audio_length: torch.Tensor
    audio_path: Tuple[str, ...]
    sample_rate: int


####### Coallate functions #######
class TrainCollate(BaseCollate):
    pass


class VerificationCollate(BaseCollate):
    """Collate function for verification pairs."""

    def __call__(self, batch) -> CNCelebVerificationBatch:
        if not batch:
            raise ValueError("CNCeleb verification batch is empty.")

        enroll_wavs = [item.enroll_audio for item in batch]
        test_wavs = [item.test_audio for item in batch]
        enroll_lengths = torch.tensor([item.enroll_length for item in batch], dtype=torch.long)
        test_lengths = torch.tensor([item.test_length for item in batch], dtype=torch.long)

        padded_enroll_wavs = pad_sequence(enroll_wavs, batch_first=True, padding_value=self.pad_value)
        padded_test_wavs = pad_sequence(test_wavs, batch_first=True, padding_value=self.pad_value)

        trial_labels = tuple(int(item.trial_label) for item in batch)
        same_gender_labels = tuple(item.same_gender_label for item in batch)
        same_country_labels = tuple(item.same_country_label for item in batch)
        enroll_paths = tuple(item.enroll_path for item in batch)
        test_paths = tuple(item.test_path for item in batch)
        enroll_ids = tuple(item.enroll_id for item in batch)
        sample_rate = batch[0].sample_rate

        return CNCelebVerificationBatch(
            enroll_audio=padded_enroll_wavs,
            test_audio=padded_test_wavs,
            enroll_length=enroll_lengths,
            test_length=test_lengths,
            trial_label=trial_labels,
            same_gender_label=same_gender_labels,
            same_country_label=same_country_labels,
            enroll_path=enroll_paths,
            test_path=test_paths,
            sample_rate=sample_rate,
            enroll_id=enroll_ids,
        )


class EnrollCollate(BaseCollate):
    """Collate function for enrollment data"""
    def __init__(self, pad_value=0):
        super().__init__(pad_value)

    def __call__(self, batch) -> CNCelebEnrollBatch:
        if not batch:
            raise ValueError("CNCeleb enrollment batch is empty.")

        audios = [item.audio for item in batch]
        lengths = torch.tensor([item.audio_length for item in batch], dtype=torch.long)
        padded_audios = pad_sequence(audios, batch_first=True, padding_value=self.pad_value)

        return CNCelebEnrollBatch(
            audio=padded_audios,
            audio_length=lengths,
            audio_path=tuple(item.enroll_path for item in batch),
            enroll_id=tuple(item.enroll_id for item in batch),
            sample_rate=batch[0].sample_rate,
        )


class TestCollate(BaseCollate):
    """Collate function for test data"""
    def __init__(self, pad_value=0):
        super().__init__(pad_value)

    def __call__(self, batch) -> CNCelebTestBatch:
        if not batch:
            raise ValueError("CNCeleb test batch is empty.")

        audios = [item.audio for item in batch]
        lengths = torch.tensor([item.audio_length for item in batch], dtype=torch.long)
        padded_audios = pad_sequence(audios, batch_first=True, padding_value=self.pad_value)

        return CNCelebTestBatch(
            audio=padded_audios,
            audio_length=lengths,
            audio_path=tuple(item.audio_path for item in batch),
            sample_rate=batch[0].sample_rate,
        )


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
        enroll_lookup: Optional[Union[pd.DataFrame, Dict[str, str]]] = None,
    ):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.audio_processor = AudioProcessor(sample_rate)

        # Load verification trials
        self.trials_df = pd.read_csv(data_filepath, sep=sep)
        self.enroll_lookup: Dict[str, str] = {}
        if enroll_lookup is not None:
            if isinstance(enroll_lookup, pd.DataFrame):
                required_cols = {"enroll_id", "enroll_path"}
                if not required_cols.issubset(enroll_lookup.columns):
                    raise ValueError(
                        f"Enrollment lookup data frame must contain {required_cols}, got {list(enroll_lookup.columns)}"
                    )
                self.enroll_lookup = {
                    str(row.enroll_id): str(row.enroll_path)
                    for _, row in enroll_lookup[list(required_cols)].iterrows()
                }
            elif isinstance(enroll_lookup, dict):
                self.enroll_lookup = {str(k): str(v) for k, v in enroll_lookup.items()}
            else:
                raise TypeError(
                    "enroll_lookup must be either a pandas.DataFrame or a mapping of enroll_id to enroll_path."
                )
        
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

    def __getitem__(self, idx) -> CNCelebVerificationSample:
        trial = self.trials_df.iloc[idx]

        enroll_id = str(trial['enroll_id']) if 'enroll_id' in trial else None
        if enroll_id is None:
            raise KeyError("Trial row is missing 'enroll_id'.")

        enroll_rel_path = self.enroll_lookup.get(enroll_id)
        if enroll_rel_path is None:
            raise KeyError(f"Missing enrollment path for enroll_id '{enroll_id}'.")

        enroll_audio_path = self.data_dir / enroll_rel_path
        enroll_audio, _ = self.audio_processor.process_audio(str(enroll_audio_path))

        enroll_length = int(enroll_audio.shape[0])
        
        # Load test audio
        test_audio_path = self.data_dir / trial['test_path']
        test_audio, _ = self.audio_processor.process_audio(str(test_audio_path))        

        test_length = int(test_audio.shape[0])

        def _sanitize_optional(value: Optional[Union[int, float]]) -> Optional[int]:
            if value is None or pd.isna(value):
                return None
            return int(value)

        same_gender = _sanitize_optional(trial.get('same_gender')) if 'same_gender' in trial else None
        same_country = _sanitize_optional(trial.get('same_country')) if 'same_country' in trial else None

        return CNCelebVerificationSample(
            enroll_audio=enroll_audio,
            test_audio=test_audio,
            enroll_length=enroll_length,
            test_length=test_length,
            trial_label=int(trial['label']),
            same_gender_label=same_gender,
            same_country_label=same_country,
            enroll_path=enroll_rel_path,
            test_path=trial['test_path'],
            sample_rate=self.sample_rate,
            enroll_id=enroll_id,
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

    def __getitem__(self, idx) -> CNCelebEnrollSample:
        row = self.enroll_df.iloc[idx]
        enroll_id = row['enroll_id']
        enroll_path = row['enroll_path']
        
        # Load enrollment audio
        audio_path = self.data_dir / enroll_path
        audio, _ = self.audio_processor.process_audio(str(audio_path))
        
        return CNCelebEnrollSample(
            audio=audio,
            audio_length=len(audio),
            enroll_id=enroll_id,
            enroll_path=enroll_path,
            sample_rate=self.sample_rate,
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

    def __getitem__(self, idx) -> CNCelebTestSample:
        row = self.test_df.iloc[idx]
        test_path = row['test_path']
        
        # Load test audio
        audio_path = self.data_dir / test_path
        audio, _ = self.audio_processor.process_audio(str(audio_path))

        return CNCelebTestSample(
            audio=audio,
            audio_length=len(audio),
            audio_path=test_path,
            sample_rate=self.sample_rate,
        )
