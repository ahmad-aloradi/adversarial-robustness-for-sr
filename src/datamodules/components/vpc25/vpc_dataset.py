from pathlib import Path
from typing import List, Union, Dict
from dataclasses import dataclass

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from src.utils import get_pylogger
from src.datamodules.components.common import get_dataset_class, DatasetItem
from src.datamodules.components.utils import BaseCollate, BaseDataset, AudioProcessor

log = get_pylogger(__name__)

DATASET_CLS, DF_COLS = get_dataset_class('vpc25')


###############  dataclasses   #################
@dataclass
class VPC25Item:
    """Single item from dataset."""
    audio: torch.Tensor
    audio_length: int
    audio_path: str
    speaker_id: str
    recording_duration: float
    gender: str
    sample_rate: int
    model: int
    class_id: int = None
    country: str = None
    text: str = ''

@dataclass
class VPC25VerificationItem:
    """Single verification trial from VoxCeleb dataset."""
    enroll_id: int
    audio: torch.Tensor
    length: int
    trial_label: int
    audio_path: str
    gender: str
    recording_duration: float
    text: str
    model: str
    sample_rate: Union[int, float] = 16000.0

###############  Collate classes  #################

class VPC25ClassCollate(BaseCollate):
    def __call__(self, batch):
        out = super(VPC25ClassCollate, self).__call__(batch)
        out.audio_path = [str(item.audio_path) for item in batch]
        return out
    

class VPCTestCoallate(BaseCollate):
    """Collate function for verification pairs"""
    def __call__(self, batch) -> VPC25VerificationItem:
        enroll_ids, audios, paths, trial_labels, genders, texts, recording_durations, models, sample_rates = zip(
            *[(item.enroll_id, item.audio, item.audio_path, item.trial_label, 
               item.gender, item.text, item.recording_duration, item.model, item.sample_rate) for item in batch])
        
        # pad the audio sequences
        paths = [str(path) for path in paths]
        lengths = torch.tensor([wav.size(0) for wav in audios])
        padded_audio = pad_sequence(audios, batch_first=True, padding_value=self.pad_value)

        return VPC25VerificationItem(
            enroll_id=enroll_ids,
            audio=padded_audio,
            length=lengths,
            trial_label=trial_labels,
            audio_path=paths,
            gender=genders,
            text=texts,
            recording_duration=recording_durations,
            model=models,
            sample_rate=sample_rates
        )

class VPC25EnrollCollate(BaseCollate):
    def __call__(self, batch) -> VPC25Item:
        out = super(VPC25EnrollCollate, self).__call__(batch)
        out.speaker_id = [item.speaker_id for item in batch][0]
        out.gender = out.gender.squeeze(0)
        model = [item.model for item in batch][0]
        return VPC25Item(model=model, **out.__dict__)
    
###############  Dataset classes  #################
class VPC25Dataset(BaseDataset):
    pass


class VPC25TestDataset(Dataset):
    """
    A PyTorch Dataset class for the VoxCeleb verification task.
    Args:
        data_dir (str): Directory where the VoxCeleb dataset is stored.
        data_filepath (str): Path to the verification test file containing pairs.
        sample_rate (int, optional): Sample rate for audio processing. Defaults to 16000.
        max_duration (Union[None, float, int], optional): Maximum duration in seconds. 
            Use None for entire utterances. Defaults to None.
    """

    def __init__(
        self,
        data_dir: str,
        test_trials_path: str,
        sample_rate: int = 16000,
        max_duration: Union[None, float, int] = None,
        sep: str = "|",
    ):
        self.data_dir = Path(data_dir)
        self.df_trials = pd.read_csv(test_trials_path, sep=sep)
        self.audio_processor = AudioProcessor(sample_rate)
        
        if isinstance(max_duration, (int, float)):
            self.max_samples = int(max_duration * sample_rate)
        elif max_duration is None:
            self.max_samples = -1
        else:
            raise ValueError("max_duration must be an int, float, or None")

    def __len__(self):
        return len(self.df_trials)

    def __getitem__(self, idx) -> VPC25VerificationItem:
        # Get trial row
        row = self.df_trials.iloc[idx]

        # Load and process both utterances
        rel_filepath = self.data_dir / row.rel_filepath
        assert rel_filepath.is_file(), f'File {rel_filepath} does not exist'
        test_audio, _ = self.audio_processor.process_audio(rel_filepath)

        return VPC25VerificationItem(
            enroll_id=row.enrollment_id,
            audio=test_audio,
            length=test_audio.shape[0],
            trial_label=row.label,
            audio_path=rel_filepath,
            gender=row.gender,
            sample_rate=self.audio_processor.sample_rate,
            text=row.text,
            recording_duration=row.recording_duration,
            model=row.model
        )

class VPC25EnrollDataset(BaseDataset):
    def __getitem__(self, idx):
        dataset_item = super(VPC25EnrollDataset, self).__getitem__(idx)
        row = self.dataset.iloc[idx]
        return VPC25Item(
            model=row.model,
            **dataset_item.__dict__
        )