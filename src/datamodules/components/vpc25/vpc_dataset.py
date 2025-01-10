from pathlib import Path
from typing import List, Union, Dict
from dataclasses import dataclass

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from src.utils import get_pylogger
from src.datamodules.components.common import get_dataset_class
from src.datamodules.components.utils import BaseCollate, BaseDataset, AudioProcessor

log = get_pylogger(__name__)

DATASET_CLS, DF_COLS = get_dataset_class('vpc25')


###############  dataclasses   #################
@dataclass
class VPC25Item():
    """Single item from dataset."""
    sample_rate: float = 16000.0

@dataclass
class VPC25VerificationItem:
    """Single verification trial from VoxCeleb dataset."""
    enroll_id: int
    test_audio: torch.Tensor
    test_length: int
    trial_label: int
    test_path: str
    gender: str
    sample_rate: int = 16000.0

###############  Collate classes  #################

class VPC25ClassCollate(BaseCollate):
    pass

class VPCTestCoallate(BaseCollate):
    """Collate function for verification pairs"""
    def __call__(self, batch) -> VPC25VerificationItem:
        enroll_ids, test_wavs, test_paths, trial_labels, gender, sample_rate = zip(
            *[(item.enroll_id, item.test_audio, item.test_path, item.trial_label, item.gender, item.sample_rate
               ) for item in batch])
        # padd the test wavs
        test_lengths = torch.tensor([wav.size(0) for wav in test_wavs])
        padded_test = pad_sequence(test_wavs, batch_first=True, padding_value=self.pad_value)

        return VPC25VerificationItem(
            enroll_id=enroll_ids,
            test_audio=padded_test,
            test_length=test_lengths,
            trial_label=trial_labels,
            test_path=test_paths,
            gender=gender,
            sample_rate=sample_rate
        )


class VPC25EnrollCollate(BaseCollate):
    def __call__(self, batch: List[Dict]) -> Dict:
        # Initialize a list to hold batch data
        batch_data = {'enroll_audios': [], 'speaker_ids': []}

        for item in batch:
            enroll_audios = item['enroll_audios']
            speaker_id = item['speaker_id']

            # Initialize a dictionary to hold padded audios for this speaker
            padded_audios = {}

            # Pad audios for each model
            for model, audios in enroll_audios.items():
                # Pad the audio sequences to the same length
                padded_audio = pad_sequence(audios, batch_first=True)
                padded_audios[model] = padded_audio

            batch_data['enroll_audios'].append(padded_audios)
            batch_data['speaker_ids'].append(speaker_id)

        return batch_data
    
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
        test_path = self.data_dir / row.test_path
        assert test_path.is_file(), f'File {test_path} does not exist'
        test_audio, _ = self.audio_processor.process_audio(test_path)
        
        return VPC25VerificationItem(
            enroll_id=row.enrollment_id,
            test_audio=test_audio,
            test_length=test_audio.shape[0],
            trial_label=row.label,
            test_path=test_path,
            gender=row.gender,
            sample_rate=self.audio_processor.sample_rate
        )


class VPC25EnrollDataset(Dataset):
    """
    A PyTorch Dataset class for the VoxCeleb verification task.
    Args:
        data_dir (str): Directory where the VoxCeleb dataset is stored.
        enrollment_path (str): Path to the enrollment file.
        sample_rate (int, optional): Sample rate for audio processing. Defaults to 16000.
        max_duration (Union[None, float, int], optional): Maximum duration in seconds. 
            Use None for entire utterances. Defaults to None.
    """

    def __init__(
        self,
        data_dir: str,
        enrollment_path: str,
        sample_rate: int = 16000,
        max_duration: Union[None, float, int] = None,
        sep: str = "|",
    ):
        self.data_dir = Path(data_dir)
        self.df_enrollment = pd.read_csv(enrollment_path, sep=sep)
        self.idx2speaker = {idx: speaker_id for idx, speaker_id in enumerate(self.df_enrollment.speaker_id.unique())}
        self.audio_processor = AudioProcessor(sample_rate)
        
        if isinstance(max_duration, (int, float)):
            self.max_samples = int(max_duration * sample_rate)
        elif max_duration is None:
            self.max_samples = -1
        else:
            raise ValueError("max_duration must be an int, float, or None")

    def __len__(self):
        return len(self.idx2speaker)

    def __getitem__(self, idx) -> Dict[Dict[str, List],  int]:
        enroll_id = self.idx2speaker[idx]

        # get all enrollments for the given enroll_id
        enrollment_paths = self.df_enrollment[self.df_enrollment.speaker_id == enroll_id].enrollment_path

        # get full path to enrollment_paths
        enrollment_paths = [self.data_dir / path for path in enrollment_paths]
        assert all(path.is_file() for path in enrollment_paths), f'All enrollment paths must be valid files'

        # Create a dictionary to store audios by model
        enroll_audios = {model: [] for model in self.df_enrollment[self.df_enrollment.speaker_id == enroll_id].model.unique()}

        # Process audios for each path
        for path in enrollment_paths:
            model = path.parts[-5]  # Extract model from path
            processed_audio = self.audio_processor.process_audio(path)[0]
            enroll_audios[model].append(processed_audio)
            
        return {'enroll_audios': enroll_audios, 'speaker_id': enroll_id}