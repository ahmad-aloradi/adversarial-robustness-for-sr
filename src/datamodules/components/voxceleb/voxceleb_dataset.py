from typing import List, Tuple, Union, Optional
import sys
from pathlib import Path
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd

from src.datamodules.components.utils import AudioProcessor
from src.datamodules.components.utils import CsvProcessor
from src.datamodules.components.common import get_dataset_class, VoxcelebDefaults, DatasetItem, CLASS_ID

DATASET_DEFAULTS = VoxcelebDefaults()
DATASET_CLS, DF_COLS = get_dataset_class(DATASET_DEFAULTS.dataset_name)


@dataclass
class VoxCelebVerificationItem:
    """Single verification trial from VoxCeleb dataset."""
    enroll_audio: torch.Tensor
    test_audio: torch.Tensor
    enroll_length: int
    test_length: int
    trial_label: int
    same_gender_label: int
    same_nationality_label: int
    enroll_path: str
    test_path: str
    sample_rate: int

@dataclass
class VoxcelebItem(DatasetItem):
    """Single item from dataset."""
    sample_rate: float = 16000.0

####### Coallate functions #######
class VoxCelebCollate:
    """Base collate class for variable length audio"""
    def __init__(self, pad_value: float = 0.0):
        self.pad_value = pad_value
    def __call__(self, batch):
        raise NotImplementedError

class TrainCollate(VoxCelebCollate):
    """Collate function for training data"""
    def __call__(self, batch) -> VoxcelebItem:
        waveforms, speaker_ids, audio_paths, nationalities, genders, sample_rates, recording_durations, texts = zip(
            *[(item.audio, item.speaker_id, item.audio_path, item.country, 
               item.gender, item.sample_rate, item.recording_duration, item.text) for item in batch])
        lengths = torch.tensor([wav.shape[0] for wav in waveforms])
        padded_waveforms = pad_sequence(waveforms, batch_first=True, padding_value=self.pad_value)

        gender_labels = torch.tensor([float(0) if gender == 'male' else float(1) for gender in genders])

        return VoxcelebItem(
            audio=padded_waveforms,
            speaker_id=speaker_ids,
            audio_length=lengths,
            audio_path=audio_paths,
            nationality=nationalities,
            gender=gender_labels,
            sample_rate=sample_rates,
            recording_duration=recording_durations,
            text=texts
        )

class VerificationCollate(VoxCelebCollate):
    """Collate function for verification pairs"""
    def __call__(self, batch) -> VoxCelebVerificationItem:
        enroll_wavs, test_wavs, labels, same_gender_labels, same_nationality_labels, enroll_paths, test_paths, sample_rate = zip(
            *[(item.enroll_audio, item.test_audio, item.trial_label, 
               item.same_gender_label, item.same_nationality_label,
               item.enroll_path, item.test_path, item.sample_rate) for item in batch])
        
        # Process enrollment utterances
        lengths1 = torch.tensor([wav.size(0) for wav in enroll_wavs])
        padded_wav1s = pad_sequence(enroll_wavs, batch_first=True, padding_value=self.pad_value)
        
        # Process test utterances 
        lengths2 = torch.tensor([wav.size(0) for wav in test_wavs])
        padded_wav2s = pad_sequence(test_wavs, batch_first=True, padding_value=self.pad_value)

        return VoxCelebVerificationItem(
            enroll_audio=padded_wav1s,
            enroll_length=lengths1,
            test_audio=padded_wav2s,
            test_length=lengths2,
            trial_label=labels,
            same_gender_label=same_gender_labels,
            same_nationality_label=same_nationality_labels,
            enroll_path=enroll_paths,
            test_path=test_paths,
            sample_rate=sample_rate
        )


####### Datasets #######

class VoxCelebDataset(Dataset):
    """Initialize the VoxCelebDataset
    Args:
        data_dir (str): Directory where the dataset is stored.
        metadat_filepath (Tuple[str, Path]): Path to the metadata file.
        sample_rate (int, optional): Sample rate for audio processing. Defaults to 16000.
        max_duration (float, int, optional): Maximum duration of audio samples in seconds. 
            Use -1 for the entire utterances. Defaults to 12.0.
        sep (str, optional): Separator used in the metadata file. Defaults to "|".
    """

    def __init__(
        self,
        data_dir: str,
        data_filepath: Tuple[str, Path],
        sample_rate: int,
        max_duration: Union[None, float, int, list] = 12.0,
        sep: str = "|",
    ):
        self.data_dir = Path(data_dir)
        self.dataset = pd.read_csv(data_filepath, sep=sep)
        self.audio_processor = AudioProcessor(sample_rate)
        if isinstance(max_duration, (int, float)):
            self.max_samples = int(max_duration * sample_rate)
        elif isinstance(max_duration, None):
            self.max_samples = -1
        else:
            raise ValueError("max_duration must be an int, float, or None")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> VoxcelebItem:
        # Retrieve data from csv
        row = self.dataset.iloc[idx]
        audio_path = row[DATASET_CLS.REL_FILEPATH]
        waveform, _ = self.audio_processor.process_audio(str(self.data_dir / audio_path))

        # Trim if longer than max_duration
        if self.max_samples != -1 and waveform.size(0) > self.max_samples:
            start = torch.randint(0, waveform.size(0) - self.max_samples, (1,))
            waveform = waveform[start:start + self.max_samples]
                
        return VoxcelebItem(
            audio=waveform,
            speaker_id=row[CLASS_ID].item(),
            audio_length=waveform.shape[0],
            audio_path=audio_path,
            country=row[DATASET_CLS.NATIONALITY],
            gender=row[DATASET_CLS.GENDER],
            sample_rate=self.audio_processor.sample_rate,
            recording_duration=row[DATASET_CLS.REC_DURATION],
            text=row.get(DATASET_CLS.TEXT, '')
        )

class VoxCelebVerificationDataset(Dataset):
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
        data_filepath: str,
        sample_rate: int = 16000,
        max_duration: Union[None, float, int] = None,
        sep: str = "|",
    ):
        self.data_dir = Path(data_dir)
        self.dataset = pd.read_csv(data_filepath, sep=sep)
        self.audio_processor = AudioProcessor(sample_rate)
        
        if isinstance(max_duration, (int, float)):
            self.max_samples = int(max_duration * sample_rate)
        elif max_duration is None:
            self.max_samples = -1
        else:
            raise ValueError("max_duration must be an int, float, or None")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> VoxCelebVerificationItem:
        # Load paths and label from dataset
        row = self.dataset.iloc[idx]
        enroll_path = row.enroll_path
        test_path = row.test_path
        
        # Load and process both utterances
        enroll_wav, _ = self.audio_processor.process_audio(str(self.data_dir / enroll_path))
        test_wav, _ = self.audio_processor.process_audio(str(self.data_dir / test_path))
        
        return VoxCelebVerificationItem(
            enroll_audio=enroll_wav,
            test_audio=test_wav,
            enroll_length=enroll_wav.shape[0],
            test_length=test_wav.shape[0],
            trial_label=row.label,
            same_gender_label=row.same_gender.item(), 
            same_nationality_label=row.same_nationality.item(),
            enroll_path=enroll_path,
            test_path=test_path,
            sample_rate=self.audio_processor.sample_rate
        )


if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader
    
    parser = argparse.ArgumentParser(description="Generate train_list.txt for VoxCeleb")
    parser.add_argument("--voxceleb_dir", 
                        type=str,
                        default="data/voxceleb/voxceleb1_2",)
    parser.add_argument("--data_filepath", 
                        type=str, 
                        default="data/voxceleb/voxceleb_metadata/metadata/voxceleb_dev.csv",
                        help="Output file path")
    parser.add_argument("--verification_file",
                        type=str,
                        help="Path to veri_test.txt if excluding verification files")
    args = parser.parse_args()    

    voxceleb_data = VoxCelebDataset(data_dir=args.voxceleb_dir, 
                                    data_filepath=args.data_filepath,
                                    sample_rate=16000.0,)

    print("Number of samples: ", len(voxceleb_data))
    print("Sample: ", voxceleb_data.__getitem__(0))

    dataloder = DataLoader(voxceleb_data, batch_size=2, collate_fn=TrainCollate())
    for batch in dataloder:
        print(batch)
        break

    # Test II
    df = pd.read_csv("data/voxceleb/voxceleb_metadata/metadata/voxceleb_dev.csv", sep='|')

    train_df, val_df = CsvProcessor.split_dataset(df=df,
                                                  save_csv=False,
                                                  train_ratio = 0.98,
                                                  speaker_overlap=False)
    
    # Print statistics
    print(f"\nDataset split statistics:")
    print(f"Training samples: {len(train_df)} ({len(train_df)/len(df):.1%})")
    print(f"Validation samples: {len(val_df)} ({len(val_df)/len(df):.1%})")
    print(f"Training speakers: {len(train_df['speaker_id'].unique())}")
    print(f"Validation speakers: {len(val_df['speaker_id'].unique())}")

    # Test III
    test_filepath = '/home/aloradi/adversarial-robustness-for-sr/data/voxceleb/voxceleb_metadata/preprocessed/veri_test_rich.csv'
    voxceleb_data = VoxCelebVerificationDataset(data_dir=args.voxceleb_dir, 
                                                data_filepath=test_filepath, 
                                                sample_rate=16000.0)
    print("Number of samples in test set: ", len(voxceleb_data))
    print("Sample: ", voxceleb_data.__getitem__(0))

    dataloder = DataLoader(voxceleb_data, batch_size=2, collate_fn=VerificationCollate())
    for batch in dataloder:
        print(batch)
        break
