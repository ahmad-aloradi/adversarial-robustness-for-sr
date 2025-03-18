from typing import List, Tuple, Union, Optional, Literal
import sys
from pathlib import Path
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd

from src.datamodules.components.utils import AudioProcessor, CsvProcessor, BaseCollate, BaseDataset
from src.datamodules.components.common import get_dataset_class, VoxcelebDefaults, DatasetItem

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
    same_country_label: int
    enroll_path: str
    test_path: str
    sample_rate: int

@dataclass
class VoxcelebItem(DatasetItem):
    """Single item from dataset."""
    sample_rate: float = 16000.0


@dataclass
class VoxCelebEnrollItem:
    """Single verification trial from VoxCeleb dataset."""
    audio: torch.Tensor
    audio_length: torch.Tensor
    audio_path: str
    speaker_id: str

####### Coallate functions #######
class TrainCollate(BaseCollate):
    pass

class VerificationCollate(BaseCollate):
    """Collate function for verification pairs"""
    def __call__(self, batch) -> VoxCelebVerificationItem:
        enroll_wavs, test_wavs, labels, same_gender_labels, same_country_labels, enroll_paths, test_paths, sample_rate = zip(
            *[(item.enroll_audio, item.test_audio, item.trial_label, 
               item.same_gender_label, item.same_country_label,
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
            same_country_label=same_country_labels,
            enroll_path=enroll_paths,
            test_path=test_paths,
            sample_rate=sample_rate
        )


class EnrollCoallate(BaseCollate):
    def __init__(self, pad_value = 0):
        super().__init__(pad_value)

    def __call__(self, batch):
        spk_ids = [item[0] for item in batch]
        audios = [item[1] for item in batch]
        paths = [item[2] for item in batch]
        lengths = torch.tensor([wav.shape[0] for wav in audios])
        # pad the audio sequences
        padded_audio = pad_sequence(audios, batch_first=True, padding_value=self.pad_value)

        assert len(spk_ids) == 1, 'Only 1 per enrollment is supported at the moment'
        
        return VoxCelebEnrollItem(
            audio=padded_audio,
            audio_length=lengths,
            speaker_id=spk_ids,
            audio_path=paths
            )
    
####### Datasets #######

class VoxCelebDataset(BaseDataset):
    pass


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
            same_country_label=row.same_country.item(),
            enroll_path=enroll_path,
            test_path=test_path,
            sample_rate=self.audio_processor.sample_rate
        )


class VoxCelebEnroll(Dataset):    
    def __init__(
        self,
        data_dir: str,
        phase: Literal['enrollment', 'test'],
        dataset: pd.DataFrame,
        sample_rate: int = 16000,
    ):
        self.data_dir = Path(data_dir)
        self.phase = phase
        self.dataset = dataset
        self.audio_processor = AudioProcessor(sample_rate)
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> VoxCelebVerificationItem:
        # Load paths and label from dataset
        row = self.dataset.iloc[idx]
        path = row.enroll_path if self.phase == 'enrollment' else row.test_path
        spk_id = row.enroll_id if self.phase == 'enrollment' else row.test_id

        # Load and process both utterances
        wav, _ = self.audio_processor.process_audio(str(self.data_dir / path))
        
        return (spk_id, wav, path)



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

    voxceleb_data = VoxCelebDataset(data_dir=args.voxceleb_dir, data_filepath=args.data_filepath, sample_rate=16000.0,)

    print("Number of samples: ", len(voxceleb_data))
    dataloder = DataLoader(voxceleb_data, batch_size=2, collate_fn=TrainCollate())
    for batch in dataloder:
        print(batch)
        break

    # Test II
    df = pd.read_csv("data/voxceleb/voxceleb_metadata/metadata/voxceleb_dev.csv", sep='|')
    train_df, val_df = CsvProcessor.split_dataset(df=df, save_csv=False, train_ratio = 0.98, speaker_overlap=False)
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
