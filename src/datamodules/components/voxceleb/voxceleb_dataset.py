from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset
from pathlib import Path
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence
from src.datamodules.components.audio_processor import AudioProcessor


@dataclass
class VoxCelebConfig:
    data_dir: str = "/path/to/voxceleb"  # Change this to your path
    veri_test_path: str = "/path/to/veri_test.txt"  # Change this to your path
    batch_size: int = 32
    num_workers: int = 4
    sample_rate: int = 16000
    max_duration: float = 8.0  # maximum duration in seconds
    train_list: str = "train_list.csv"


class VoxCelebCollate:
    """Base collate class for variable length audio"""
    def __init__(self, pad_value: float = 0.0):
        self.pad_value = pad_value

    def __call__(self, batch):
        raise NotImplementedError


class TrainCollate(VoxCelebCollate):
    """Collate function for training data"""
    def __call__(self, batch):
        waveforms, speaker_ids = zip(*batch)
        lengths = torch.tensor([wav.size(0) for wav in waveforms])
        padded_waveforms = pad_sequence(waveforms, batch_first=True, padding_value=self.pad_value)
        return {
            'waveforms': padded_waveforms,
            'lengths': lengths,
            'speaker_ids': torch.tensor(speaker_ids)
        }


class VerificationCollate(VoxCelebCollate):
    """Collate function for verification pairs"""
    def __call__(self, batch):
        wav1s, wav2s, labels = zip(*batch)
        # Process first utterances
        lengths1 = torch.tensor([wav.size(0) for wav in wav1s])
        padded_wav1s = pad_sequence(wav1s, batch_first=True, padding_value=self.pad_value)
        # Process second utterances
        lengths2 = torch.tensor([wav.size(0) for wav in wav2s])
        padded_wav2s = pad_sequence(wav2s, batch_first=True, padding_value=self.pad_value)
        return {
            'waveforms1': padded_wav1s,
            'lengths1': lengths1,
            'waveforms2': padded_wav2s,
            'lengths2': lengths2,
            'labels': torch.tensor(labels)
        }


class VoxCelebDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        metadata: List[Tuple[str, int]],
        sample_rate: int = 16000,
        max_duration: float = 15.0,
    ):
        self.data_dir = Path(data_dir)
        self.metadata = metadata
        self.max_samples = int(max_duration * sample_rate)
        self.audio_processor = AudioProcessor(sample_rate)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        audio_path, speaker_id = self.metadata[idx]
        waveform = self.audio_processor.process_audio(str(self.data_dir / audio_path))
        
        # Trim if longer than max_duration
        if waveform.size(0) > self.max_samples:
            start = torch.randint(0, waveform.size(0) - self.max_samples, (1,))
            waveform = waveform[start:start + self.max_samples]
        
        # Skip if shorter than minimum duration
        if waveform.size(0) < self.min_samples:
            # Get a new sample
            return self.__getitem__(torch.randint(0, len(self), (1,)).item())
        
        return waveform, speaker_id


class VoxCelebVerificationDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        veri_test_path: str,
        sample_rate: int = 16000,
    ):
        self.data_dir = Path(data_dir)
        self.audio_processor = AudioProcessor(sample_rate)
        
        # Load verification pairs
        self.pairs = []
        with open(veri_test_path, 'r') as f:
            for line in f:
                label, path1, path2 = line.strip().split()
                self.pairs.append((int(label), path1, path2))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        label, path1, path2 = self.pairs[idx]
        # Load and process both utterances
        wav1 = self.audio_processor.process_audio(str(self.data_dir / path1))
        wav2 = self.audio_processor.process_audio(str(self.data_dir / path2))        
        return wav1, wav2, label


    import argparse
    parser = argparse.ArgumentParser(description="Generate train_list.txt for VoxCeleb")
    parser.add_argument("--voxceleb_dir", type=str, required=True, 
                      help="Root directory of VoxCeleb dataset")
    parser.add_argument("--output_file", type=str, default="train_list.txt",
                      help="Output file path")
    parser.add_argument("--verification_file", type=str,
                      help="Path to veri_test.txt if excluding verification files")
    
    args = parser.parse_args()
    
    generate_train_list(
        voxceleb_dir=args.voxceleb_dir,
        output_file=args.output_file,
        verification_file=args.verification_file
    )