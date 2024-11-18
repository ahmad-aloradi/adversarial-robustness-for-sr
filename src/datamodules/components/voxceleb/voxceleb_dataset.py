from typing import List, Tuple
import torch
from torch.utils.data import Dataset
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
import pandas as pd

import sys
sys.path.append(f"/home/aloradi/adversarial-robustness-for-sr")
from src.datamodules.components.utils import AudioProcessor


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
            'x_enrol': padded_wav1s,
            'x_enrol_len': lengths1,
            'x_test': padded_wav2s,
            'x_test_len': lengths2,
            'labels': torch.tensor(labels)
        }


class VoxCelebDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        metadat_filepath: Tuple[str, Path],
        sample_rate: int = 16000,
        max_duration: float = 12.0,
        sep: str = "|",
    ):
        self.data_dir = Path(data_dir)
        self.metadata = pd.read_csv(metadat_filepath,sep=sep)
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate train_list.txt for VoxCeleb")
    parser.add_argument("--voxceleb_dir", 
                        type=str,
                        default="data/voxceleb/voxceleb1_2",)
    parser.add_argument("--metadata_file", 
                        type=str, 
                        default="data/voxceleb/voxceleb_metadata/preprocessed/voxceleb_dev.csv",
                        help="Output file path")
    parser.add_argument("--verification_file",
                        type=str,
                        help="Path to veri_test.txt if excluding verification files")
    args = parser.parse_args()    

    voxceleb_data = VoxCelebDataset(data_dir=args.voxceleb_dir, 
                                    metadat_filepath=args.metadata_file)

    print("Number of samples: ", len(voxceleb_data))
    print("Sample: ", voxceleb_data[0])
