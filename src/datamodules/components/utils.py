import os
from operator import itemgetter
from typing import Iterator, Optional, Tuple, Union
from pathlib import Path
import random

import pandas as pd
import torch
import torchaudio
import torchaudio.functional as F
from torch.utils.data import Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler


def split_dataset(
        df: pd.DataFrame,
        train_ratio: float = 0.95,
        speaker_overlap: bool = False,
        save_csv: bool = True,
        speaker_id_col: str = "speaker_id",
        train_csv: str = "train.csv",
        val_csv: str = "validation.csv",
        sep: str = '|',
        seed: int = 42
        ) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Splits a dataset into training and validation sets.

    Parameters:
    df (pd.DataFrame): The input dataframe to split.
    train_ratio (float): The ratio of the dataset to use for training. Default is 95%
    speaker_overlap (bool): If True, splits randomly across all samples. If False, splits by unique speakers. Default is False.
    save_csv (bool): If True, saves the splits to CSV files. If False, returns the splits as dataframes. Default is True.
    speaker_id_col (str): The column name for speaker IDs. Default is "speaker_id".
    train_csv (str): The filename for the training set CSV. Default is "train.csv".
    val_csv (str): The filename for the validation set CSV. Default is "validation.csv".
    sep (str): The delimiter to use in the CSV files. Default is '|'.
    seed (int): The random seed for reproducibility. Default is 42.

    Returns:
    Optional[Tuple[pd.DataFrame, pd.DataFrame]]: A tuple containing the training and validation dataframes if save_csv is False. Otherwise, returns None.
    """

    if speaker_overlap:
        # Random split across all samples
        shuffled_df = df.sample(frac=1.0, random_state=RANDOM_SEED)
        split_idx = int(len(shuffled_df) * train_ratio)
        train_df = shuffled_df[:split_idx]
        val_df = shuffled_df[split_idx:]
    else:
        # Split by speakers
        speakers = df[speaker_id_col].unique()
        random.seed(seed)
        random.shuffle(speakers)

        train_speakers = speakers[:int(len(speakers) * train_ratio)]
        train_df = df[df[speaker_id_col].isin(train_speakers)]
        val_df = df[~df[speaker_id_col].isin(train_speakers)]

    if save_csv:
        train_df.fillna('N/A').to_csv(train_csv, index=False, sep=sep)
        val_df.fillna('N/A').to_csv(val_csv, index=False, sep=sep)

    else:
        return train_df, val_df


class AudioProcessor:
    """Handles audio loading and preprocessing"""
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def load_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """Load audio file and return waveform and sample rate"""
        waveform, sr = torchaudio.load(audio_path)
        return waveform, sr

    def convert_to_mono(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert multi-channel audio to mono"""
        if waveform.shape[0] > 1:
            return torch.mean(waveform, dim=0, keepdim=True)
        return waveform

    def resample(self, waveform: torch.Tensor, orig_sr: int) -> torch.Tensor:
        """Resample audio to target sample rate"""
        if orig_sr == self.sample_rate:
            return waveform
        return torchaudio.functional.resample(
                waveform, 
                orig_freq=orig_sr, 
                new_freq=self.sample_rate
            )

    def normalize_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """Normalize audio file and return waveform"""
        waveform = waveform.squeeze(0)
        assert waveform.dim() == 1, "Expected single-channel audio"
        waveform = waveform - waveform.mean()
        waveform = waveform / (waveform.abs().max() + torch.finfo(torch.float32).eps)
        return waveform.unsqueeze(0)

    def resolve_path(self, audio_path: str) -> Path:
        """
        Safely resolve and validate audio file path
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Resolved absolute path
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If path is invalid
        """
        try:
            path = Path(audio_path).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Audio file not found: {path}")
            return path
        except Exception as e:
            raise ValueError(f"Invalid audio path {audio_path}: {str(e)}")

    def process_audio(self, audio_path: str) -> torch.Tensor:
        """
        Complete audio processing pipeline with robust path handling
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Processed audio tensor
        """
        try:
            # Resolve and validate path
            abs_path = self.resolve_path(audio_path)
            waveform, sr = self.load_audio(str(abs_path))
            # Process pipeline
            waveform = self.convert_to_mono(waveform)
            waveform = self.resample(waveform, orig_sr=sr)
            waveform = self.normalize_audio(waveform)
            return waveform.squeeze(0)
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Audio file not found: {audio_path}") from e
        
        except Exception as e:
            raise RuntimeError(f"Error processing audio file {audio_path}: {str(e)}") from e

    def apply_preemphasis(
        self,
        waveform: torch.Tensor,
        coeff: float = 0.97
    ) -> torch.Tensor:
        """
        Apply pre-emphasis using torchaudio's functional API
        
        Args:
            waveform: Input waveform tensor of shape (channels, samples) or (samples,)
            coeff: Pre-emphasis coefficient (default: 0.97)
            
        Returns:
            Pre-emphasized waveform of same shape as input
        """
        # Ensure 2D input
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            
        return F.preemphasis(waveform, coeff)
    
    def remove_preemphasis(
        self,
        waveform: torch.Tensor,
        coeff: float = 0.97
    ) -> torch.Tensor:
        """
        Apply de-emphasis using torchaudio's functional API
        
        Args:
            waveform: Pre-emphasized waveform tensor of shape (channels, samples) or (samples,)
            coeff: Pre-emphasis coefficient (default: 0.97)
            
        Returns:
            De-emphasized waveform of same shape as input
        """
        # Ensure 2D input
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            
        return F.deemphasis(waveform, coeff)


class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.

    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.

        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)


class DistributedSamplerWrapper(DistributedSampler):
    """Wrapper over `Sampler` for distributed training. Allows you to use any
    sampler in distributed mode. It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each process can
    pass a DistributedSamplerWrapper instance as a DataLoader sampler, and load
    a subset of subsampled data of the original dataset that is exclusive to
    it.

    .. note::     Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
              distributed training
            rank (int, optional): Rank of the current process
              within ``num_replicas``
            shuffle (bool, optional): If true (default),
              sampler will shuffle the indices
        """
        super().__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.

        Returns:
            python iterator
        """
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        sub_sampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(sub_sampler_indexes))



# Example usage and test
if __name__ == "__main__":
    # Create a sample waveform
    from matplotlib import pyplot as plt
    waveform, sample_rate = torchaudio.load("data/sample.wav") 
    
    ap = AudioProcessor(sample_rate=sample_rate)
    # Apply pre-emphasis
    emphasized = ap.apply_preemphasis(waveform, coeff=0.09)
    # Remove pre-emphasis to verify reconstruction
    reconstructed = ap.remove_preemphasis(emphasized, coeff=0.09)
    
    # Plot original and reconstructed waveforms
    plt.figure(figsize=(12, 6))
    plt.plot(waveform, label="Original waveform")
    plt.plot(emphasized.T, label="emphasized waveform")
    plt.plot(reconstructed.T, label="Reconstructed waveform")
    plt.legend()
    plt.show()