from operator import itemgetter
from typing import Iterator, Optional, Tuple, List, Dict
from pathlib import Path
import random

import pandas as pd
import torch
import torchaudio
import torchaudio.functional as F
from torch.utils.data import Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from src.datamodules.components.common import BaseDatasetCols, CLASS_ID
from src import utils

log = utils.get_pylogger(__name__)
DATESET_CLS = BaseDatasetCols()

class CsvProcessor:
    """
    Utility class for handling CSV files with metadata.
    """
        
    def __init__(self, verbose: bool = False, fill_value: str = 'N/A'):
        self.verbose = verbose
        self.fill_value = fill_value


    def read_csv(self, csv_path: str, sep: str = '|') -> pd.DataFrame:
        """
        Read CSV file and return DataFrame

        Args:
            csv_path: Path to CSV file
            sep: Separator for CSV file (default: '|')

        Returns:
            pd.DataFrame: DataFrame from CSV file
        """
        # Read voxceleb's metada file
        df = pd.read_csv(csv_path, sep=sep)
        return df.fillna(self.fill_value)


    def concatenate_metadata(self, csv_paths, fill_value='N/A', speaker_id_col='speaker_id', sep='|'
                             ) -> pd.DataFrame:
        """
        Concatenate multiple CSVs with handling for different columns and unique IDs.
        
        Args:
            csv_paths (list): List of paths to CSV files
            fill_value: Value to fill missing columns with (default: None)
            speaker_id_col (str): Name of the speaker ID column (default: 'speaker_id')
            utterance_id_col (str): Name of the utterance ID column (default: 'utterance_id')
            rel_path_col (str): Name of the relative path column (default: 'rel_path')
        
        Returns:
            pd.DataFrame: Concatenated DataFrame with unique IDs
        """
        # Read all CSVs and store their DataFrames
        dfs = []
        for path in csv_paths:
            df = self.read_csv(path, sep=sep)
            dfs.append(df)
        
        combined_df = pd.concat(dfs, ignore_index=True)        
        # Ensure speaker IDs are unique by adding prefix if needed
        if speaker_id_col in combined_df.columns:
            speaker_counts = combined_df[speaker_id_col].value_counts()
            duplicate_speakers = speaker_counts[speaker_counts > 1].index
           
            for speaker in duplicate_speakers:
                mask = combined_df[speaker_id_col] == speaker
                indices = combined_df[mask].index
                for i, idx in enumerate(indices):
                    if i > 0:  # Skip first occurrence
                        combined_df.loc[idx, speaker_id_col] = f"{speaker}_v{i}"
                    
        combined_df = combined_df.fillna(fill_value)
        
        return combined_df


    @staticmethod
    def concatenate_csvs(csv_paths, 
                         fill_value='N/A',
                         rel_path_col=DATESET_CLS.REL_FILEPATH,
                         sep='|') -> pd.DataFrame:
        """
        Concatenate multiple CSVs with handling for different columns and unique paths.
        
        Args:
            csv_paths (list): List of paths to CSV files
            fill_value: Value to fill missing columns with (default: None)
            rel_path_col (str): Name of the relative path column (default: 'rel_path')
        
        Returns:
            pd.DataFrame: Concatenated DataFrame with unique IDs
        """
        # Read all CSVs and store their DataFrames
        dfs = []
        for path in csv_paths:
            df = pd.read_csv(path, sep=sep)                
            dfs.append(df)
        
        # Concatenate all DataFrames
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Ensure relative paths are unique
        utterance_counts = combined_df[rel_path_col].value_counts()
        duplicate_utterances = utterance_counts[utterance_counts > 1].index
        
        if not duplicate_utterances.empty:
            error_msg = "Duplicate utterance IDs found:\n"
            for utterance, count in duplicate_utterances.items():
                error_msg += f"- '{utterance}' appears {count} times\n"
            raise ValueError(error_msg)
            
        # Fill missing values
        combined_df = combined_df.fillna(fill_value)
        
        return combined_df


    @staticmethod
    def append_speaker_stats(df: pd.DataFrame, 
                             speaker_stats: pd.DataFrame,
                             col_id: str = DATESET_CLS.SPEAKER_ID) -> pd.DataFrame:
        """Append speaker stats to metadata"""
        df = df.merge(speaker_stats, on=col_id, how='left')
        df = df.sort_values('total_dur/spk', ascending=False)
        df = df.reset_index(drop=True)
        return df


    @staticmethod
    def get_speakers_stats(df: pd.DataFrame,
                           col_id: str,
                           duration_col: str,
                           rounding: int = 4) -> pd.DataFrame:
        speaker_stats = df.groupby(col_id).agg(
            {duration_col: ['sum', 'mean', 'count']}).round(rounding)

        speaker_stats.columns = pd.MultiIndex.from_tuples([
            (duration_col, 'total_dur/spk'),
            (duration_col, 'mean_dur/spk'),
            (duration_col, 'utterances/spk')
        ])
        speaker_stats.columns = speaker_stats.columns.get_level_values(1)
        # Reset index to make speaker_id a column
        speaker_stats = speaker_stats.reset_index()
        return speaker_stats


    @staticmethod
    def generate_training_ids(combined_df: pd.DataFrame,
                              id_col: str = DATESET_CLS.SPEAKER_ID,
                              verbose=True) -> pd.DataFrame:
        """
        Generate training IDs from combined VoxCeleb1 and VoxCeleb2 metadata

        Args:
            metadata_files: List of paths to metadata CSV files
            
        Returns:
            Dictionary mapping original speaker IDs to numerical training IDs
            
        Example:
            {'id1': 0, 'id2': 1, ...}
        """        
        # Sort speakers for consistent ordering
        sorted_speakers = sorted(combined_df[id_col].unique())

        # Create mapping dictionary
        speaker_to_id = {speaker: idx for idx, speaker in enumerate(sorted_speakers)}
        
        if verbose:
            log.info(f"Generated training IDs for {len(speaker_to_id)} unique speakers")

        return speaker_to_id


    @staticmethod
    def update_metadata_with_training_ids(df: pd.DataFrame,
                                          speaker_to_id: Dict[str, int],
                                          id_col: str = DATESET_CLS.SPEAKER_ID, 
                                          verbose: bool = True,
                                          class_id_col: str = CLASS_ID) -> pd.DataFrame:
        """
        Update metadata CSV file with training_id column
        
        Args:
            df: metadata as da dataframe
            speaker_to_id: Dictionary mapping speaker IDs to training IDs
            backup: Whether to create backup of original file
        """                        
        # Add class_id column
        df[class_id_col] = df[id_col].map(speaker_to_id).astype(int)
        
        # Verify no missing mappings
        missing_ids = df[df[class_id_col].isna()][id_col].unique()
        if len(missing_ids) > 0:
            raise RuntimeWarning(f"Warning: No training ID mapping for speakers: {missing_ids}")
        
        if verbose:
            log.info(f"Total speakers: {len(df[id_col].unique())}")
            log.info(f"Training ID range: {df[class_id_col].min()} - {df[class_id_col].max()}")
        
        return df


    def process(self,
                dataset_files: List[str],
                spks_metadata_paths: List[str],
                speaker_id_col: str = DATESET_CLS.SPEAKER_ID,
                rel_path_col: str = DATESET_CLS.REL_FILEPATH,
                duration_col: str = DATESET_CLS.REC_DURATION,
                rounding: int = 4,
                class_id: str = CLASS_ID,
                verbose: bool = True,
                sep: str = '|') -> Tuple[pd.DataFrame, pd.DataFrame]:
        
        # read and append metadata
        dataset_df = self.concatenate_csvs(dataset_files, fill_value=self.fill_value, rel_path_col=rel_path_col, sep=sep)

        # generate training ids
        speaker_to_id = self.generate_training_ids(dataset_df, id_col=speaker_id_col, verbose=verbose)

        # append training ids to metadata        
        updated_dataset_df = self.update_metadata_with_training_ids(df=dataset_df,
                                                                    speaker_to_id=speaker_to_id,
                                                                    id_col=speaker_id_col,
                                                                    verbose=verbose,
                                                                    class_id_col=class_id)

        # get speaker stats
        speaker_stats = self.get_speakers_stats(df=updated_dataset_df,
                                                col_id=speaker_id_col,
                                                duration_col=duration_col,
                                                rounding=rounding)

        # append speaker stats to metadata
        final_df = self.append_speaker_stats(df=updated_dataset_df, speaker_stats=speaker_stats, col_id=speaker_id_col)

        # get combined speakers' metadata
        combined_metadatda = self.concatenate_metadata(spks_metadata_paths, 
                                                       fill_value='N/A',
                                                       speaker_id_col=speaker_id_col,
                                                       sep=sep)

        speaker_to_id_df = self.lookup_with_metadata(speaker_to_id=speaker_to_id, 
                                                     speaker_stats=speaker_stats,
                                                     metadata=combined_metadatda,
                                                     speaker_id_col=speaker_id_col,
                                                     class_id=class_id)
        return final_df, speaker_to_id_df
            

    def lookup_with_metadata(self,
                             speaker_to_id: Dict,
                             speaker_stats: pd.DataFrame,
                             metadata: pd.DataFrame,
                             speaker_id_col: str = DATESET_CLS.SPEAKER_ID,
                             class_id: str = CLASS_ID) -> pd.DataFrame:
        # Save speaker lookup table with class IDs
        speaker_to_id_df = pd.DataFrame({speaker_id_col: speaker_to_id.keys(), class_id: speaker_to_id.values()})
        speaker_to_id_df = self.append_speaker_stats(df=speaker_to_id_df, speaker_stats=speaker_stats, col_id=speaker_id_col)
        speaker_to_id_df = speaker_to_id_df.merge(metadata, on=speaker_id_col, how='left')
        return speaker_to_id_df


    @staticmethod
    def split_dataset(
            df: pd.DataFrame,
            train_ratio: float = 0.95,
            speaker_overlap: bool = False,
            save_csv: bool = True,
            speaker_id_col: str = "speaker_id",
            train_csv: str = "train.csv",
            val_csv: str = "validation.csv",
            sep: str = '|',
            seed: int = 42,
            fill_value: str = 'N/A'
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
            shuffled_df = df.sample(frac=1.0, random_state=seed)
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
            train_df.fillna(fill_value).to_csv(train_csv, index=False, sep=sep)
            val_df.fillna(fill_value).to_csv(val_csv, index=False, sep=sep)

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
            return waveform.squeeze(0), self.sample_rate
            
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


class SimpleAudioDataset(Dataset):
    def __init__(self, df, wav_dir, crop_len=None, sr=16000):
        self.df = df
        self.wav_dir = wav_dir
        self.crop_len = crop_len # in seconds

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        wav_path = str(self.wav_dir / self.df.iloc[idx].rel_filepath)
        waveform, sr = torchaudio.load(wav_path)
        if self.crop_len:
            waveform = waveform[:, : int(sr * self.crop_len)]
        return waveform.squeeze(0), idx

    @staticmethod
    def collate_fn(batch):
        """Collate function to ensure all audio data are of same length.
        Pads shorter sequences and truncates longer ones.

        Args:
            batch: List of tuples (waveform, index)
        
        Returns:
            Tuple of (padded_waveforms, indices)
        """
        # Extract waveforms and indices
        waveforms, indices = zip(*batch)

        # Find max length in current batch
        max_len = max(len(waveform) for waveform in waveforms)

        # Pad or truncate each waveform to max_len
        processed_waveforms = []
        for waveform in waveforms:
            curr_len = len(waveform)
            if curr_len > max_len:
                # Truncate
                processed_waveforms.append(waveform[:max_len])
            else:
                # Pad with zeros
                padding = torch.zeros(max_len - curr_len)
                processed_waveforms.append(torch.cat([waveform, padding]))

        # Stack all waveforms into a single tensor
        waveforms_tensor = torch.stack(processed_waveforms)
        return waveforms_tensor, indices


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