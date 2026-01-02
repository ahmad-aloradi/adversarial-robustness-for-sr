from operator import itemgetter
from typing import Iterator, Optional, Tuple, List, Dict, Union, Any
from pathlib import Path
import random
import json

import pandas as pd
import torch
import torchaudio
import torchaudio.functional as F
from torch.utils.data import Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence

from src.datamodules.components.common import BaseDatasetCols, DatasetItem
from src.utils.pylogger import get_pylogger

log = get_pylogger(__name__)
DATESET_CLS = BaseDatasetCols()


def make_dataloader(
    dataset: Dataset,
    loader_kwargs: dict,
    collate_fn,
    batch_sampler_cfg=None,
) -> "torch.utils.data.DataLoader":
    """Create a DataLoader with optional batch_sampler (e.g. PKSpeakerBatchSampler) + DDP support.

    Args:
        dataset: The PyTorch Dataset to load from.
        loader_kwargs: Dict of DataLoader kwargs (batch_size, shuffle, num_workers, etc.).
        collate_fn: Collate function for the DataLoader.
        batch_sampler_cfg: None for default batching, or a Hydra config/partial for a batch sampler.
            If using Hydra DI, set `_partial_: true` so the sampler factory receives the dataset.

    Returns:
        A configured DataLoader.
    """
    from torch.utils.data import DataLoader

    if batch_sampler_cfg is not None:
        # Handle Hydra DictConfig: instantiate it (with _partial_=true it becomes a callable factory)
        if hasattr(batch_sampler_cfg, "_target_"):
            import hydra
            batch_sampler_cfg = hydra.utils.instantiate(batch_sampler_cfg)

        # If it's a partial/factory, call it with the dataset to get the actual sampler.
        batch_sampler = batch_sampler_cfg(dataset) if callable(batch_sampler_cfg) else batch_sampler_cfg

        # Wrap for DDP if needed (so each rank gets different batches).
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if not isinstance(batch_sampler, DistributedSamplerWrapper):
                batch_sampler = DistributedSamplerWrapper(batch_sampler, shuffle=False, drop_last=True)

        # DataLoader cannot accept batch_size/shuffle/drop_last alongside batch_sampler.
        kwargs = {k: v for k, v in loader_kwargs.items() if k not in ("batch_size", "shuffle", "drop_last")}
        return DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn, **kwargs)

    return DataLoader(dataset, collate_fn=collate_fn, **loader_kwargs)


######### Processing utils #########

def segment_utterance(
    speaker_id: str,
    rel_filepath: str,
    recording_duration: float,
    segment_duration: float,
    segment_overlap: float,
    min_segment_duration: float,
    original_row: dict,
    base_start_time: float = 0.0,
    vad_speech_timestamps: Optional[Union[str, List[Tuple[float, float]]]] = None,
    segment_max_silence_ratio: Optional[float] = None,
) -> list:
    """
    Segment a single utterance into fixed-duration chunks.
    
    This is a shared utility function used across all dataset processors
    (CNCeleb, VoxCeleb, LibriSpeech) for consistent segmentation behavior.
    
    Tail handling: If the remaining audio after the last full segment is shorter
    than segment_duration, it is merged into the previous segment rather than
    creating a tiny final segment. This means min_segment_duration only applies
    to recordings shorter than segment_duration (i.e., the first/only segment).
    
    Args:
        speaker_id: Speaker identifier
        rel_filepath: Relative file path
        recording_duration: Total duration of the recording in seconds
        segment_duration: Desired segment duration in seconds
        segment_overlap: Overlap between segments in seconds
        min_segment_duration: Minimum duration for short recordings (< segment_duration)
            to be included as a single segment. Has no effect on longer recordings.
        original_row: Dictionary containing other metadata fields
        base_start_time: Offset for absolute timestamps (e.g., from VAD)
        vad_speech_timestamps: Optional speech segments for silence filtering
        segment_max_silence_ratio: Optional max silence ratio to filter segments
        
    Returns:
        List of dictionaries, each representing a segment
    """
    # Early exit: recording too short to be useful
    if recording_duration < min_segment_duration:
        return []
    
    # Calculate step size (hop) between segments
    step_size = segment_duration - segment_overlap
    if step_size <= 0:
        raise ValueError(f"segment_overlap ({segment_overlap}) must be less than segment_duration ({segment_duration})")
    
    # Build segment ID prefix from speaker_id and filepath components
    path_no_ext = Path(rel_filepath).with_suffix('').as_posix()
    path_tokens = path_no_ext.replace('/', '_').replace('\\', '_').split('_')
    speaker_parts = {p.lower() for p in speaker_id.split('_')}
    
    unique_tokens = []
    seen_lower = set()
    for token in path_tokens:
        token_lower = token.lower()
        if token and token_lower not in speaker_parts and token_lower not in seen_lower:
            unique_tokens.append(token)
            seen_lower.add(token_lower)
    
    segment_prefix = f"{speaker_id}_{'_'.join(unique_tokens)}" if unique_tokens else speaker_id
    
    # Handle VAD chunk suffix if present
    if isinstance(original_row, dict) and original_row.get('vad_chunk_id'):
        safe_chunk = str(original_row.get('vad_chunk_id')).replace('/', '_').replace('\\', '_')
        segment_prefix = f"{segment_prefix}_{safe_chunk}"

    def _parse_speech_segments(value: Optional[Union[str, List[Tuple[float, float]]]]) -> Optional[List[Tuple[float, float]]]:
        if value is None:
            return None
        if isinstance(value, list):
            try:
                return [(float(s), float(e)) for (s, e) in value]
            except Exception:
                return None
        if isinstance(value, str):
            try:
                decoded = json.loads(value)
                if isinstance(decoded, list):
                    return [(float(s), float(e)) for (s, e) in decoded]
            except Exception:
                return None
        return None

    def _speech_fraction(seg_start: float, seg_end: float, speech: List[Tuple[float, float]]) -> float:
        if seg_end <= seg_start:
            return 0.0
        total = 0.0
        for s, e in speech:
            overlap = max(0.0, min(seg_end, e) - max(seg_start, s))
            total += overlap
        return total / (seg_end - seg_start)

    def _exceeds_silence_threshold(start_abs: float, end_abs: float) -> bool:
        """Check if segment exceeds the silence ratio threshold."""
        if speech_segments is None or segment_max_silence_ratio is None:
            return False
        speech_ratio = _speech_fraction(start_abs, end_abs, speech_segments)
        return (1.0 - speech_ratio) > float(segment_max_silence_ratio)

    speech_segments = _parse_speech_segments(vad_speech_timestamps)
    segments = []
    start_time = 0.0
    segment_idx = 0

    while start_time < recording_duration:
        remaining = recording_duration - start_time
        
        # Tail merging: if remaining < segment_duration and we have previous segments,
        # extend the last segment to cover the remainder instead of creating a short one.
        if segments and remaining < segment_duration:
            last_seg = segments[-1]
            last_start_abs = float(last_seg['start_time'])
            merged_end_abs = base_start_time + recording_duration
            
            # Only merge if it doesn't exceed silence threshold
            if not _exceeds_silence_threshold(last_start_abs, merged_end_abs):
                last_seg['end_time'] = round(merged_end_abs, 3)
                last_seg['segment_duration'] = round(merged_end_abs - last_start_abs, 3)
            break

        # Calculate segment boundaries
        end_time = min(start_time + segment_duration, recording_duration)
        start_time_abs = base_start_time + start_time
        end_time_abs = base_start_time + end_time

        # Skip segments that exceed silence threshold
        if _exceeds_silence_threshold(start_time_abs, end_time_abs):
            start_time += step_size
            continue

        segment = {
            'segment_id': f"{segment_prefix}_seg{segment_idx:04d}",
            'speaker_id': speaker_id,
            'rel_filepath': rel_filepath,
            'start_time': round(start_time_abs, 3),
            'end_time': round(end_time_abs, 3),
            'segment_duration': round(end_time - start_time, 3),
            'recording_duration': recording_duration,
            **{k: v for k, v in original_row.items() if k not in ['speaker_id', 'rel_filepath', 'recording_duration']}
        }
        segments.append(segment)
        segment_idx += 1
        start_time += step_size
    
    return segments


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
            dfs.append(self.read_csv(path, sep=sep))
        
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
                    

        return combined_df.fillna(fill_value)


    @staticmethod
    def concatenate_csvs(csv_paths: Union[str, List[str]],
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
        if isinstance(csv_paths, str):
            csv_paths = [csv_paths]
        
        dfs = []
        for path in csv_paths:
            df = pd.read_csv(path, sep=sep)                
            dfs.append(df)
        
        # Concatenate all DataFrames
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Check for duplicates based on data structure.
        # - Pre-segmented data: unique by 'segment_id'
        # - VAD-split file-level data: unique by 'vad_chunk_id'
        # - Otherwise: unique by relative path
        unique_key = None
        if 'segment_id' in combined_df.columns:
            unique_key = 'segment_id'
        elif 'vad_chunk_id' in combined_df.columns:
            unique_key = 'vad_chunk_id'

        if unique_key is not None:
            duplicate_counts = combined_df[unique_key].value_counts()
            duplicates = duplicate_counts[duplicate_counts > 1]
            if not duplicates.empty:
                error_msg = f"Duplicate {unique_key} values found:\n"
                for key_val, count in duplicates.items():
                    error_msg += f"- '{key_val}' appears {count} times\n"
                raise ValueError(error_msg)
        else:
            duplicate_counts = combined_df[rel_path_col].value_counts()
            duplicates = duplicate_counts[duplicate_counts > 1]
            if not duplicates.empty:
                error_msg = "Duplicate file paths found:\n"
                for filepath, count in duplicates.items():
                    error_msg += f"- '{filepath}' appears {count} times\n"
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
                           col_id: str = DATESET_CLS.SPEAKER_ID,
                           duration_col: str = DATESET_CLS.REC_DURATION,
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
                                          class_id_col: str = DATESET_CLS.CLASS_ID) -> pd.DataFrame:
        """
        Update metadata CSV file with training_id column
        
        Args:
            df: metadata as da dataframe
            speaker_to_id: Dictionary mapping speaker IDs to training IDs
            backup: Whether to create backup of original file
        """                        
        # Add class_id column
        df_copy = df.copy()
        df_copy[class_id_col] = df_copy[id_col].map(speaker_to_id).astype(int)
        
        # Verify no missing mappings
        missing_ids = df_copy[df_copy[class_id_col].isna()][id_col].unique()
        if len(missing_ids) > 0:
            raise RuntimeWarning(f"Warning: No training ID mapping for speakers: {missing_ids}")
        
        if verbose:
            log.info(f"Total speakers: {len(df_copy[id_col].unique())}")
        
        return df_copy


    def process(self,
                dataset_files: List[str],
                spks_metadata_paths: List[str],
                speaker_id_col: str = DATESET_CLS.SPEAKER_ID,
                rel_path_col: str = DATESET_CLS.REL_FILEPATH,
                duration_col: str = DATESET_CLS.REC_DURATION,
                rounding: int = 4,
                class_id: str = DATESET_CLS.CLASS_ID,
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
                             class_id: str = DATESET_CLS.CLASS_ID) -> pd.DataFrame:
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
        try:
            waveform, sr = torchaudio.load(audio_path)
            return waveform, sr
        except Exception as e:
            msg = f"Failed to load audio file: {audio_path}. Error: {e}"
            log.error(msg)
            raise RuntimeError(msg) from e

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

    def normalize_audio(self, waveform: torch.Tensor, method: str = "rms", target_level: float = -20.0, 
                        clip_limit: float = 0.999, epsilon: float = 1e-8) -> torch.Tensor:
        """
        Normalize audio using various methods to handle different scenarios better than simple max normalization.
        
        Args:
            waveform: Input waveform tensor of shape (1, samples)
            method: Normalization method: 'peak', 'rms', 'percentile', or 'dynamic'
            target_level: Target level in dB for RMS normalization (typically -20 dB)
            clip_limit: Clipping limit for preventing excessive amplification (0-1)
            epsilon: Small value to prevent division by zero
            
        Returns:
            Normalized waveform tensor of shape (1, samples)
        """
        waveform = waveform.squeeze(0)
        assert waveform.dim() == 1, "Expected single-channel audio"
        
        # Center the waveform by removing DC offset
        waveform = waveform - waveform.mean()
        
        if method == "peak":
            # Traditional peak normalization
            peak = waveform.abs().max() + epsilon
            waveform = waveform / peak
            
        elif method == "rms":
            # RMS normalization (based on signal energy)
            rms = torch.sqrt(torch.mean(waveform ** 2))
            target_rms = 10 ** (target_level / 20)  # Convert dB to linear
            gain = target_rms / (rms + epsilon)
            waveform = waveform * gain
            
        elif method == "percentile":
            # Percentile-based normalization (robust to outliers)
            sorted_abs = torch.sort(waveform.abs())[0]
            idx = min(int(len(sorted_abs) * 0.995), len(sorted_abs) - 1)
            ref_level = sorted_abs[idx] + epsilon
            waveform = waveform / ref_level
            
        elif method == "dynamic":
            # Dynamic range compression (logarithmic compression)
            sign = torch.sign(waveform)
            abs_wave = waveform.abs() + epsilon
            compressed = sign * torch.log1p(abs_wave) / torch.log1p(torch.tensor(1.0))
            peak = compressed.abs().max() + epsilon
            waveform = compressed / peak
        
        else:
            raise ValueError(f"Unsupported normalization method: {method}")
        
        # Apply clipping to prevent excessive values
        if clip_limit < 1.0:
            waveform = torch.clamp(waveform, min=-clip_limit, max=clip_limit)
        
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
            ValueError: If path is invalid or file is empty
        """
        try:
            path = Path(audio_path).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Audio file not found: {path}")
            if path.stat().st_size == 0:
                raise ValueError(f"Audio file is empty: {path}")
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


######### Datasets #########

class BaseCollate:
    """Static collate class for batching data items."""
    def __init__(self, pad_value: float = 0.0):
        self.pad_value = pad_value
    
    def __call__(self, batch: list[DatasetItem]) -> DatasetItem:
        """
        Collate a batch of DatasetItems into a single batched DatasetItem.

        Args:
            batch: List of DatasetItem instances to be collated

        Returns:
            DatasetItem: Batched data with padded sequences and processed labels
        """
        # Unzip the batch into separate lists
        waveforms, speaker_ids, class_id, audio_paths, nationalities, genders, \
        sample_rates, recording_durations, texts = zip(
            *[(item.audio, item.speaker_id, item.class_id, item.audio_path, item.country,
               item.gender, item.sample_rate, item.recording_duration, item.text) 
              for item in batch]
        )
        
        # Process audio lengths and pad sequences
        lengths = torch.tensor([wav.shape[0] for wav in waveforms])
        padded_waveforms = pad_sequence(waveforms, batch_first=True, padding_value=self.pad_value)
        
        # Convert gender labels to numerical values
        gender_labels = torch.tensor([0.0 if gender == 'male' else 1.0 for gender in genders])
        
        return DatasetItem(
            audio=padded_waveforms,
            speaker_id=speaker_ids,
            class_id=torch.tensor(class_id),
            audio_length=lengths,
            audio_path=audio_paths,
            country=nationalities,
            gender=gender_labels,
            sample_rate=sample_rates,
            recording_duration=recording_durations,
            text=texts
        )


class BaseDataset(Dataset):
    """Static base dataset class for audio processing and feature extraction."""
    
    DATASET_CLS = BaseDatasetCols()
    
    @staticmethod
    def _calculate_max_samples(max_duration: Union[None, float, int], sample_rate: int) -> int:
        """Calculate maximum number of samples based on duration and sample rate."""
        if max_duration is None or max_duration == -1:
            return -1
        if max_duration > 0:
            return int(max_duration * sample_rate)
        raise ValueError(f"max_duration must be -1, None, or a positive number, got {max_duration}")
    
    @staticmethod
    def _load_audio(
        audio_path: Union[str, Path],
        audio_processor: AudioProcessor,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        max_samples: int = -1
    ) -> torch.Tensor:
        """
        Load and process audio with flexible segmentation options.
        
        Args:
            audio_path: Path to audio file
            audio_processor: AudioProcessor instance for processing
            start_time: Optional start time in seconds (for pre-segmentation)
            end_time: Optional end time in seconds (for pre-segmentation)
            max_samples: Maximum samples for random cropping (ignored if start_time/end_time provided)
            
        Returns:
            Processed audio waveform tensor
        """
        # Get audio info to determine original sample rate and duration
        try:
            info = torchaudio.info(str(audio_path))
        except Exception as e:
            msg = f"Failed to get audio info for file: {audio_path}. Error: {e}"
            log.error(msg)
            raise RuntimeError(msg) from e

        sr = info.sample_rate
        total_frames = info.num_frames

        assert (start_time is None and end_time is None) or (start_time is not None and end_time is not None), \
            "Either both start_time and end_time must be provided, or neither." 
        
        # Case 1: Pre-segmentation - use provided start_time and end_time
        if start_time is not None and end_time is not None:
            # Validate against actual file length to prevent psf_fseek errors
            start_frame = int(start_time * sr)
            end_frame = int(end_time * sr)
            
            # Clamp to valid range
            start_frame = min(start_frame, total_frames)
            end_frame = min(end_frame, total_frames)
            
            frame_offset = start_frame
            num_frames = max(0, end_frame - start_frame)
            
            if num_frames == 0:
                raise RuntimeError(f"Segment {start_time}-{end_time}s is outside file boundaries (length {total_frames/sr:.2f}s)")
        
        # Case 2: Random cropping - calculate start_time and end_time randomly
        else:
            if max_samples == -1 or total_frames <= max_samples:
                # Load entire file
                frame_offset = 0
                num_frames = total_frames
            else:
                # Calculate random segment of max_samples length
                max_offset = total_frames - max_samples
                frame_offset = torch.randint(0, max_offset, (1,)).item()
                num_frames = max_samples
        
        # Load the calculated segment (works for both cases)
        try:
            waveform, sr = torchaudio.load(
                str(audio_path),
                frame_offset=frame_offset,
                num_frames=num_frames
            )
        except Exception as e:
            msg = f"Failed to load audio file: {audio_path}. Error: {e}"
            log.error(msg)
            raise RuntimeError(msg) from e
        
        waveform = waveform.squeeze(0)  # Remove channel dimension
        
        # Resample if needed
        if sr != audio_processor.sample_rate:
            waveform = F.resample(waveform, sr, audio_processor.sample_rate)
        
        # Normalize
        waveform = audio_processor.normalize_audio(waveform).squeeze(0)
        
        return waveform
    
    @staticmethod
    def _load_dataset(data_filepath: Union[str, Path], sep: str = "|") -> pd.DataFrame:
        """Load and validate dataset from CSV file."""
        assert isinstance(data_filepath, (str, Path)), "data_filepath must be a string or Path"

        try:
            df = pd.read_csv(data_filepath, sep=sep)
            # append class_id columns if it does not exist
            if BaseDataset.DATASET_CLS.CLASS_ID not in df.columns:
                df[BaseDataset.DATASET_CLS.CLASS_ID] = None
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset from {data_filepath}: {str(e)}")
        
    def __init__(
        self,
        data_dir: Union[str, Path],
        data_filepath: Union[str, Path],
        sample_rate: Union[int, float],
        max_duration: Union[None, float, int] = 12.0,
        sep: str = "|",
    ):
        """
        Initialize the BaseDataset.
        
        Args:
            data_dir: Directory containing the audio files
            data_filepath: Path to the metadata CSV file
            sample_rate: Target sample rate for audio processing
            max_duration: Maximum duration of audio samples in seconds (-1 for full length)
            sep: Separator used in the metadata CSV file
        """
        self.data_dir = Path(data_dir)
        self.dataset = self._load_dataset(data_filepath, sep)
        self.audio_processor = AudioProcessor(sample_rate)
        self.max_samples = self._calculate_max_samples(max_duration, sample_rate)

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> DatasetItem:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            DatasetItem containing processed audio and metadata
        """
        row = self.dataset.iloc[idx]
        audio_path = self.data_dir / row[BaseDataset.DATASET_CLS.REL_FILEPATH]
        
        # Check if CSV has pre-segmented data with time boundaries
        has_segments = 'start_time' in self.dataset.columns and 'end_time' in self.dataset.columns
        
        if has_segments:
            # Pre-segmented: use explicit time boundaries
            waveform = self._load_audio(
                audio_path, 
                self.audio_processor,
                start_time=row['start_time'],
                end_time=row['end_time']
            )
            actual_duration = row['end_time'] - row['start_time']
        else:
            # Random cropping: use max_samples
            waveform = self._load_audio(
                audio_path,
                self.audio_processor,
                max_samples=self.max_samples
            )
            actual_duration = row[BaseDataset.DATASET_CLS.REC_DURATION].item()
        
        non_class_id = row[BaseDataset.DATASET_CLS.CLASS_ID] is None

        return DatasetItem(
            audio=waveform,
            speaker_id=row[BaseDataset.DATASET_CLS.SPEAKER_ID],
            class_id=row[BaseDataset.DATASET_CLS.CLASS_ID] if non_class_id else row[BaseDataset.DATASET_CLS.CLASS_ID].item(),
            audio_length=waveform.shape[0],
            audio_path=str(audio_path),
            country=row[BaseDataset.DATASET_CLS.NATIONALITY],
            gender=row[BaseDataset.DATASET_CLS.GENDER],
            sample_rate=self.audio_processor.sample_rate,
            recording_duration=actual_duration,
            text=row.get(BaseDataset.DATASET_CLS.TEXT, '')
        )


class SimpleAudioDataset(Dataset):
    def __init__(self, df, wav_dir, crop_len=None, sr=16000):
        self.df = df
        self.wav_dir = wav_dir
        self.crop_len = crop_len # in seconds

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        wav_path = str(self.wav_dir / self.df.iloc[idx].rel_filepath)
        try:
            waveform, sr = torchaudio.load(wav_path)
        except Exception as e:
            msg = f"Failed to load audio file: {wav_path}. Error: {e}"
            log.error(msg)
            raise RuntimeError(msg) from e
            
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


class PKSpeakerBatchSampler(Sampler[List[int]]):
    """Batch sampler that yields batches with P speakers and K utterances each.

    Each yielded element is a list of dataset indices of length P*K.

    Notes:
        - Requires the dataset to expose a pandas DataFrame on `dataset.dataset`
          containing a `speaker_id` column.
        - If a speaker has fewer than K utterances, sampling falls back to sampling
          with replacement for that speaker.
    """

    def __init__(
        self,
        dataset: Dataset,
        num_speakers: int,
        num_utterances: int,
        *,
        seed: int = 0,
        shuffle: bool = True,
        drop_last: bool = True,
        batches_per_epoch: Optional[int] = None,
        speaker_id_col: str = DATESET_CLS.SPEAKER_ID,
    ) -> None:
        if num_speakers <= 0:
            raise ValueError(f"num_speakers must be > 0, got {num_speakers}")
        if num_utterances <= 0:
            raise ValueError(f"num_utterances must be > 0, got {num_utterances}")

        self.dataset = dataset
        self.num_speakers = int(num_speakers)
        self.num_utterances = int(num_utterances)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.speaker_id_col = speaker_id_col
        self.epoch = 0

        self._indices_by_speaker = self._build_index()
        self._speakers = list(self._indices_by_speaker.keys())
        if not self._speakers:
            raise ValueError("No speakers found for PK sampling.")

        if batches_per_epoch is None:
            batch_size = self.num_speakers * self.num_utterances
            if batch_size <= 0:
                raise ValueError("Invalid batch size computed for PK sampler.")
            if self.drop_last:
                self.batches_per_epoch = len(self.dataset) // batch_size
            else:
                self.batches_per_epoch = (len(self.dataset) + batch_size - 1) // batch_size
        else:
            self.batches_per_epoch = int(batches_per_epoch)

        if self.batches_per_epoch <= 0:
            raise ValueError(
                "PKSpeakerBatchSampler has zero batches per epoch. "
                "Increase dataset size, decrease P/K, or set batches_per_epoch explicitly."
            )

    def _build_index(self) -> Dict[str, List[int]]:
        df = getattr(self.dataset, "dataset", None)
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "PKSpeakerBatchSampler requires dataset.dataset to be a pandas.DataFrame "
                f"with a '{self.speaker_id_col}' column."
            )

        if self.speaker_id_col not in df.columns:
            raise KeyError(
                f"Dataset dataframe is missing speaker id column '{self.speaker_id_col}'. "
                f"Available columns: {list(df.columns)}"
            )

        indices_by_speaker: Dict[str, List[int]] = {}
        for idx, speaker in enumerate(df[self.speaker_id_col].astype(str).tolist()):
            indices_by_speaker.setdefault(speaker, []).append(idx)
        return indices_by_speaker

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)

        speakers = self._speakers
        for _ in range(self.batches_per_epoch):
            if self.shuffle:
                if len(speakers) >= self.num_speakers:
                    batch_speakers = rng.sample(speakers, self.num_speakers)
                else:
                    batch_speakers = [rng.choice(speakers) for _ in range(self.num_speakers)]
            else:
                # Deterministic (but still cycles if not enough speakers)
                batch_speakers = [speakers[i % len(speakers)] for i in range(self.num_speakers)]

            batch_indices: List[int] = []
            for spk in batch_speakers:
                pool = self._indices_by_speaker[spk]
                if len(pool) >= self.num_utterances:
                    chosen = rng.sample(pool, self.num_utterances) if self.shuffle else pool[: self.num_utterances]
                else:
                    chosen = [rng.choice(pool) for _ in range(self.num_utterances)]
                batch_indices.extend(chosen)

            if self.shuffle:
                rng.shuffle(batch_indices)
            yield batch_indices


######### Datasets utils #########

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
        drop_last: bool = False,
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
            drop_last=drop_last,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[Any]:
        """Iterate over sampler.

        Returns:
            Iterator over elements produced by the wrapped sampler.
        """
        # Materialize the parent sampler's indices-of-indices so we can safely
        # return an iterator over *elements* of the wrapped sampler.
        #
        # Using itemgetter(*iterable) is compact but has a sharp edge:
        # if the iterable has length 1, itemgetter returns a single element,
        # and iter(single_element) would iterate that element (e.g., a list of
        # ints) instead of yielding it as one sample.
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = list(super().__iter__())
        return (self.dataset[i] for i in indexes_of_indexes)

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic shuffling and propagate to wrapped sampler.

        This is especially useful when wrapping custom samplers/batch samplers
        (e.g., PKSpeakerBatchSampler) that implement their own `set_epoch`.
        """
        super().set_epoch(epoch)
        if hasattr(self.sampler, "set_epoch"):
            try:
                self.sampler.set_epoch(epoch)
            except TypeError:
                # Some user samplers may define set_epoch without a compatible signature.
                pass


if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader
    from matplotlib import pyplot as plt

    parser = argparse.ArgumentParser(description="Generate train_list.txt for VoxCeleb")
    parser.add_argument("--data_dir", 
                        type=str,
                        default="data/librispeech",)
    parser.add_argument("--data_filepath", 
                        type=str, 
                        default="data/librispeech/metadata/dev-clean.csv")
    args = parser.parse_args() 

    print("Starting BASE Dataset test...")
    dataset = BaseDataset(data_dir=args.data_dir, data_filepath=args.data_filepath, sample_rate=16000.0)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=BaseCollate())

    for batch in dataloader:
        print(batch)
        break
    
    ap = AudioProcessor(sample_rate=batch.sample_rate[0])
    emphasized = ap.apply_preemphasis(batch.audio[0], coeff=0.09)
    reconstructed = ap.remove_preemphasis(emphasized, coeff=0.09)
    
    # Plot original and reconstructed waveforms
    plt.figure(figsize=(12, 6))
    plt.plot(batch.audio[0], label="Original waveform")
    plt.plot(emphasized.T, label="emphasized waveform")
    plt.plot(reconstructed.T, label="Reconstructed waveform")
    plt.legend()
    plt.show()