from pathlib import Path
from typing import List, Dict, Union
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio
import glob
from dataclasses import dataclass

from src.utils import get_pylogger

log = get_pylogger(__name__)

@dataclass
class VPCBatch:
    audio: torch.Tensor
    audio_lens: torch.Tensor
    sample_rate: List[int]
    text: List[str]
    speaker_id: List[str]
    utterance_id: List[str]
    gender: List[str]
    duration: torch.Tensor
    source_dir: List[str]

class AnonymizedLibriSpeechDataset(Dataset):
    """Dataset for anonymized LibriSpeech-like data."""
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        subset_dirs: List[str] = ['b2_system'],
        max_len: float = 8.0,
        split: str = 'train-clean-360',   # 'train-clean-360', 'dev', or 'test'
        transform=None,
        sep="|", # Separator for CSV files
        min_duration: float = 2.0, # Minimum duration in seconds,
        csv_filename: str = "combined_data.csv"
    ):
        """
        Initialize the dataset.
        
        Args:
            root_dir: Root directory containing all data
            subset_dirs: List of subset directories to include (e.g., ['b2_system', 'b5_b6_systems'])
            transform: Optional transform to be applied to audio
        """
        super().__init__()
        self.root_dir = Path(root_dir)
        self.subset_dirs = subset_dirs
        self.transform = transform
        self.sep = sep
        self.min_duration = min_duration
        self.csv_filename = csv_filename
        self.split = split
        self.max_len = max_len
        
        # Load and combine all CSV files
        self.data = self._load_all_csvs()
        
    def _load_all_csvs(self) -> pd.DataFrame:
        """Load and combine all CSV files from the specified directories."""
        all_dfs = []
        
        for subset_dir in self.subset_dirs:
            subset_path = self.root_dir / subset_dir
            if not subset_path.exists():
                log.warining(f"Directory {subset_path} does not exist")
                continue
                
            # Find all combined_data.csv files in subdirectories
            system_ext = '_'.join(subset_path.name.split('_')[1: ])
            csv_files = glob.glob(str(subset_path / f"*{self.split}_{system_ext}*" / self.csv_filename), recursive=True)
            
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file, sep=self.sep)
                    # Add source directory information
                    df['source_dir'] = Path(csv_file).parent.name
                    all_dfs.append(df)
                except Exception as e:
                    print(f"Error reading {csv_file}: {e}")
        
        if not all_dfs:
            raise RuntimeError("No valid CSV files found")
            
        # Combine all dataframes
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Drop duplicates if any
        # Add unique identifier by combining utterance_id and source_dir
        combined_df['utterance_id_unique'] = combined_df['utterance_id'] + '_' + combined_df['source_dir']
        
        # Filter out utterances with duration less than min_duration
        combined_df = combined_df[combined_df.duration > self.min_duration]
        combined_df.reset_index(drop=True, inplace=True)

        # Reorder columns to place utterance_id_unique right after utterance_id
        columns = combined_df.columns.tolist()
        columns.insert(columns.index('utterance_id') + 1, columns.pop(columns.index('utterance_id_unique')))
        combined_df = combined_df[columns]

        return combined_df

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a dataset item.
        
        Returns:
            Dictionary containing:
                - audio: tensor of audio samples
                - sample_rate: sampling rate
                - text: transcription
                - speaker_id: speaker identifier
                - utterance_id: utterance identifier
                - gender: speaker gender
                - duration: audio duration
                - source_dir: source directory name
        """
        row = self.data.iloc[idx]
        
        # Construct full path to audio file
        audio_path = self.root_dir / row['wav_path']
        
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.squeeze(0)
        max_duration_sec = int(self.max_len * sample_rate)

        if waveform.shape[-1] > self.max_len * sample_rate:
            start = torch.randint(0, waveform.shape[-1] - max_duration_sec, (1,))
            waveform = waveform[start:start + max_duration_sec]

        # Apply transform if specified
        if self.transform is not None:
            waveform = self.transform(waveform)
        
        return {
            'audio': waveform,
            'sample_rate': sample_rate,
            'text': row['text'],
            'speaker_id': row['speaker_id'],
            'utterance_id': row['utterance_id'],
            'gender': row['gender'],
            'duration': row['duration'],
            'source_dir': row['source_dir']
        }
    

class VPC25PaddingCollate:
    def __init__(self, pad_value: float = 0.0) -> None:
        """Initialize the coallate."""
        self.pad_value = pad_value

    def __call__(self, batch):
        """Collate function for DataLoader.
        
        Args:
            batch: List of dictionaries from __getitem__
        
        Returns:
            Dictionary with batched data
        """

        # Get all audio tensors and convert to shape (length, channels)
        waveforms = [item['audio'] for item in batch]
        lengths = torch.tensor([wav.shape[0] for wav in waveforms])
        padded_waveform = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True, padding_value=self.pad_value)

        return VPCBatch(
            audio=padded_waveform,
            audio_lens=lengths,
            sample_rate=([item['sample_rate'] for item in batch]),
            text=[item['text'] for item in batch],
            speaker_id=[item['speaker_id'] for item in batch],
            utterance_id=[item['utterance_id'] for item in batch],
            gender=[item['gender'] for item in batch],
            duration=torch.tensor([item['duration'] for item in batch]),
            source_dir=[item['source_dir'] for item in batch]
        )