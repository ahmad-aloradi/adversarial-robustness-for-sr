from typing import Tuple, Union
from pathlib import Path
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd

from src.datamodules.components.utils import AudioProcessor
from src.datamodules.components.common import get_dataset_class, LibriSpeechDefaults, DatasetItem

DATASET_DEFAULTS = LibriSpeechDefaults()
DATASET_CLS, DF_COLS = get_dataset_class(DATASET_DEFAULTS.dataset_name)


@dataclass
class LibrispeechItem(DatasetItem):
    """Single item from dataset."""
    sample_rate: float = 16000.0


class Collate():
    """Collate function for training data"""
    def __init__(self, pad_value: float = 0.0):
        self.pad_value = pad_value

    def __call__(self, batch) -> LibrispeechItem:
        waveforms, speaker_ids, audio_paths, countries, genders, sample_rates, recording_durations, audio_lengths, texts = zip(
            *[(item.audio, item.speaker_id, item.audio_path, item.country, 
               item.gender, item.sample_rate, item.recording_duration, item.audio_length, item.text) for item in batch])
        
        padded_waveforms = pad_sequence(waveforms, batch_first=True, padding_value=self.pad_value)
        gender_labels = torch.tensor([float(0) if gender == 'male' else float(1) for gender in genders])

        return LibrispeechItem(
            audio=padded_waveforms,
            speaker_id=speaker_ids,
            audio_length=audio_lengths,
            audio_path=audio_paths,
            country=countries,
            gender=gender_labels,
            sample_rate=sample_rates,
            recording_duration=recording_durations,
            text=texts
        )


class LibrispeechDataset(Dataset):
    """Initialize the Librispeech dataset.
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

    def __getitem__(self, idx) -> LibrispeechItem:
        # Retrieve data from csv
        row = self.dataset.iloc[idx]
        audio_path = row[DATASET_CLS.REL_FILEPATH]
        waveform, _ = self.audio_processor.process_audio(str(self.data_dir / audio_path))

        # Trim if longer than max_duration
        if self.max_samples != -1 and waveform.size(0) > self.max_samples:
            start = torch.randint(0, waveform.size(0) - self.max_samples, (1,))
            waveform = waveform[start:start + self.max_samples]
                
        return LibrispeechItem(
            audio=waveform,
            speaker_id=row[DATASET_CLS.SPEAKER_ID],
            audio_length=waveform.shape[0],
            audio_path=audio_path,
            country=row[DATASET_CLS.NATIONALITY],
            gender=row[DATASET_CLS.GENDER],
            sample_rate=self.audio_processor.sample_rate,
            recording_duration=row[DATASET_CLS.REC_DURATION],
            text=row[DATASET_CLS.TEXT]
        )


if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description="Generate train_list.txt for VoxCeleb")
    parser.add_argument("--librispeech_dir", 
                        type=str,
                        default="data/librispeech",)
    parser.add_argument("--data_filepath", 
                        type=str, 
                        default="data/librispeech/metadata/dev-clean.csv")
    args = parser.parse_args()    

    librispeech_data = LibrispeechDataset(data_dir=args.librispeech_dir,
                                          data_filepath=args.data_filepath,
                                          sample_rate=16000.0,)
    print("Number of samples: ", len(librispeech_data))
    print("Sample: ", librispeech_data.__getitem__(0))

    # test dataset in a dataloder
    dataloder = DataLoader(librispeech_data, batch_size=2, shuffle=True, collate_fn=Collate())
    for batch in dataloder:
        print(batch)
        break
