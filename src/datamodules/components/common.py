"""
Common constants and functions used across the datamodules.
"""
from dataclasses import dataclass, astuple
from typing import Literal, NamedTuple
import torch

#########   Base dataclasses   #########

CLASS_ID = 'class_id'

@dataclass(frozen=True)
class BaseDatasetCols:
    DATASET: Literal['dataset_name'] = 'dataset_name'
    LANGUAGE: Literal['language'] = 'language'
    NATIONALITY: Literal['country'] = 'country'
    SR: Literal['sample_rate'] = 'sample_rate'
    SPEAKER_ID: Literal['speaker_id'] = 'speaker_id'
    GENDER: Literal['gender'] = 'gender'
    SPLIT: Literal['split'] = 'split'
    REC_DURATION: Literal['recording_duration'] = 'recording_duration'
    REL_FILEPATH: Literal['rel_filepath'] = 'rel_filepath'
    TEXT: Literal['text'] = 'text'


@dataclass
class DatasetItem:
    """Single item from dataset."""
    audio: torch.Tensor
    audio_length: int
    audio_path: str
    speaker_id: int
    recording_duration: float
    gender: str
    sample_rate: int
    country: str = None
    text: str = ''

#########   Voxceleb dataclass   #########

@dataclass(frozen=True)
class Voxceleb(BaseDatasetCols):
    SPEAKER_NAME: Literal['speaker_name'] = 'speaker_name'
    SOURCE: Literal['source'] = 'source'

@dataclass(frozen=True)
class VoxcelebSpeaker:
    speaker_id: Literal['speaker_id'] = 'speaker_id'
    gender: Literal['gender'] = 'gender'
    vggface_id: Literal['speaker_name'] = 'speaker_name'
    nationality: Literal['country'] = 'country'
    split: Literal['split'] = 'split'
    source: Literal['source'] = 'source'

class VoxcelebDefaults(NamedTuple):
  dataset_name: str = 'voxceleb'
  language: str = None 
  country: str = None
  sample_rate: float = 16000


#########   LibriSpeech dataclasses   #########

@dataclass(frozen=True)
class Librispeech(BaseDatasetCols):
    SPEAKER_NAME: Literal['speaker_name'] = 'speaker_name'

@dataclass(frozen=True)
class LibrispeechSpeaker:
    ID: Literal['speaker_id'] = 'speaker_id'
    SEX: Literal['gender'] = 'gender'
    SUBSET: Literal['split'] = 'split'
    MINUTES: Literal['total_dur/spk'] = 'total_dur/spk'
    NAME: Literal['speaker_name'] = 'speaker_name'
    
class LibriSpeechDefaults(NamedTuple):
  dataset_name: str = 'librispeech'
  language: str = 'en'
  country: str = 'us'
  sample_rate: float = 16000

#########   Common functions   #########

def get_dataset_class(dataset: str):
    """
    Get the dataset-specific dataclass columns based on the dataset name.
    Args:
        dataset (str): Name of the dataset.
    Returns:
        dataclass_instance (dataclass): Dataclass instance of the dataset.
        dataset_columns (list): List of columns in the dataset.
    """
    if dataset == 'voxceleb':
        dataclass_instance =  Voxceleb()
    elif dataset == 'librispeech':
        dataclass_instance = Librispeech()
    else:
        dataclass_instance =  BaseDatasetCols()    
    return dataclass_instance, list(astuple(dataclass_instance))


def get_speaker_class(dataset: str):
    """
    Get the dataset-specific dataclass columns based on the dataset name.
    Args:
        dataset (str): Name of the dataset.
    Returns:
        dataclass_instance (dataclass): Dataclass
    """
    if dataset == 'librispeech':
        dataclass_instance = LibrispeechSpeaker()
    elif dataset == 'voxceleb':
        dataclass_instance = VoxcelebSpeaker()
    else:
        raise NotImplementedError(f"Speaker dataclass not implemented for {dataset}")
    return dataclass_instance, list(astuple(dataclass_instance))