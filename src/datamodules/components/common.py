"""
Common constants and functions used across the datamodules.
"""

from dataclasses import dataclass, asdict, astuple
from typing import Literal


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


@dataclass(frozen=True)
class Voxceleb(BaseDatasetCols):
    SPEAKER_NAME: Literal['speaker_name'] = 'speaker_name'
    SOURCE: Literal['source'] = 'source'

@dataclass(frozen=True)
class Librispeech(BaseDatasetCols):
    SPEAKER_NAME: Literal['speaker_name'] = 'speaker_name'


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
