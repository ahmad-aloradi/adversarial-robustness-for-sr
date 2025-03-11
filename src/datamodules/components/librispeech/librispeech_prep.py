"""
LibriSpeech dataset preparation
"""

import os
from pathlib import Path
from multiprocessing.dummy import Pool as ThreadPool
from dataclasses import asdict

import yaml
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from hydra import initialize, compose
from omegaconf import DictConfig

from src.datamodules.components.common import get_dataset_class, get_speaker_class, LibriSpeechDefaults

DATASET_DEFAULTS = LibriSpeechDefaults()
DATESET_CLS, DF_COLS = get_dataset_class(DATASET_DEFAULTS.dataset_name)
SPEAKER_CLS, _ = get_speaker_class(DATASET_DEFAULTS.dataset_name)


def init_default_config_to_df(df):
    dataset_defaults = DATASET_DEFAULTS._asdict()
    for key in dataset_defaults.keys():
        df[key] = dataset_defaults[key]
    return df


def write_dataset_csv(df, path, sep='|', fillna_value='N/A'):
    """Save updated metadata"""
    df = df.fillna(fillna_value)
    df.to_csv(path, sep=sep, index=False)


def get_audio_based_features(audio_filepath, dataset_dir):
    audio = sf.SoundFile(dataset_dir + os.sep + audio_filepath)
    return pd.Series([audio.samplerate,  float(len(audio)/audio.samplerate)], [DATESET_CLS.SR, DATESET_CLS.REC_DURATION])


def process(config, delimiter, save_csv=True):
    """Generates .csv file for the dataset

    Parameters
    ----------
    df: pandas dataframe
        For the schema, see utils.get_empty_dataframe()

    config: dictionary
        Read from the corresponding .yaml config file.

    delimiter: string
        Delimiter for the .csv file.
    """
    # Read the file as raw text first
    with open(config['speaker_filepath'], 'r') as f:
        lines = [line.strip() for line in f if not line.startswith(';')]

    data = []
    for line in lines:
        if line:  # Skip empty lines
            parts = line.rsplit(delimiter, 4)
            # special case; an incorrectly labelled line
            if '|CBW|Simon' in line:
                # delete the last '|' in the line
                line = '60   | M | train-clean-100  | 20.18 | CBW Simon'
                parts = line.rsplit(delimiter, 4)
                data.append([part.strip() for part in parts])
            elif len(parts) == 5:
                data.append([part.strip() for part in parts])

    df_speaker = pd.DataFrame(data, columns=list(asdict(SPEAKER_CLS))) 
    # Post-processing speaker dataframe
    df_speaker = df_speaker.rename(columns=asdict(SPEAKER_CLS))
    df_speaker[SPEAKER_CLS.SEX] = df_speaker[SPEAKER_CLS.SEX].apply(lambda gender: 'male' if 'M' in gender else 'female')
    df_speaker[SPEAKER_CLS.ID] = df_speaker[SPEAKER_CLS.ID].apply(lambda speaker_id: DATASET_DEFAULTS.dataset_name + '_' + str(speaker_id))
    df_speaker[SPEAKER_CLS.MINUTES] = df_speaker[SPEAKER_CLS.MINUTES].apply(lambda minutes: round(float(minutes) * 60, 2))
    if save_csv:
        write_dataset_csv(df=df_speaker,path=config['speaker_csv_path'])


    dfs = {
        config['train_dir']: pd.DataFrame([], columns=list(set(DF_COLS) - set(df_speaker.columns))),
        config['dev_dir']: pd.DataFrame([], columns=list(set(DF_COLS) - set(df_speaker.columns))),
        config['test_dir']: pd.DataFrame([], columns=list(set(DF_COLS) - set(df_speaker.columns)))
        }

    for subset in dfs.keys():
        # Get the list of relative audio filepaths from directory        
        audio_filepaths, listed_audio_rel_filepaths = get_audio_filepaths_from_dir(config, subset)
        annotations = list()
        speaker_ids = list()
        audio_rel_filepaths = list()
        chapter_paths = set(os.path.dirname(filepath) for filepath in audio_filepaths if subset in filepath)

        # Iterate over chapters
        for chapter_path in tqdm(chapter_paths):
            # Get all audio files in this chapter
            chapter_audio_files = Path(chapter_path).rglob(f"*.{config['audio_file_type']}")
            
            # Get chapter text file
            chapter_id = os.path.basename(chapter_path)
            speaker_id = os.path.basename(os.path.dirname(chapter_path))
            file_ending = speaker_id + '-' + chapter_id + config['annotation_format']
            chapter_text_path = os.path.join(chapter_path, file_ending)
            
            # Read text file once for the entire chapter
            with open(chapter_text_path, 'r') as f:
                lines = f.readlines()
                # Create annotations dict for quick lookup
                annotations_dict = dict(line.strip().split(' ', 1) for line in lines)
                
            # Process each audio file in the chapter
            for audio_filepath in chapter_audio_files:
                audio_file_rel_path = os.path.join(os.path.basename(subset), os.path.relpath(audio_filepath, subset))
                _, _, _, speaker_id_file, chapter_id_file, utt_id = os.path.splitext(audio_file_rel_path)[0].split(os.sep)
                assert chapter_id_file == chapter_id, 'Chapter ID mismatch'
                assert speaker_id_file == speaker_id, 'Speaker ID mismatch'

                audio_rel_filepaths.append(audio_file_rel_path)
                speaker_ids.append(str(speaker_id))
                annotations.append(annotations_dict[utt_id])

        assert len(listed_audio_rel_filepaths) == len(annotations) == len(audio_rel_filepaths), \
            'Audio files and annotations don\'t match'

        # Write the information extracted within the loop into the dataframe
        dfs[subset][DATESET_CLS.REL_FILEPATH] = audio_rel_filepaths
        dfs[subset][DATESET_CLS.TEXT] = annotations
        dfs[subset][DATESET_CLS.SPEAKER_ID] = speaker_ids

        # Initialize the default config to the dataframe
        dfs[subset] = init_default_config_to_df(dfs[subset])
        # Add database name as prefix to avoid any future ambiguity due to non-unique speaker_ids
        dfs[subset][DATESET_CLS.SPEAKER_ID] = dfs[subset][DATESET_CLS.SPEAKER_ID].apply(lambda s_id: DATASET_DEFAULTS.dataset_name + '_' + str(s_id))
        # Join the speaker information to the dataframe
        dfs[subset] = dfs[subset].join(df_speaker.set_index(DATESET_CLS.SPEAKER_ID), on=DATESET_CLS.SPEAKER_ID, rsuffix='_')

        # Add audio based features to the df (this may take a while)
        print(f'Processing audio based features... subset: {subset}')
        with ThreadPool(100) as p:
            dfs[subset][[DATESET_CLS.SR, DATESET_CLS.REC_DURATION]] =\
                p.map(lambda x: get_audio_based_features(x, config['dataset_dir']), dfs[subset][DATESET_CLS.REL_FILEPATH])
                # p.map(get_audio_based_features, dfs[subset][DATESET_CLS.REL_FILEPATH], config['dataset_dir'])

        # Re-order columns such that 'text'and 'rel_filepath' are the first two columns
        dfs[subset] = dfs[subset][DF_COLS]

        if save_csv:
            write_dataset_csv(df=dfs[subset], path=os.path.join(config['metadata_path'], f'{os.path.basename(subset)}.csv'))
        
    return dfs, df_speaker

def get_audio_filepaths_from_dir(config, subset):
    # Find all the audio and files in the dataset directory
    subset = Path(subset)
    filepaths = [str(path) for path in subset.rglob(f"*.{config['audio_file_type']}")]
    print('Total speech files found in the audio directory: {}'.format(len(filepaths)))
    rel_filepaths_from_dir = [os.path.relpath(filepath, config['dataset_dir']) for filepath in filepaths]
    return filepaths, rel_filepaths_from_dir

def read_config(config_path):
    try:
        with open(config_path, 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except FileNotFoundError:
        raise FileNotFoundError('Config file {} does not exist'.format(config_path))


def read_hydra_config(config_path: str = "conf", config_name: str = "config", overrides: list = None) -> DictConfig:
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=config_name, overrides=overrides)
        return cfg

def generate_csvs(config, save_csv=True, delimiter="|"):
    os.makedirs(config['metadata_path'], exist_ok=True)

    if os.path.isfile(config['train_csv']) and os.path.isfile(config['dev_csv']) and os.path.isfile(config['test_csv']) and os.path.isfile(config['speaker_csv_path']):
        # load them and return them
        read_paths = [config['train_csv'], config['dev_csv'], config['test_csv']]
        write_paths = [config['train_csv_exp_filepath'], config['dev_csv_exp_filepath'], config['test_csv_exp_filepath']]
        return {write_path: pd.read_csv(read_path, sep=delimiter)
                for read_path, write_path in zip(read_paths, write_paths)}, \
               pd.read_csv(config['speaker_csv_path'], sep=delimiter)
    
    dfs, df_speaker = process(config, delimiter=config['sep'], save_csv=save_csv)
    return dfs, df_speaker


if __name__ == '__main__':
    config = read_hydra_config(config_path='../../../../configs',
                               config_name='train.yaml',
                               overrides=[
                                   f"paths.data_dir={os.environ['HOME']}/adversarial-robustness-for-sr/data",
                                   'datamodule=datasets/librispeech',
                                   f"datamodule.dataset.artifacts_dir={os.environ['HOME']}/adversarial-robustness-for-sr/data/librispeech/metadata"
                                   ])
    config = config.datamodule.dataset
    dfs, df_speaker = generate_csvs(config=config)