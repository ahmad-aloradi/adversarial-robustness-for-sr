"""
LibriSpeech dataset preparation
"""

import os
import glob
import pandas as pd
from multiprocessing.dummy import Pool as ThreadPool
from tqdm import tqdm
import soundfile as sf
import yaml
import hydra
from hydra import initialize, compose
from omegaconf import DictConfig

from src.datamodules.components.common import get_dataset_class

DatasetCols, DF_COLS = get_dataset_class('librispeech')


def init_default_config_to_df(df, config):
    for key in config['dataset_df_cols']:
            df[key] = config['dataset_df_cols'][key]
    return df

def write_dataset_csv(df, path, sep='|', fillna_value='N/A'):
    """Save updated metadata"""
    df = df.fillna(fillna_value)
    df.to_csv(path, sep=sep, index=False)

def get_audio_based_features(audio_filepath):
    audio = sf.SoundFile(audio_filepath)
    return pd.Series([audio.samplerate,  float(len(audio)/audio.samplerate)], 
                     ['sample_rate', 'recording_duration'])

def process(config, delimiter):
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
    with open(os.path.join(config['speaker_filepath']), 'r') as f:
        lines = [line.strip() for line in f if not line.startswith(';')]

    data = []
    for line in lines:
        if line:  # Skip empty lines
            parts = line.rsplit(delimiter, 4)
            if len(parts) == 5:
                data.append([part.strip() for part in parts])

    df_speaker = pd.DataFrame(data, columns=list(config['speaker_df_cols'].keys()))    
    df_speaker = df_speaker.rename(columns=config['speaker_df_cols'])
    df_speaker['gender'] = df_speaker['gender'].apply(lambda gender: 'male' if 'M' in gender else 'female')
    df_speaker['speaker_id'] = df_speaker['speaker_id'].apply(lambda speaker_id: str(speaker_id))
    df_speaker[config['speaker_df_cols']['MINUTES']] =\
        df_speaker[config['speaker_df_cols']['MINUTES']].apply(lambda minutes: float(minutes) * 60)
    write_dataset_csv(df=df_speaker, path=os.path.join(config['metdata_path'], config['speaker_csv']))

    for subset in [config['train_dir'], config['dev_dir'], config['test_dir']]:
        # Get the list of relative audio filepaths from directory        
        audio_filepaths, listed_audio_rel_filepaths = get_audio_filepaths_from_dir(config, subset)
        annotations = list()
        speaker_ids = list()
        audio_rel_filepaths = list()
        chapter_paths = set(os.path.dirname(filepath) for filepath in audio_filepaths if subset in filepath)
        df = pd.DataFrame([], columns=config['dataset_df_cols'])

        # Iterate over chapters
        for chapter_path in tqdm(chapter_paths):
            # Get all audio files in this chapter
            chapter_audio_files = glob.glob(os.path.join(chapter_path, f"*.{config['audio_file_type']}"))
            
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
                audio_file_rel_path = os.path.relpath(audio_filepath, subset)
                _, _, speaker_id_file, chapter_id_file, utt_id = os.path.splitext(audio_file_rel_path)[0].split(os.sep)
                assert chapter_id_file == chapter_id, 'Chapter ID mismatch'
                assert speaker_id_file == speaker_id, 'Speaker ID mismatch'

                audio_rel_filepaths.append(audio_file_rel_path)
                speaker_ids.append(str(speaker_id))
                annotations.append(annotations_dict[utt_id])

        assert len(listed_audio_rel_filepaths) == len(annotations) == len(audio_rel_filepaths), \
            'Audio files and annotations don\'t match'

        # Write the information extracted within the loop into the dataframe
        df['rel_filepath'] = audio_rel_filepaths
        df['text'] = annotations
        df['speaker_id'] = speaker_ids
        # Join the speaker information to the dataframe
        df = df.join(df_speaker.set_index('speaker_id'), on='speaker_id', rsuffix='_')
        # Initialize the common parameters (config['dataset_df_cols']) into the dataframe
        df = init_default_config_to_df(df, config)
        # Add database name as prefix to avoid any future ambiguity due to non-unique speaker_ids
        df['speaker_id'] = df['speaker_id'].apply(lambda s_id: 'librispeech_' + str(s_id))

        # Add audio based features to the df (this may take a while)
        print(f'Processing audio based features... subset: {subset}')
        with ThreadPool(100) as p:
            df[['sample_rate', 'recording_duration']] =\
                p.map(get_audio_based_features,
                      df['rel_filepath'].apply(lambda rel_filepath: os.path.join(subset, rel_filepath)))

        # Re-order columns such that 'text'and 'rel_filepath' are the first two columns
        df = df[DF_COLS]

        write_dataset_csv(df=df,
                          path=os.path.join(config['metdata_path'], f'{os.path.basename(subset)}.csv')
                          )

def get_audio_filepaths_from_dir(config, subset):
    # Find all the audio and text files in the dataset directory
    audio_filepaths = [
        audio_filepath for audio_filepath in glob.glob(
            os.path.join(subset,f"*/*/*/*/*.{config['audio_file_type']}")) if os.path.isfile(audio_filepath)
            ]
    print('Total speech files found in the audio directory: {}'.format(len(audio_filepaths)))

    audio_rel_filepaths_from_dir = [os.path.relpath(audio_filepath, config['dataset_dir'])
                                    for audio_filepath in audio_filepaths]

    return audio_filepaths, audio_rel_filepaths_from_dir

def read_config(config_path):
    try:
        with open(config_path, 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except FileNotFoundError:
        raise FileNotFoundError('Config file {} does not exist'.format(config_path))

def read_hydra_config(config_path: str = "conf", config_name: str = "config") -> DictConfig:
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=config_name)
        return cfg


def main():
    config = read_hydra_config(config_path='../../../../configs/datamodule/datasets', config_name='librispeech.yaml')
    os.makedirs(config['metdata_path'], exist_ok=True)
    process(config, delimiter=config['sep'])

if __name__ == '__main__':
    main()