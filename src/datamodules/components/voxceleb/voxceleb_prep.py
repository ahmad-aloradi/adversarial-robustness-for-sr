from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from tqdm import tqdm
import soundfile as sf
import pandas as pd
import wget
from dataclasses import dataclass, asdict
import os

from multiprocessing import Pool, cpu_count, Manager
from tqdm.auto import tqdm
from multiprocessing.managers import DictProxy
import sys 
sys.path.append(f"/home/aloradi/adversarial-robustness-for-sr")
from src import utils

log = utils.get_pylogger(__name__)

@dataclass
class VoxCelebUtterance:
    utterance_id: str
    speaker_id: str
    class_id: int
    path: str
    source: str
    duration: float
    split: str
    gender: Optional[str] = None
    nationality: Optional[str] = None


class VoxCelebProcessor:
    """Process combined VoxCeleb 1 & 2 datasets and generate metadata"""
    
    METADATA_URLS = {
        'vox1': 'https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/vox1_meta.csv',
        'vox2': 'https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/vox2_meta.csv',
        'test_file': 'https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test.txt'
    }
    DATASET_PATHS = {
        'wav_dir': 'voxceleb1_2',
        'downloaded_metadata_dir': 'voxceleb_metadata/downloaded',
        'vox_metadata': 'vox_meta.csv',
        'dev_metadata_file': 'voxceleb_dev.csv',
        'speaker_lookup': 'speaker_lookup.csv',
        'preprocess_stats_file': 'preprocess_stats.csv'
    }

    def __init__(self, 
                 root_dir: Union[str, Path], 
                 artifcats_dir: Union[str, Path],
                 verbose: bool = True, 
                 sep: str = '|'):
        """
        Initialize VoxCeleb processor
        
        Args:
            root_dir: Root directory containing 'wav' and 'meta' subdirectories
            verbose: Print verbose output
            sep: Separator for metadata files
        """
        self.root_dir = Path(root_dir)
        self.artifcats_dir = Path(artifcats_dir)
        self.wav_dir = self.root_dir / self.DATASET_PATHS['wav_dir']
        self.downloaded_metadata_dir = self.root_dir / self.DATASET_PATHS['downloaded_metadata_dir']
        
        # Downloaded metadata files
        self.vox1_metadata = self.downloaded_metadata_dir / os.path.basename(self.METADATA_URLS['vox1'])
        self.vox2_metadata = self.downloaded_metadata_dir / os.path.basename(self.METADATA_URLS['vox2'])
        self.veri_test = self.downloaded_metadata_dir / os.path.basename(self.METADATA_URLS['test_file'])
        
        # Created files
        self.vox_metadata = self.artifcats_dir / self.DATASET_PATHS['vox_metadata']
        self.preprocess_stats_file = self.artifcats_dir / self.DATASET_PATHS['preprocess_stats_file']
        self.dev_metadata_file = self.artifcats_dir / self.DATASET_PATHS['dev_metadata_file']
        self.speaker_lookup_file = self.artifcats_dir / self.DATASET_PATHS['speaker_lookup']

        # Validate wav directory
        if not self.wav_dir.exists():
            raise FileNotFoundError(f"WAV directory not found: {self.wav_dir}")
        
        # Ensure metadata directories exists
        self.downloaded_metadata_dir.mkdir(exist_ok=True)
        self.artifcats_dir.mkdir(exist_ok=True)
        
        # Separator for metadata files and verbose mode
        self.sep = sep
        self.verbose = verbose

        # Download metadata files if needed
        self._ensure_metadata_files()

        # Load test files and speakers
        self.test_speakers, test_df = self._load_test_files_and_spks()
        if self.verbose:
            log.info(f"Number of test files {len(test_df)} and test speakers {len(self.test_speakers)}")

        # Create or load metadata
        metadata, speaker_to_id = self.load_speaker_metadata()
        self.speaker_metadata, self.speaker_metadata_df = metadata
        self.speaker_to_id, speaker_to_id_df = speaker_to_id
        VoxCelebProcessor.save_csv(speaker_to_id_df, self.speaker_lookup_file, sep=self.sep)
        if self.verbose:
            log.info(f"Saved speaker_lookup in {self.speaker_lookup_file}")

    def _download_file(self, url: str, target_path: Path) -> None:
        """Download a file from URL to target path"""
        try:
            log.info(f"Downloading {url} to {target_path}")
            wget.download(url, str(target_path))
            log.info()  # New line after wget progress bar
        except Exception as e:
            raise RuntimeError(f"Failed to download {url}: {e}")

    def _ensure_metadata_files(self) -> None:
        """Download metadata files if they don't exist"""
        for dataset, url in self.METADATA_URLS.items():
            if dataset == 'test_file':
                target_path = self.veri_test
            else:
                target_path = self.downloaded_metadata_dir / os.path.basename(self.METADATA_URLS[dataset])
            
            if not target_path.exists():
                try:
                    self._download_file(url, target_path)
                except Exception as e:
                    raise RuntimeError(f"Failed to download metadata for {dataset}: {e}")

    def _remove_white_spaces(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove white spaces from a dataframe"""
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        df = df.replace(r'\s+', '', regex=True)
        return df

    def _process_vox1_metadata(self) -> pd.DataFrame:
        """Process VoxCeleb1 metadata"""
        df = pd.read_csv(self.vox1_metadata, sep='\t', dtype='object')

        # Replace spaces with underscores in column names
        df.columns = [col.replace(' ', '_') for col in df.columns]
        # Rename columns for consistency
        df.columns = ['speaker_id', 'vggface_id', 'gender', 'nationality', 'split']
        # Add source column
        df['source'] = 'voxceleb1'
        
        df = self._remove_white_spaces(df)
        return df

    def _process_vox2_metadata(self) -> pd.DataFrame:
        """Process VoxCeleb2 metadata"""
        df = pd.read_csv(self.vox2_metadata, sep=',', dtype='object')

        # Rename columns for consistency
        df.columns = ['speaker_id', 'vggface_id', 'gender', 'split']
        # Add missing columns
        df['nationality'] = None
        df['source'] = 'voxceleb2'
        
        df = self._remove_white_spaces(df)
        return df

    def _postprocess_vox_combined_metadata(self, df: pd.DataFrame, speaker_to_id: Dict[str, int]
                                           ) -> pd.DataFrame:
        df = self.update_metadata_with_training_ids(df=df, speaker_to_id=speaker_to_id, 
                                                    id_col='speaker_id')
        # Reorder columns to put class_id second
        cols = df.columns.tolist()
        cols.remove('class_id')
        cols.insert(1, 'class_id')
        df = df[cols]
        return df

    def generate_training_ids(self, combined_df: pd.DataFrame,id_col: str = 'speaker_id'
                              ) -> pd.DataFrame:
        """
        Generate training IDs from combined VoxCeleb1 and VoxCeleb2 metadata

        Args:
            metadata_files: List of paths to metadata CSV files
            
        Returns:
            Dictionary mapping original speaker IDs to numerical training IDs
            
        Example:
            {'id10001': 0, 'id10002': 1, ...}
        """        
        # Sort speakers for consistent ordering
        sorted_speakers = sorted(combined_df[id_col].unique())

        # Create mapping dictionary
        speaker_to_id = {speaker: idx for idx, speaker in enumerate(sorted_speakers)}
        
        if self.verbose:
            log.info(f"Generated training IDs for {len(speaker_to_id)} unique speakers")

        return speaker_to_id

    def update_metadata_with_training_ids(self, df: pd.DataFrame, speaker_to_id: Dict[str, int],
                                          id_col: str = 'speaker_id',
                                          ) -> pd.DataFrame:
        """
        Update metadata CSV file with training_id column
        
        Args:
            df: metadata as da dataframe
            speaker_to_id: Dictionary mapping speaker IDs to training IDs
            backup: Whether to create backup of original file
        """                        
        # Add class_id column
        df['class_id'] = df[id_col].map(speaker_to_id)
        
        # Verify no missing mappings
        missing_ids = df[df['class_id'].isna()][id_col].unique()
        if len(missing_ids) > 0:
            raise RuntimeWarning(f"Warning: No training ID mapping for speakers: {missing_ids}")
        
        if self.verbose:
            log.info(f"Total speakers: {len(df[id_col].unique())}")
            log.info(f"Training ID range: {df['class_id'].min()} - {df['class_id'].max()}")
        
        return df

    @staticmethod
    def save_csv(df, path, sep='|'):
        """Save updated metadata"""
        df = df.fillna('N/A')
        df.to_csv(path, sep=sep, index=False)

    def _create_combined_speaker_metadata(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
        """Load and merge speaker metadata from both VoxCeleb1 and VoxCeleb2"""
        vox1_df = self._process_vox1_metadata()
        vox2_df = self._process_vox2_metadata()
        df = pd.concat([vox1_df, vox2_df], ignore_index=True)
        # Append speaker_ids
        speaker_to_id = self.generate_training_ids(combined_df=df)
        df = self._postprocess_vox_combined_metadata(df=df, speaker_to_id=speaker_to_id)

        # Save speaker lookup table with class IDs
        speaker_to_id_df = pd.DataFrame({'speaker_id': speaker_to_id.keys(),
                                         'class_id': speaker_to_id.values()})
        return df, speaker_to_id_df, speaker_to_id

    def load_speaker_metadata(self) -> Tuple[Tuple[Dict, pd.DataFrame], Tuple[Dict, pd.DataFrame]]:
        """Load and merge speaker metadata from both VoxCeleb1 and VoxCeleb2"""
        
        if not self.vox_metadata.exists():
            # Create combined metadata file if it doesn't exist
            df, speaker_to_id_df, speaker_to_id = self._create_combined_speaker_metadata() 

        else:
            # Load combined metadata
            if self.verbose:
                log.info(f"Loading metadata from {self.vox_metadata}")
            df = pd.read_csv(self.vox_metadata, sep=self.sep, dtype='object')
            assert self.speaker_lookup_file.exists(), f"Speaker lookup file not found: {self.speaker_lookup_file}"
            speaker_to_id_df = pd.read_csv(self.speaker_lookup_file, sep=self.sep, dtype='object')
            speaker_to_id = speaker_to_id_df.set_index('speaker_id').to_dict()['class_id']
        
        # Convert to dictionary format
        metadata = {}
        for _, row in df.iterrows():
            metadata[row['speaker_id']] = {
                'gender': row['gender'],
                'nationality': row['nationality'],
                'source': row['source'],
                'split': row['split'],
                'class_id': row['class_id'],
            }
        
        if self.verbose:
            log.info(f"Loaded metadata for {len(metadata)} speakers")

        return (metadata, df), (speaker_to_id, speaker_to_id_df)

    def _load_test_files_and_spks(self) -> set:
        """Load verification test files to exclude"""
        if not self.veri_test.exists():
            raise FileNotFoundError(f"veri_test.txt file not found: {self.veri_test}")
        
        # Read verification file
        veri_df = pd.read_csv(self.veri_test, sep=' ', header=None, names=['label', 'enrollment', 'test'])
        
        enrollment_spks = set(veri_df.enrollment.apply(lambda x: x.split(os.sep)[0]))
        test_spks = set(veri_df.test.apply(lambda x: x.split(os.sep)[0]))
        assert test_spks == enrollment_spks, "Enrollment and test speakers don't match"

        return test_spks, veri_df

    def _generate_utterance_id(self, rel_path: Path, dataset: str) -> str:
        """Generate unique utterance ID"""
        rel_path_str = str(rel_path)
        unique_str = rel_path_str.replace(os.sep, '_').split('.')[0]
        return f"{dataset}_{unique_str}"

    def _init_tqdm_worker(self):
        """Disable internal tqdm bars in worker processes"""
        tqdm.set_lock(None)

    def _get_voxceleb_utterances(self, wav_paths: list, min_duration: float
                                 ) -> Tuple[List[VoxCelebUtterance], dict]:
        """
        Process WAV files with a progress bar and return processed utterances and statistics.

        This method processes a list of WAV file paths, filtering out files that don't meet
        the minimum duration requirement or are part of the speakers in the test set. It uses multiprocessing
        to speed up the processing.

        Args:
            wav_paths (list): A list of Path objects representing the WAV files to process.
            min_duration (float): The minimum duration (in seconds) for a WAV file to be included.

        Returns:
            Tuple[List[VoxCelebUtterance], dict]: A tuple containing:
                - A list of VoxCelebUtterance objects for the valid utterances.
                - A dictionary with processing statistics, including total files processed
                  and counts of skipped files (due to duration, being in test set, or errors).
        """
        total_files = len(wav_paths)
        
        # Create a progress bar
        pbar = tqdm(total=total_files, desc="Processing WAV files")
        
        # Create a manager for shared stats
        with Manager() as manager:
            stats: DictProxy = manager.dict(total = manager.dict(count=0, paths=manager.list()),
                                            duration = manager.dict(count=0, paths=manager.list()),
                                            test = manager.dict(count=0, paths=manager.list()),
                                            )
            # Define callback function for updating progress
            def update_progress(*args):
                pbar.update()
            # Create pool and process files
            with Pool(processes=cpu_count(), initializer=self._init_tqdm_worker) as pool:
                # Create async result
                async_results = [
                    pool.apply_async(
                        self._process_single_voxceleb_utterance,
                        args=(wav_path, min_duration, stats),
                        callback=update_progress
                    )
                    for wav_path in wav_paths
                ]

                # Get results
                utterances = []
                for result in async_results:
                    utterance = result.get()
                    if utterance is not None:
                        utterances.append(utterance)

                utterances = [result.get() for result in async_results if result.get() is not None]
            # Close progress bar
            pbar.close()
            
            # Convert DictProxy and ListProxy back to regular dict and lists for stats
            stats['total']['paths'] = list(stats['total']['paths'])
            stats['test']['paths'] = list(stats['test']['paths'])
            stats['duration']['paths'] = list(stats['duration']['paths'])
            final_stats = {'total': dict(stats['total']),
                           'duration': dict(stats['duration']), 
                           'test': dict(stats['test'])}

            return utterances, final_stats

    def _process_single_voxceleb_utterance(self, wav_path: Path, min_duration: float, stats: dict
                                           ) -> Optional[VoxCelebUtterance]:
        """Process a single VoxCeleb utterance file and create corresponding metadata.
        This function processes an individual WAV file from the VoxCeleb dataset, checking duration
        requirements and speaker eligibility. It creates a VoxCelebUtterance object with metadata
        if the file meets all criteria.
        Args:
            wav_path (Path): Path to the WAV file to process
            min_duration (float): Minimum required duration in seconds for valid utterances
            stats (dict): Dictionary to track processing statistics and skipped files
        Returns:
            Optional[VoxCelebUtterance]: VoxCelebUtterance object if processing successful,
                None if file should be skipped (test speaker or too short)
        """
        
        stats['total']['count'] += 1
        
        # Get relative path from wav directory
        rel_path = wav_path.relative_to(self.wav_dir)
        rel_path_str = str(rel_path)
        speaker_id = rel_path.parts[0]

        # Skip if in test files
        if speaker_id in self.test_speakers:
            stats['test']['count'] += 1
            stats['test']['paths'].append(rel_path_str)
            return None
        
        # Get audio info
        info = sf.info(wav_path)
        if info.duration < min_duration:
            stats['duration']['count'] += 1
            stats['duration']['paths'].append(rel_path_str)
            return None

        return VoxCelebUtterance(
            utterance_id=self._generate_utterance_id(rel_path, 
                                                     self.speaker_metadata[speaker_id]['source']),
                speaker_id=speaker_id,
                path=rel_path_str,
                duration=info.duration,
                **self.speaker_metadata.get(speaker_id, {})
            )

    def generate_metadata(self,
                          base_search_dir: str,
                          min_duration: float = 1.0, 
                          save_df: bool = True,
                          ) -> Tuple[Tuple[List[VoxCelebUtterance], pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame]]:
        """Generate metadata for all valid utterances in the VoxCeleb dataset.
        This method scans through wav files in the dataset directory, processes them to extract utterance
        information, and generates metadata files containing utterance and speaker statistics.
        Args:
            min_duration (float, optional): Minimum duration in seconds for a valid utterance. Defaults to 1.0.
            save_df (bool, optional): Whether to save the generated metadata to CSV files. Defaults to True.
            save_dir (str, optional): Directory to save metadata files. If None, uses default paths. Defaults to None.
        Returns:
            Tuple[Tuple[List[VoxCelebUtterance], pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame]]:
                A nested tuple containing:
                - First tuple:
                    - List of VoxCelebUtterance objects for valid utterances (None if metadata exists)
                    - DataFrame with utterance statistics (None if metadata exists)
                - Second tuple:
                    - DataFrame with development set metadata
                    - DataFrame with speaker total metadata
        Notes:
            - If metadata files already exist, loads and returns existing metadata instead of regenerating
            - Saves three CSV files if save_df=True:
                1. Development metadata file
                2. Preprocessing statistics file
                3. VoxCeleb metadata file
        """        

        base_search_dir = Path(base_search_dir)
        dev_file_location = base_search_dir / self.DATASET_PATHS['dev_metadata_file']
        vox_metadata_file = base_search_dir / self.DATASET_PATHS['vox_metadata']
        dev_metadata_file = base_search_dir / self.DATASET_PATHS['dev_metadata_file']
        speaker_lookup_file = base_search_dir / self.DATASET_PATHS['speaker_lookup']

        if not dev_file_location.is_file():
            wav_paths = list(self.wav_dir.rglob("*.wav"))

            if self.verbose:
                log.info(f"Iterating over {len(wav_paths)} audio files ...")

            # Process files with progress bar
            utterances, utterances_stats = self._get_voxceleb_utterances(wav_paths, min_duration)
            utterances_stats = pd.DataFrame.from_dict(utterances_stats, orient='columns')

            # Print statistics
            if self.verbose:
                log.info("\nProcessing Summary:")
                log.info(f"Total files scanned: {utterances_stats['total']}")
                log.info(f"Valid utterances: {len(utterances)}")
                log.info(f"Skipped files:")

            # Save utterances ans stats as csvs
            dev_metadata, speaker_total_metadata = self.utterances_to_csv(
                utterances=utterances,
                dev_metadata_file=self.dev_metadata_file)
            
            if save_df:
                VoxCelebProcessor.save_csv(dev_metadata, self.dev_metadata_file, sep=self.sep)
                VoxCelebProcessor.save_csv(utterances_stats, self.preprocess_stats_file, sep=self.sep)
                VoxCelebProcessor.save_csv(speaker_total_metadata, self.vox_metadata, sep=self.sep)

                log.info(f"Saved {len(utterances)} utterances to {self.dev_metadata_file}")

            # Print statistics
            VoxCelebProcessor.print_utts_statistics(utterances)
            return (utterances, utterances_stats), (dev_metadata, speaker_total_metadata)

        else:
            assert vox_metadata_file.is_file(), f"Vox metadata file not found: {vox_metadata_file}"
            assert speaker_lookup_file.is_file(), f"Speaker lookup file not found: {speaker_lookup_file}"
            assert vox_metadata_file.is_file(), f"Vox metadata file not found: {vox_metadata_file}"

            vox_metadata = pd.read_csv(vox_metadata_file, sep=self.sep)
            dev_metadata = pd.read_csv(dev_metadata_file, sep=self.sep)
            speaker_lookup = pd.read_csv(speaker_lookup_file, sep=self.sep)

            if self.verbose:
                log.info(f"Loading existing metadata file {dev_metadata_file} with {len(dev_metadata)} total files")

            if save_df:
                VoxCelebProcessor.save_csv(vox_metadata, self.vox_metadata, sep=self.sep)
                VoxCelebProcessor.save_csv(dev_metadata, self.dev_metadata_file, sep=self.sep)
                VoxCelebProcessor.save_csv(speaker_lookup, self.speaker_lookup_file, sep=self.sep)

            return (None, None), (dev_metadata, speaker_lookup)

    def utterances_to_csv(self, utterances: List[VoxCelebUtterance],
                          dev_metadata_file: Union[str, Path]) -> None:
        """
        Save list of VoxCelebUtterance objects to CSV file with pipe delimiter
        
        Args:
            utterances: List of VoxCelebUtterance objects
            dev_metadata_file: Path to save CSV file
        """
        dev_metadata_file = Path(dev_metadata_file)
        dev_metadata_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert utterances to list of dicts, then to a DataFrame
        utterance_dicts = [asdict(utterance) for utterance in utterances]
        df = pd.DataFrame(utterance_dicts)

        # Append speaker ids and their durations stats
        df = self.update_metadata_with_training_ids(df=df, speaker_to_id=self.speaker_to_id, id_col='speaker_id')
        
        # Append speaker stats and Sort by total duration per speaker
        speaker_stats = self._get_speakers_stats(df, col_id='speaker_id')
        df_with_stats = self._append_speaker_stats(df, speaker_stats, col_id='speaker_id')
        speaker_metadata_with_stats = self._append_speaker_stats(self.speaker_metadata_df, speaker_stats, col_id='speaker_id')      
        
        return df_with_stats, speaker_metadata_with_stats
    
    def _append_speaker_stats(self, df: pd.DataFrame, 
                              speaker_stats: pd.DataFrame, 
                              col_id: str = 'speaker_id') -> pd.DataFrame:
        """Append speaker stats to metadata"""
        df = df.merge(speaker_stats, on=col_id, how='left')
        df = df.sort_values('total_dur/spk', ascending=False)
        return df

    def _get_speakers_stats(self, df: pd.DataFrame, col_id: str = 'speaker_id') -> pd.DataFrame:
        speaker_stats = df.groupby(col_id).agg(
            {'duration': ['sum', 'mean', 'count']}).round(4)

        speaker_stats.columns = pd.MultiIndex.from_tuples([
            ('duration', 'total_dur/spk'),
            ('duration', 'mean_dur/spk'),
            ('duration', 'utterances/spk')
        ])
        speaker_stats.columns = speaker_stats.columns.get_level_values(1)
        # Reset index to make speaker_id a column
        speaker_stats = speaker_stats.reset_index()
        return speaker_stats


    @staticmethod
    def enrich_verification_file(
        veri_test_path: Union[str, Path],
        metadata_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        sep: str = '|'
    ) -> pd.DataFrame:
        """
        Enrich verification test file with metadata information.
        
        Args:
            veri_test_path: Path to veri_test.txt
            metadata_path: Path to vox_metadata.csv
            output_path: Optional path to save the enriched CSV
            
        Returns:
            DataFrame with enriched verification trials
        """
        # Read metadata
        metadata_df = pd.read_csv(metadata_path, sep='|').fillna('N/A')

        # Create lookup dictionaries for faster access
        metadata_lookup = metadata_df.set_index('speaker_id').to_dict('index')

        # Read verification file
        veri_df = pd.read_csv(veri_test_path, sep=' ', header=None,
                            names=['label', 'enroll_path', 'test_path'])

        # Extract speaker IDs from paths
        veri_df['enroll_id'] = veri_df['enroll_path'].apply(lambda x: x.split('/')[0])
        veri_df['test_id'] = veri_df['test_path'].apply(lambda x: x.split('/')[0])
        
        # Add metadata for both speakers
        for field in ['nationality', 'gender', 'class_id']:
            veri_df[f'enroll_{field}'] = veri_df['enroll_id'].map(
                lambda x: metadata_lookup[x][field] if x in metadata_lookup else 'N/A'
            )
            veri_df[f'test_{field}'] = veri_df['test_id'].map(
                lambda x: metadata_lookup[x][field] if x in metadata_lookup else 'N/A'
            )
        
        # Add trial type (same/different nationality, gender)
        veri_df['same_nationality'] = (
            (veri_df['enroll_nationality'] == veri_df['test_nationality']) & 
            (veri_df['enroll_nationality'] != 'N/A')
        ).astype(int)

        veri_df['same_gender'] = (
            veri_df['enroll_gender'] == veri_df['test_gender']
        ).astype(int)

        # Reorder columns for clarity
        column_order = [
            'label', 
            'enroll_path', 'test_path',
            'enroll_id', 'test_id',
            'enroll_gender', 'test_gender',
            'enroll_nationality', 'test_nationality',
            'enroll_class_id', 'test_class_id',
            'same_gender', 'same_nationality'
        ]
        veri_df = veri_df[column_order]

        # Print statistics
        VoxCelebProcessor.print_test_statistics(veri_df)

        # Save if output path provided
        if output_path:
            VoxCelebProcessor.save_csv(veri_df, output_path, sep=sep)            
        else:
            return veri_df


    @staticmethod
    def print_utts_statistics(utterances: List[VoxCelebUtterance]) -> None:
        total_vox1 = sum(1 for u in utterances if u.source == 'voxceleb1')
        total_vox2 = sum(1 for u in utterances if u.source == 'voxceleb2')
        log.info(f"\nTotal utterances: {len(utterances)}")
        log.info(f"VoxCeleb1 utterances: {total_vox1}")
        log.info(f"VoxCeleb2 utterances: {total_vox2}")
        log.info(f"Unique speakers: {len({u.speaker_id for u in utterances})}")

    @staticmethod
    def print_test_statistics(df: pd.DataFrame) -> None:
        log.info(f"Total trials: {len(df)}")
        log.info(f"Positive trials: {(df['label'] == 1).sum()}")
        log.info(f"Same gender trials: {df['same_gender'].sum()}")
        log.info(f"Same nationality trials: {df['same_nationality'].sum()}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate VoxCeleb metadata")
    parser.add_argument("--root_dir", 
                        type=str,
                        default="data/voxceleb",
                        help="Root directory containing both VoxCeleb1 and VoxCeleb2")
    parser.add_argument("--artifacts_dir", 
                    type=str,
                    default="data/voxceleb/voxceleb_metadata/preprocessed",
                    help="Root directory containing both VoxCeleb1 and VoxCeleb2")
    parser.add_argument("--verbose",
                        action="store_true",
                        help="Print verbose output")
    parser.add_argument("--preprocess_stats_file", 
                        type=str,
                        default="data/voxceleb/voxceleb_metadata/preprocess_metadata/preprocess_stats.csv")
    parser.add_argument("--min_duration",
                        type=float,
                        default=0.5, 
                        help="Minimum duration in seconds. Utterances shorter than this will be excluded")
    parser.add_argument("--sep", 
                        type=str,
                        default="|",
                        help="Separator used for the metadata file")
    
    args = parser.parse_args()
    
    # Run Voxceleb Processor
    voxceleb_processor = VoxCelebProcessor(args.root_dir,
                                           artifcats_dir=args.artifacts_dir, 
                                           verbose=args.verbose, sep=args.sep)
    utterances_and_stats, utterances_dataframes = voxceleb_processor.generate_metadata(args.min_duration)
    utterances, utterances_stats = utterances_and_stats
    dev_metadata, speaker_total_metadata = utterances_dataframes

     # Run veri_test.txt enricher
    output_path = 'data/voxceleb/voxceleb_metadata/preprocessed/veri_test_rich.csv' 
    if os.path.exists(output_path):
        log.info(f"Output file already exists: {output_path}")
        enriched_df = pd.read_csv(output_path, sep='|')
    else:
        enriched_df = VoxCelebProcessor.enrich_verification_file(
            'data/voxceleb/voxceleb_metadata/downloaded/veri_test.txt',
            'data/voxceleb/voxceleb_metadata/preprocessed/vox_meta.csv',
            output_path=None
            )
