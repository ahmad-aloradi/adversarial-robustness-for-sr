from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from tqdm import tqdm
import soundfile as sf
import pandas as pd
import wget
from dataclasses import dataclass, asdict
import os
import random

from multiprocessing import Pool, cpu_count, Manager
from tqdm.auto import tqdm
from multiprocessing.managers import ListProxy, DictProxy

@dataclass
class VoxCelebUtterance:
    utterance_id: str
    speaker_id: str
    path: str
    source: str
    duration: float
    split: str
    gender: Optional[str] = None
    language: Optional[str] = None
    nationality: Optional[str] = None


def split_dataset(
        df: pd.DataFrame,
        train_ratio: float = 0.95,
        speaker_overlap: bool = False,
        save_csv: bool = True,
        output_dir: Optional[Union[str, Path]] = None,
        sep: str = '|'
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset CSV into train and validation sets.
    
    Args:
        df: Dataset dataframe
        train_ratio: Ratio of training data from 0 to 1 (default: 0.9)
        speaker_overlap: Allow speaker overlap between sets (default: False)
        save_csv: Save train and validation CSV files (default: True)
        output_dir: Directory to save train.csv and validation.csv (default: None). 
        Only used if save_csv is True.

        
    Returns:
        Tuple of (train_df, val_df)
    """
    RANDOM_SEED = 42
    
    if speaker_overlap:
        # Random split across all samples
        shuffled_df = df.sample(frac=1.0, random_state=RANDOM_SEED)
        split_idx = int(len(shuffled_df) * train_ratio)
        train_df = shuffled_df[:split_idx]
        val_df = shuffled_df[split_idx:]
    else:
        # Split by speakers
        speakers = df['speaker_id'].unique()
        random.seed(RANDOM_SEED)
        random.shuffle(speakers)
        
        train_speakers = speakers[:int(len(speakers) * train_ratio)]
        train_df = df[df['speaker_id'].isin(train_speakers)]
        val_df = df[~df['speaker_id'].isin(train_speakers)]
    
    # Save splits
    if save_csv:
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        train_df.fillna('N/A').to_csv(output_dir / 'train.csv', index=False, sep=sep)
        val_df.fillna('N/A').to_csv(output_dir / 'validation.csv', index=False, sep=sep)
    
    return train_df, val_df

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
        'artifcats_dir': 'voxceleb_metadata/preprocessed',
        'vox_metadata': 'vox_meta.csv',
        'dev_metadata_file': "voxceleb_dev.csv",
        'preprocess_stats_file': 'preprocess_stats.csv'
    }
    OUT_COLS = ['voxceleb_id', 'vggface_id', 'gender', 'nationality', 'split']

    def __init__(self, root_dir: Union[str, Path], 
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
        self.wav_dir = self.root_dir / self.DATASET_PATHS['wav_dir']
        self.downloaded_metadata_dir = self.root_dir / self.DATASET_PATHS['downloaded_metadata_dir']
        self.artifcats_dir = self.root_dir / self.DATASET_PATHS['artifcats_dir']
        
        # Downloaded metadata files
        self.vox1_metadata = self.downloaded_metadata_dir / os.path.basename(self.METADATA_URLS['vox1'])
        self.vox2_metadata = self.downloaded_metadata_dir / os.path.basename(self.METADATA_URLS['vox2'])
        self.veri_test = self.downloaded_metadata_dir / os.path.basename(self.METADATA_URLS['test_file'])
        
        # Created files
        self.vox1_metadata_training = self.artifcats_dir / os.path.basename(self.METADATA_URLS['vox1'])
        self.vox2_metadata_training = self.artifcats_dir / os.path.basename(self.METADATA_URLS['vox2'])
        self.vox_metadata = self.artifcats_dir / self.DATASET_PATHS['vox_metadata']
        self.preprocess_stats_file = self.artifcats_dir / self.DATASET_PATHS['preprocess_stats_file']
        self.dev_metadata_file = self.artifcats_dir / self.DATASET_PATHS['dev_metadata_file']

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
        
        # Load metadata
        self.speaker_metadata = self._load_speaker_metadata()
        self.test_files = self._load_test_files()
        self.test_speakers = self._load_test_speakers(self.test_files)
        
        if self.verbose:
            print(f"Number of test files {len(self.test_files)} and test speakers {len(self.test_speakers)}")

    def _download_file(self, url: str, target_path: Path) -> None:
        """Download a file from URL to target path"""
        try:
            print(f"Downloading {url} to {target_path}")
            wget.download(url, str(target_path))
            print()  # New line after wget progress bar
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
        meta_path = self.vox1_metadata
        df = pd.read_csv(meta_path, sep='\t', dtype='object')

        # Replace spaces with underscores in column names
        df.columns = [col.replace(' ', '_') for col in df.columns]

        # Rename columns for consistency
        df.columns = self.OUT_COLS
        
        # Add source column
        df['source'] = 'voxceleb1'
        
        df = self._remove_white_spaces(df)
        return df

    def _process_vox2_metadata(self) -> pd.DataFrame:
        """Process VoxCeleb2 metadata"""
        meta_path = self.vox2_metadata
        df = pd.read_csv(meta_path, sep=',', dtype='object')
        
        # Rename columns for consistency
        df.columns = [col for col in self.OUT_COLS if col != 'nationality']
        
        # Add missing columns
        df['nationality'] = None
        df['source'] = 'voxceleb2'
        
        df = self._remove_white_spaces(df)
        return df

    def generate_training_ids(self, combined_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate training IDs from combined VoxCeleb1 and VoxCeleb2 metadata

        Args:
            metadata_files: List of paths to metadata CSV files
            
        Returns:
            Dictionary mapping original speaker IDs to numerical training IDs
            
        Example:
            {'id10001': 0, 'id10002': 1, ...}
        """        
        
        # Handle different column names in vox1 and vox2
        id_col = self.OUT_COLS[0] if self.OUT_COLS[0] in combined_df.columns else self.OUT_COLS[1]
        # Sort speakers for consistent ordering
        sorted_speakers = sorted(combined_df[id_col].unique())
        # Create mapping dictionary
        speaker_to_id = {speaker: idx for idx, speaker in enumerate(sorted_speakers)}
        
        if self.verbose:
            print(f"Generated training IDs for {len(speaker_to_id)} unique speakers")

        return speaker_to_id

    def update_metadata_with_training_ids(self,
                                          df: pd.DataFrame,
                                          speaker_to_id: Dict[str, int],
                                          id_col: str = 'voxceleb_id',
                                          ) -> pd.DataFrame:
        """
        Update metadata CSV file with training_id column
        
        Args:
            df: metadata as da dataframe
            speaker_to_id: Dictionary mapping speaker IDs to training IDs
            backup: Whether to create backup of original file
        """                        
        # Add training_id column
        df['training_id'] = df[id_col].map(speaker_to_id)
        
        # Verify no missing mappings
        missing_ids = df[df['training_id'].isna()][id_col].unique()
        if len(missing_ids) > 0:
            raise RuntimeWarning(f"Warning: No training ID mapping for speakers: {missing_ids}")
        
        if self.verbose:
            print(f"Total speakers: {len(df[id_col].unique())}")
            print(f"Training ID range: {df['training_id'].min()} - {df['training_id'].max()}")
        
        return df

    def save_csv(self, df, path):
        """Save updated metadata"""
        df = df.fillna('N/A')
        df.to_csv(path, sep=self.sep, index=False)

    def _load_speaker_metadata(self) -> Dict:
        """Load and merge speaker metadata from both VoxCeleb1 and VoxCeleb2"""
        # Create combined metadata file if it doesn't exist
        combined_path = self.vox_metadata
        
        if not combined_path.exists():
            # Process both datasets
            vox1_df = self._process_vox1_metadata()
            vox2_df = self._process_vox2_metadata()

            # Combine datasets
            df = pd.concat([vox1_df, vox2_df], ignore_index=True)

            self.speaker_to_id = self.generate_training_ids(combined_df=df)
            for df_, save_df_path in zip([df, vox1_df, vox2_df], 
                                         [combined_path, self.vox1_metadata_training,
                                          self.vox2_metadata_training]):
                # df_ = self.update_metadata_with_training_ids(df=df_, speaker_to_id=self.speaker_to_id, id_col='voxceleb_id')   
                self.save_csv(df_, save_df_path)
 
            if self.verbose:
                print(f"Saved combined metadata to {combined_path}")

        else:
            # Load combined metadata
            if self.verbose:
                print(f"Loading metadata from {combined_path}")
            df = pd.read_csv(combined_path, sep=self.sep, dtype='object')
            self.speaker_to_id = self.generate_training_ids(combined_df=df) 
        
        # Convert to dictionary format
        metadata = {}
        for _, row in df.iterrows():
            metadata[row['voxceleb_id']] = {
                'gender': row['gender'],
                'nationality': row['nationality'],
                'source': row['source'],
                'split': row['split']
            }
        
        if self.verbose:
            print(f"Loaded metadata for {len(metadata)} speakers")

        return metadata

    def _load_test_files(self) -> set:
        """Load verification test files to exclude"""
        test_files = set()
        veri_test_path = self.veri_test
        
        if not veri_test_path.exists():
            raise FileNotFoundError(f"veri_test.txt file not found: {veri_test_path}")
        
        try:
            with open(veri_test_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    assert len(parts) == 3, 'Invalid veri_test.txt structure'
                    test_files.add(parts[1])
                    test_files.add(parts[2])
            
        except Exception as e:
            raise RuntimeError(f"Error loading test files: {e}")
        
        return test_files

    def _load_test_speakers(self, test_files) -> set:
        return set(col.split(os.sep)[0] for col in test_files)

    def _generate_utterance_id(self, rel_path: Path, speaker_id: str, dataset: str) -> str:
        """Generate unique utterance ID"""
        rel_path_str = str(rel_path)
        unique_str = rel_path_str.replace(os.sep, '_').split('.')[0]
        return f"{dataset}_{speaker_id}_{unique_str}"

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
            stats = manager.dict(total = manager.dict(count=0, paths=manager.list()), 
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
           
            # Close progress bar
            pbar.close()
            
            # Convert manager.dict back to regular dict for stats
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
            stats['duration']['paths'].append(rel_path_str)
            return None
        
        # Get audio info
        info = sf.info(wav_path)
        if info.duration < min_duration:
            stats['duration']['count'] += 1
            stats['test']['paths'].append(rel_path_str)
            return None

        return VoxCelebUtterance(
            utterance_id=self._generate_utterance_id(
                rel_path, speaker_id, self.speaker_metadata[speaker_id]['source']),
                speaker_id=speaker_id,
                path=rel_path_str,
                duration=info.duration,
                **self.speaker_metadata.get(speaker_id, {})
            )

    def generate_metadata(self, min_duration: float = 1.0) -> List[VoxCelebUtterance]:
        """Generate metadata for all valid utterances"""
        
        wav_paths = list(self.wav_dir.rglob("*.wav"))

        if self.verbose:
            print(f"Iterating over {len(wav_paths)} audio files ...")

        # Process files with progress bar
        utterances, stats = self._get_voxceleb_utterances(wav_paths, min_duration)

        # Print statistics
        if self.verbose:
            print("\nProcessing Summary:")
            print(f"Total files scanned: {stats['total']}")
            print(f"Valid utterances: {len(utterances)}")
            print(f"Skipped files:")
            for reason, count in stats['skipped'].items():
                print(f"  - {reason}: {count}")

        # Save utterances ans stats as csvs
        self.save_utterances_to_csv(utterances=utterances, dev_metadata_file=self.dev_metadata_file)
        pd.DataFrame.from_dict(stats, orient='columns').to_csv(self.preprocess_stats_file, sep=args.sep, index=False)

        return utterances, stats

    def save_utterances_to_csv(self, utterances: List[VoxCelebUtterance], 
                               dev_metadata_file: Union[str, Path]) -> None:
        """
        Save list of VoxCelebUtterance objects to CSV file with pipe delimiter
        
        Args:
            utterances: List of VoxCelebUtterance objects
            dev_metadata_file: Path to save CSV file
        """
        dev_metadata_file = Path(dev_metadata_file)
        dev_metadata_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert utterances to list of dicts
        utterance_dicts = [asdict(utterance) for utterance in utterances]
        
        # Convert to DataFrame & append speaker ids
        df = pd.DataFrame(utterance_dicts)
        # df = self.update_metadata_with_training_ids(df=df, speaker_to_id=self.speaker_to_id, id_col='speaker_id') 
        self.save_csv(df, dev_metadata_file)

        if self.verbose:
            print(f"Saved {len(utterances)} utterances to {dev_metadata_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate VoxCeleb metadata")
    parser.add_argument("--root_dir", 
                        type=str,
                        default="data/voxceleb",
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
    
    # Test I
    # voxceleb_processor = VoxCelebProcessor(args.root_dir,  verbose=args.verbose, sep=args.sep)
    # utterances, stats = voxceleb_processor.generate_metadata(args.min_duration)

    # # Print statistics
    # total_vox1 = sum(1 for u in utterances if u.source == 'voxceleb1')
    # total_vox2 = sum(1 for u in utterances if u.source == 'voxceleb2')
    # print(f"\nTotal utterances: {len(utterances)}")
    # print(f"VoxCeleb1 utterances: {total_vox1}")
    # print(f"VoxCeleb2 utterances: {total_vox2}")
    # print(f"Unique speakers: {len({u.speaker_id for u in utterances})}")

    # Test II
    df = pd.read_csv("data/voxceleb/voxceleb_metadata/preprocessed/voxceleb_dev.csv", sep='|')

    train_df, val_df = split_dataset(df=df, 
                                     save_csv=True, 
                                     train_ratio = 0.95, 
                                     speaker_overlap=False,
                                     output_dir="data/voxceleb/voxceleb_metadata/preprocessed")
    
    # Print statistics
    print(f"\nDataset split statistics:")
    print(f"Training samples: {len(train_df)} ({len(train_df)/len(df):.1%})")
    print(f"Validation samples: {len(val_df)} ({len(val_df)/len(df):.1%})")
    print(f"Training speakers: {len(train_df['speaker_id'].unique())}")
    print(f"Validation speakers: {len(val_df['speaker_id'].unique())}")