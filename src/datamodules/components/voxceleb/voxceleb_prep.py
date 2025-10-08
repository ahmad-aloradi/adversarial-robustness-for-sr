import os
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import soundfile as sf
import pandas as pd
import wget
from dataclasses import dataclass, asdict
from hydra import initialize, compose
from tqdm.auto import tqdm

from src import utils
from src.modules.components.utils import LanguagePredictionModel
from src.datamodules.components.common import get_dataset_class, get_speaker_class, VoxcelebDefaults
from src.datamodules.components.utils import segment_utterance

DATASET_DEFAULTS = VoxcelebDefaults()
DATESET_CLS, DF_COLS = get_dataset_class(DATASET_DEFAULTS.dataset_name)
SPEAKER_CLS, _ = get_speaker_class(DATASET_DEFAULTS.dataset_name)

log = utils.get_pylogger(__name__)


@dataclass
class VoxCelebUtterance:
    speaker_id: str
    rel_filepath: str
    recording_duration: float
    source: str
    split: str
    class_id: Union[int, float] = None
    dataset_name: str = DATASET_DEFAULTS.dataset_name
    sample_rate: int = DATASET_DEFAULTS.sample_rate
    language: str = DATASET_DEFAULTS.language
    gender: Optional[str] = None
    country: Optional[str] = DATASET_DEFAULTS.country
    speaker_name: Optional[str] = None
    text: Optional[str] = None


def _process_vox_wav(task_tuple: Tuple[str, Path, pd.DataFrame, str, float]):
    """Clean worker function with all data passed as arguments (no globals)."""
    wav_path_str, wav_dir, speaker_metadata_df, dataset_name, min_duration = task_tuple
    
    wav_path = Path(wav_path_str)
    
    # Fast file info retrieval
    info = sf.info(wav_path)
    
    # Early rejection for duration
    if info.duration < min_duration:
        return None
    
    # Extract speaker info
    rel_path = wav_path.relative_to(wav_dir)
    speaker_id = f"{dataset_name}_{rel_path.parts[0]}"
    
    # Get metadata from DataFrame (fast indexed lookup)
    try:
        speaker_row = speaker_metadata_df.loc[speaker_id]
        speaker_meta = speaker_row.to_dict()
    except KeyError:
        # Speaker not in metadata, use defaults
        speaker_meta = {}
    
    # Create utterance with unpacked metadata
    utt = VoxCelebUtterance(
        speaker_id=speaker_id,
        rel_filepath=str(rel_path),
        recording_duration=info.duration,
        **speaker_meta
    )
    
    return utt



class VoxCelebProcessor:
    """Process combined VoxCeleb 1 & 2 datasets and generate metadata"""
    
    METADATA_URLS = {
        'vox1': 'https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/vox1_meta.csv',
        'vox2': 'https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/vox2_meta.csv',
    }
    DATASET_PATHS = {
        'wav_dir': 'voxceleb1_2',
        'downloaded_metadata_dir': 'voxceleb_metadata/downloaded',
        'vox_metadata': 'vox_meta.csv',
        'speaker_lookup': 'speaker_lookup.csv',
        'preprocess_stats_file': 'preprocess_stats.csv'
    }

    def __init__(self, 
                 root_dir: Union[str, Path], 
                 artifcats_dir: Union[str, Path],
                 verbose: bool = True, 
                 sep: str = '|',
                 use_pre_segmentation: bool = False,
                 segment_duration: float = 4.0,
                 segment_overlap: float = 0.0,
                 min_segment_duration: float = 0.5):
        """
        Initialize VoxCeleb processor
        
        Args:
            root_dir: Root directory containing 'wav' and 'meta' subdirectories
            artifcats_dir: Directory for generated artifacts
            verbose: Print verbose output
            sep: Separator for metadata files
            use_pre_segmentation: Whether to use pre-segmentation during preprocessing
            segment_duration: Duration of each segment in seconds
            segment_overlap: Overlap between segments in seconds
            min_segment_duration: Minimum segment duration to be included in seconds
        """
        
        self.sep = sep
        self.verbose = verbose

        self.root_dir = Path(root_dir)
        self.artifcats_dir = Path(artifcats_dir)
        self.wav_dir = self.root_dir / self.DATASET_PATHS['wav_dir']
        self.downloaded_metadata_dir = self.root_dir / self.DATASET_PATHS['downloaded_metadata_dir']
        
        # Downloaded metadata files
        self.vox1_metadata = self.downloaded_metadata_dir / os.path.basename(self.METADATA_URLS['vox1'])
        self.vox2_metadata = self.downloaded_metadata_dir / os.path.basename(self.METADATA_URLS['vox2'])
        
        # Created files - now generic (not test-specific)
        self.vox_metadata = self.artifcats_dir / self.DATASET_PATHS['vox_metadata']
        self.preprocess_stats_file = self.artifcats_dir / self.DATASET_PATHS['preprocess_stats_file']
        self.dev_metadata_file = self.artifcats_dir / "voxceleb_total.csv"

        # Validate wav directory
        if not self.wav_dir.exists():
            raise FileNotFoundError(f"WAV directory not found: {self.wav_dir}")
        
        # Ensure metadata directories exists
        self.downloaded_metadata_dir.mkdir(parents=True, exist_ok=True)
        self.artifcats_dir.mkdir(parents=True, exist_ok=True)
        
        # Download basic metadata files if needed
        self._ensure_metadata_files()

        # Create or load metadata
        self.speaker_metadata_df = self.load_speaker_metadata()
        
        # Segmentation parameters (passed from config)
        self.use_pre_segmentation = use_pre_segmentation
        self.segment_duration = segment_duration
        self.segment_overlap = segment_overlap
        self.min_segment_duration = min_segment_duration


    def _download_file(self, url: str, target_path: Path) -> None:
        """Download a file from URL to target path"""
        try:
            log.info(f"Downloading {url} to {target_path}")
            wget.download(url, str(target_path))
            log.info('\n\n')  # New line after wget progress bar
        except Exception as e:
            raise RuntimeError(f"Failed to download {url}: {e}")


    def _ensure_metadata_files(self) -> None:
        """Download basic metadata files if they don't exist"""
        for dataset, url in self.METADATA_URLS.items():
            target_path = self.downloaded_metadata_dir / os.path.basename(url)
            
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
        return self._remove_white_spaces(df)


    def _process_vox2_metadata(self) -> pd.DataFrame:
        """Process VoxCeleb2 metadata"""
        df = pd.read_csv(self.vox2_metadata, sep=',', dtype='object')
        # Rename columns for consistency
        df.columns = ['speaker_id', 'vggface_id', 'gender', 'split']
        # Add missing columns
        df['nationality'] = None
        df['source'] = 'voxceleb2'
        return self._remove_white_spaces(df)


    def _create_combined_speaker_metadata(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
        """Load and merge speaker metadata from both VoxCeleb1 and VoxCeleb2"""
        vox1_df = self._process_vox1_metadata()
        vox2_df = self._process_vox2_metadata()
        df = pd.concat([vox1_df, vox2_df], ignore_index=True)
        return df


    def load_speaker_metadata(self) -> pd.DataFrame:
        """Load and merge speaker metadata from both VoxCeleb1 and VoxCeleb2
        
        Returns:
            pd.DataFrame: Combined speaker metadata with standardized columns
        """
        
        if not self.vox_metadata.exists():
            # Create combined metadata file if it doesn't exist
            df = self._create_combined_speaker_metadata()
            # Post-process speakers metadata
            df = df.rename(columns=asdict(SPEAKER_CLS))
            df[SPEAKER_CLS.speaker_id] = df[SPEAKER_CLS.speaker_id].apply(
                lambda x: DATASET_DEFAULTS.dataset_name + '_' + str(x)
            )
            df[SPEAKER_CLS.gender] = df[SPEAKER_CLS.gender].apply(
                lambda x: 'male' if x == 'm' else 'female'
            )
        else:
            # Load combined metadata
            if self.verbose:
                log.info(f"Loading metadata from {self.vox_metadata}")
            df = pd.read_csv(self.vox_metadata, sep=self.sep, dtype='object')

        if self.verbose:
            log.info(f"Loaded metadata for {len(df)} speakers")

        return df


    def _get_voxceleb_utterances(self, wav_paths: List[str], min_duration: float) -> Tuple[List['VoxCelebUtterance'], Dict]:
        """Process WAV files in parallel using ProcessPoolExecutor (clean & fast)."""
        total_files = len(wav_paths)
        
        if self.verbose:
            log.info(f"Processing {total_files:,} files...")
        
        # Set speaker_id as index for fast lookups in worker processes
        speaker_df_indexed = self.speaker_metadata_df.set_index('speaker_id')
        
        # Prepare tasks: each task contains all needed data (no globals)
        tasks = [
            (str(p), self.wav_dir, speaker_df_indexed, DATASET_DEFAULTS.dataset_name, min_duration)
            for p in wav_paths
        ]
        
        utterances: List[VoxCelebUtterance] = []
        rejected_count = 0
        
        # Use fewer workers to avoid oversubscription
        n_workers = min(cpu_count(), 8)
        
        if self.verbose:
            log.info(f"Using {n_workers} worker processes")

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all tasks at once
            future_to_path = {executor.submit(_process_vox_wav, task): task[0] for task in tasks}
            
            # Process results as they complete
            for future in tqdm(as_completed(future_to_path), total=len(tasks), desc="Processing files"):
                utterance = future.result()
                
                if utterance is not None:
                    utterances.append(utterance)
                else:
                    rejected_count += 1

        # Final statistics
        valid_count = len(utterances)
        final_stats = {
            'total': {'count': valid_count},
            'duration': {'count': rejected_count, 'paths': []}
        }
        
        if self.verbose:
            log.info(f"Processing complete: {valid_count:,} valid files, {rejected_count:,} rejected")

        return utterances, final_stats


    def generate_metadata(self,
                          base_search_dir: str,
                          min_duration: float = 1.0, 
                          save_df: bool = True,
                          ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate metadata for all valid utterances in the VoxCeleb dataset.
        This method scans through wav files in the dataset directory, processes them to extract utterance
        information, and generates metadata files containing utterance and speaker statistics.
        Args:
            min_duration (float, optional): Minimum duration in seconds for a valid utterance. Defaults to 1.0.
            save_df (bool, optional): Whether to save the generated metadata to CSV files. Defaults to True.
            base_search_dir (str): Directory to search for existing metadata files.
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                - DataFrame with development set metadata (utterances or segments)
                - DataFrame with speaker metadata
        Notes:
            - If metadata files already exist, loads and returns existing metadata instead of regenerating
            - Saves three CSV files if save_df=True:
                1. Development metadata file
                2. Preprocessing statistics file
                3. VoxCeleb metadata file
        """        

        base_search_dir = Path(base_search_dir)
        vox_metadata_file = base_search_dir / self.DATASET_PATHS['vox_metadata']
        dev_metadata_file_search_path = base_search_dir / self.dev_metadata_file.name

        if not dev_metadata_file_search_path.is_file():
            wav_paths = list(self.wav_dir.rglob("*.wav"))

            if self.verbose:
                log.info(f"Iterating over {len(wav_paths)} audio files ...")

            # Process files with progress bar
            utterances, utterances_stats = self._get_voxceleb_utterances(wav_paths, min_duration)
            utterances_stats = pd.DataFrame.from_dict(utterances_stats, orient='columns')

            # Print statistics
            if self.verbose:
                log.info(f"\nSummary: {utterances_stats['total']} files, {len(utterances)} valid")

            # Convert utterances to dataframe
            dev_metadata = self.utterances_to_csv(utterances=utterances,
                                                  dev_metadata_file=self.dev_metadata_file)
            
            # Apply segmentation only if enabled
            if self.use_pre_segmentation:
                if self.verbose:
                    log.info(f"Pre-segmentation enabled. Processing {len(dev_metadata)} utterances...")
                
                all_segments = []
                for _, row in tqdm(dev_metadata.iterrows(), total=len(dev_metadata), desc="Segmenting utterances"):
                    segments = segment_utterance(
                        speaker_id=row['speaker_id'],
                        rel_filepath=row['rel_filepath'],
                        recording_duration=row['recording_duration'],
                        segment_duration=self.segment_duration,
                        segment_overlap=self.segment_overlap,
                        min_segment_duration=self.min_segment_duration,
                        original_row=row.to_dict()
                    )
                    all_segments.extend(segments)
                
                if self.verbose:
                    log.info(f"Generated {len(all_segments)} segments from {len(dev_metadata)} utterances")
                    log.info(f"Average segments per utterance: {len(all_segments)/len(dev_metadata):.2f}")
                
                dev_metadata = pd.DataFrame(all_segments)

            if save_df:
                VoxCelebProcessor.save_csv(dev_metadata, self.dev_metadata_file, sep=self.sep)
                VoxCelebProcessor.save_csv(utterances_stats, self.preprocess_stats_file, sep=self.sep)
                VoxCelebProcessor.save_csv(self.speaker_metadata_df, self.vox_metadata, sep=self.sep)
                if self.verbose:
                    log.info(f"Saved {len(dev_metadata)} {'segments' if self.use_pre_segmentation else 'utterances'} to {self.dev_metadata_file}")
                    log.info(f"Saved speaker metadata to {self.vox_metadata}")

            VoxCelebProcessor.print_utts_statistics(utterances)

        else:
            dev_metadata = pd.read_csv(dev_metadata_file_search_path, sep=self.sep)
            assert vox_metadata_file.is_file(), f"Vox metadata file not found: {vox_metadata_file}"

            vox_metadata = pd.read_csv(vox_metadata_file, sep=self.sep)
            if not self.speaker_metadata_df.equals(vox_metadata):
                log.warning("Speaker metadata has changed since last run")
            if self.verbose:
                log.info(f"Loading existing metadata file {dev_metadata_file_search_path} with {len(dev_metadata)} total files")

            if save_df:
                VoxCelebProcessor.save_csv(vox_metadata, self.vox_metadata, sep=self.sep)
                VoxCelebProcessor.save_csv(dev_metadata, self.dev_metadata_file, sep=self.sep)
            
        return dev_metadata, self.speaker_metadata_df


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
        return df[DF_COLS]


    @staticmethod
    def save_csv(df, path, sep='|', fillna_value='N/A'):
        """Save updated metadata"""
        df = df.fillna(fillna_value)
        df.to_csv(path, sep=sep, index=False)

    @staticmethod
    def enrich_verification_file(
        veri_test_path: Optional[Union[str, Path]],
        metadata_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        sep: str = '|',
        veri_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Enrich verification test file with metadata information.
        
        Args:
            veri_test_path: Path to veri_test.txt (required if veri_df is None)
            metadata_path: Path to vox_metadata.csv
            output_path: Optional path to save the enriched CSV
            sep: Separator for CSV files
            veri_df: Optional pre-loaded verification DataFrame (avoids file I/O)
            
        Returns:
            DataFrame with enriched verification trials
        """
        
        # Read metadata and set index for fast lookups
        metadata_df = pd.read_csv(metadata_path, sep='|').fillna('N/A')
        metadata_indexed = metadata_df.set_index(DATESET_CLS.SPEAKER_ID)

        # Load verification DataFrame if not provided
        if veri_df is None:
            if veri_test_path is None:
                raise ValueError("Either veri_test_path or veri_df must be provided")
            veri_df = pd.read_csv(veri_test_path, sep=' ', header=None,
                                names=['label', 'enroll_path', 'test_path'])
        else:
            # Standardize column names if needed
            veri_df = veri_df.copy()  # Avoid modifying original
            if 'enrollment' in veri_df.columns:
                veri_df.rename(columns={'enrollment': 'enroll_path', 'test': 'test_path'}, inplace=True)

        # Extract speaker IDs from paths (vectorized)
        dataset_prefix = f"{DATASET_DEFAULTS.dataset_name}_"
        veri_df['enroll_id'] = dataset_prefix + veri_df['enroll_path'].str.split('/').str[0]
        veri_df['test_id'] = dataset_prefix + veri_df['test_path'].str.split('/').str[0]
        
        # Add metadata for both speakers (safe lookups with try/except in lambda)
        for field in [DATESET_CLS.NATIONALITY, DATESET_CLS.GENDER]:
            veri_df[f'enroll_{field}'] = veri_df['enroll_id'].apply(
                lambda x: metadata_indexed.loc[x, field] if x in metadata_indexed.index else 'N/A'
            )
            veri_df[f'test_{field}'] = veri_df['test_id'].apply(
                lambda x: metadata_indexed.loc[x, field] if x in metadata_indexed.index else 'N/A'
            )
        
        # Add trial type (same/different nationality, gender)
        veri_df[f'same_{DATESET_CLS.NATIONALITY}'] = (
            (veri_df[f'enroll_{DATESET_CLS.NATIONALITY}'] == veri_df[f'test_{DATESET_CLS.NATIONALITY}']) & 
            (veri_df[f'enroll_{DATESET_CLS.NATIONALITY}'] != 'N/A')
        ).astype(int)

        veri_df[f'same_{DATESET_CLS.GENDER}'] = (
            veri_df[f'enroll_{DATESET_CLS.GENDER}'] == veri_df[f'test_{DATESET_CLS.GENDER}']
        ).astype(int)

        # Reorder columns for clarity
        column_order = [
            'label', 
            'enroll_path', 'test_path',
            'enroll_id', 'test_id',
            f'enroll_{DATESET_CLS.GENDER}',f'test_{DATESET_CLS.GENDER}',
            f'enroll_{DATESET_CLS.NATIONALITY}', f'test_{DATESET_CLS.NATIONALITY}',
            f'same_{DATESET_CLS.GENDER}', f'same_{DATESET_CLS.NATIONALITY}'
        ]
        veri_df = veri_df[column_order]

        # Print statistics
        VoxCelebProcessor.print_test_statistics(veri_df)

        # Save if output path provided
        if output_path:
            VoxCelebProcessor.save_csv(veri_df, output_path, sep=sep)            
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
        log.info(f"Same nationality trials: {df[f'same_{DATESET_CLS.NATIONALITY}'].sum()}")


class VoxCelebTestFilter:
    """Handle test speaker exclusion after main metadata processing"""
    
    def __init__(self, root_dir: Union[str, Path], verbose: bool = True):
        """
        Initialize test filter
        
        Args:
            root_dir: Root directory containing metadata
            verbose: Print verbose output
        """
        self.root_dir = Path(root_dir)
        self.downloaded_metadata_dir = self.root_dir / VoxCelebProcessor.DATASET_PATHS['downloaded_metadata_dir']
        self.verbose = verbose
    
    def download_test_file(self, test_file: str) -> Path:
        """Download verification test file if it doesn't exist"""
        test_urls = {
            'veri_test2': 'https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt',
            'veri_test_extended2': 'https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/list_test_all2.txt', 
            'veri_test_hard2': 'https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/list_test_hard2.txt'
        }
        
        if test_file not in test_urls:
            raise ValueError(f"Unsupported test file: {test_file}. Supported options: {list(test_urls.keys())}")
        
        url = test_urls[test_file]
        target_path = self.downloaded_metadata_dir / os.path.basename(url)
        
        # Ensure the downloaded metadata directory exists
        self.downloaded_metadata_dir.mkdir(parents=True, exist_ok=True)
        
        if not target_path.exists():
            if self.verbose:
                log.info(f"Downloading test file {test_file} from {url}")
                log.info(f"Saving to: {target_path}")
            
            try:
                wget.download(url, str(target_path))
                if self.verbose:
                    log.info('\n✓ Download completed successfully\n')
            except Exception as e:
                raise RuntimeError(f"Failed to download test file {test_file} from {url}: {e}")
        else:
            if self.verbose:
                log.info(f"Test file {test_file} already exists at: {target_path}")
                
        return target_path
    
    def get_test_speakers(self, test_file: str) -> Tuple[set, pd.DataFrame]:
        """Get test speakers from verification file"""
        veri_test_path = self.download_test_file(test_file)
        
        # Read verification file
        veri_df = pd.read_csv(veri_test_path, sep=' ', header=None, names=['label', 'enrollment', 'test'])
        
        enrollment_spks = set(veri_df.enrollment.apply(lambda x: x.split(os.sep)[0]))
        test_spks = set(veri_df.test.apply(lambda x: x.split(os.sep)[0]))
        assert test_spks == enrollment_spks, "Enrollment and test speakers don't match"

        if self.verbose:
            log.info(f"Found {len(test_spks)} test speakers in {test_file}")
            
        return test_spks, veri_df
    
    def filter_dev_metadata(self, dev_metadata_df: pd.DataFrame, test_speakers: set) -> pd.DataFrame:
        """Filter out test speakers from development metadata"""
        # Extract speaker ID without dataset prefix for comparison
        dev_metadata_df['raw_speaker_id'] = dev_metadata_df[DATESET_CLS.SPEAKER_ID].apply(
            lambda x: x.replace(DATASET_DEFAULTS.dataset_name + '_', '')
        )
        
        before_count = len(dev_metadata_df)
        filtered_df = dev_metadata_df[~dev_metadata_df['raw_speaker_id'].isin(test_speakers)].copy()
        filtered_df = filtered_df.drop('raw_speaker_id', axis=1)
        
        after_count = len(filtered_df)
        excluded_count = before_count - after_count
        
        if self.verbose:
            log.info(f"Filtered out {excluded_count} utterances from {len(test_speakers)} test speakers")
            log.info(f"Remaining utterances: {after_count} (from {before_count})")
            
        return filtered_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate VoxCeleb metadata")
    parser.add_argument("--root_dir", 
                        type=str,
                        default="data/voxceleb",
                        help="Root directory containing both VoxCeleb1 and VoxCeleb2")
    parser.add_argument(
        "--artifacts_dir",
        type=str,
        default="data/voxceleb/voxceleb_metadata/metadata",
        help="Artifacts directory for generated files"
        )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output"
        )
    parser.add_argument(
        "--min_duration",
        type=float,
        default=0.5,
        help="Minimum duration in seconds. Utterances shorter than this will be excluded"
        )
    parser.add_argument(
        "--sep",
        type=str,
        default="|",
        help="Separator used for the metadata file"
        )
    parser.add_argument(
        "--test_file",
        type=str,
        choices=["veri_test2", "veri_test_extended2", "veri_test_hard2"],
        default="veri_test2",
        help="Test file name for generating test-specific files. Options: veri_test2, veri_test_extended2, veri_test_hard2",
    )
    parser.add_argument(
        "--predict_lang",
        type=bool,
        default=True,
        help="Whether to predict language for utterances"
    )

    args = parser.parse_args()

    # Normalize paths and ensure artifact directory exists
    args.root_dir = Path(args.root_dir).expanduser().resolve()
    if not args.root_dir.exists():
        raise FileNotFoundError(f"Root directory does not exist: {args.root_dir}")
    if not args.root_dir.is_dir():
        raise NotADirectoryError(f"Root directory is not a directory: {args.root_dir}")
    
    args.artifacts_dir = Path(args.artifacts_dir).expanduser().resolve()
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Run VoxCeleb Processor (now generic, no test-specific processing)
    voxceleb_processor = VoxCelebProcessor(
        root_dir=args.root_dir,
        artifcats_dir=args.artifacts_dir,
        verbose=args.verbose,
        sep=args.sep,
    )

    if not voxceleb_processor.dev_metadata_file.resolve().is_file():
        dev_metadata, speaker_metadata = voxceleb_processor.generate_metadata(
            base_search_dir=args.artifacts_dir, 
            min_duration=args.min_duration
            )

    dev_metadata = pd.read_csv(str(voxceleb_processor.dev_metadata_file.resolve()), sep=args.sep)

    if args.predict_lang:
        # Identify language of an audio file
        lang_id_cfg = {'batch_size': 64, 'num_workers': 8, 'shuffle': False, 'drop_last': False}
        lang_id_model = LanguagePredictionModel(wav_dir=voxceleb_processor.wav_dir, crop_len=8)

        dev_metadata = lang_id_model.forward(df=dev_metadata, cfg=lang_id_cfg)
        VoxCelebProcessor.save_csv(dev_metadata, str(voxceleb_processor.dev_metadata_file.resolve()), sep=args.sep)

    # Now handle test speaker exclusion separately
    test_filter = VoxCelebTestFilter(root_dir=args.root_dir, verbose=args.verbose)
    test_speakers, veri_df = test_filter.get_test_speakers(args.test_file)
    
    # Filter dev metadata to exclude test speakers
    filtered_dev_metadata = test_filter.filter_dev_metadata(dev_metadata, test_speakers)
    
    # Save filtered metadata with test-specific naming
    test_specific_dir = args.artifacts_dir / args.test_file
    test_specific_dir.mkdir(parents=True, exist_ok=True)
    
    filtered_dev_file = test_specific_dir / f"voxceleb_dev_{args.test_file}.csv"
    VoxCelebProcessor.save_csv(filtered_dev_metadata, filtered_dev_file, sep=args.sep)
    
    # Run veri_test.txt enricher
    veri_test_path = test_filter.download_test_file(args.test_file)
    output_path = test_specific_dir / f"{args.test_file}.csv"
    enriched_df = VoxCelebProcessor.enrich_verification_file(
        veri_test_path,
        voxceleb_processor.vox_metadata,
        output_path=output_path,
        sep=args.sep
        )
