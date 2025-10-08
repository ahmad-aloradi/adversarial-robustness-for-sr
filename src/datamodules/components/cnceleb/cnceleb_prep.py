import os
import argparse
from pathlib import Path
from typing import Optional, Union, Tuple
from multiprocessing import cpu_count
import pandas as pd
import soundfile as sf
from dataclasses import dataclass
from tqdm.auto import tqdm
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.datamodules.components.common import get_dataset_class, get_speaker_class, CNCelebDefaults
from src.datamodules.components.utils import segment_utterance


DATASET_DEFAULTS = CNCelebDefaults()
DATASET_CLS, DF_COLS = get_dataset_class(DATASET_DEFAULTS.dataset_name)
SPEAKER_CLS, _ = get_speaker_class(DATASET_DEFAULTS.dataset_name)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Ensure at least one console handler so the user always sees logs when running standalone.
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter('[%(levelname)s] %(name)s: %(message)s'))
    logger.addHandler(_handler)

@dataclass
class CNCelebUtterance:
    speaker_id: str
    rel_filepath: str
    recording_duration: float
    split: str
    class_id: Union[int, float] = None
    dataset_name: str = DATASET_DEFAULTS.dataset_name
    sample_rate: int = DATASET_DEFAULTS.sample_rate
    language: str = DATASET_DEFAULTS.language
    gender: Optional[str] = None
    country: Optional[str] = DATASET_DEFAULTS.country
    speaker_name: Optional[str] = None
    text: Optional[str] = None


def process_audio_file(args_tuple):
    """Process a single audio file - optimized for multiprocessing."""
    (audio_path, root_dir_str, sample_rate, cnceleb1_name, cnceleb2_name) = args_tuple
    
    assert sample_rate == DATASET_DEFAULTS.sample_rate, f"Expected sample rate {DATASET_DEFAULTS.sample_rate}, got {sample_rate}"
    audio_path = Path(audio_path)
    root_dir = Path(root_dir_str)
    
    # Get audio info
    info = sf.info(str(audio_path))
        
    # Extract relative path and speaker info
    rel_path = audio_path.relative_to(root_dir)
    speaker_id = _extract_speaker_id(rel_path)
    assert speaker_id.startswith('id'), f"Invalid speaker ID extracted: {speaker_id}"

    # Determine dataset version from root directory path
    split_name = _determine_dataset_split(audio_path, cnceleb1_name, cnceleb2_name)
    
    # Create utterance data
    return {
        'speaker_id': speaker_id,
        'rel_filepath': str(root_dir.name / audio_path.relative_to(root_dir)),
        'recording_duration': info.duration,
        'split': split_name,
        'class_id': None,
        'dataset_name': DATASET_DEFAULTS.dataset_name,
        'sample_rate': info.samplerate,
        'language': DATASET_DEFAULTS.language,
        'gender': None,
        'country': DATASET_DEFAULTS.country,
        'speaker_name': None,
        'text': None
    }


def _determine_dataset_split(audio_path: Path, cnceleb1_name: str, cnceleb2_name: Optional[str] = None) -> str:
    """Determine the dataset split based on audio file path."""
    audio_path_str = str(audio_path)
    
    if cnceleb2_name and cnceleb2_name in audio_path_str:
        return cnceleb2_name
    elif cnceleb1_name in audio_path_str:
        return cnceleb1_name
    else:
        raise ValueError(f"Could not determine dataset split from path: {audio_path}")


def _extract_speaker_id(rel_path: Path) -> Optional[str]:
    """Extract speaker ID from relative path."""
    parts = rel_path.parts
    assert len(parts) >= 2, f"Invalid relative path: {rel_path}"
        
    first_part = parts[0]
    
    # Handle different directory structures
    if first_part in ['data', 'dev']:
        # data/id00010/... or dev/id00010/...
        return parts[1]
    
    elif first_part == 'eval':
        if len(parts) > 2:
            second_part = parts[1]
            if second_part == 'test':
                # eval/test/id00800-*.flac
                filename = parts[-1]
                return filename.split('-')[0] if '-' in filename else filename.split('.')[0]
            elif second_part == 'enroll':
                # eval/enroll/id00800-enroll.flac
                filename = parts[-1]
                return filename.split('-')[0] if '-' in filename else filename.split('.')[0]
            else:
                # eval/id00010/...
                return parts[1]
        else:
            # Direct file in eval/
            filename = parts[-1]
            return filename.split('-')[0] if '-' in filename else filename.split('.')[0]
    else:
        # Try to extract from filename
        filename = parts[-1]
        if filename.startswith('id') and '-' in filename:
            return filename.split('-')[0]
        elif filename.startswith('id'):
            return filename.split('.')[0]
        
        # Try first directory if it's a speaker ID
        if first_part.startswith('id'):
            return first_part
            
    raise ValueError(f"Could not extract speaker ID from relative path: {rel_path}")


class CNCelebProcessor:

    DATASET_PATHS = {
        'wav_dir': 'voxceleb1_2',
        'downloaded_metadata_dir': 'voxceleb_metadata/downloaded',
        'vox_metadata': 'vox_meta.csv',
        'speaker_lookup': 'speaker_lookup.csv',
        'preprocess_stats_file': 'preprocess_stats.csv'
    }

    """Simplified CNCeleb dataset processor."""

    def __init__(self, root_dir: Union[str, Path], artifacts_dir: Union[str, Path], 
                 cnceleb1: str, 
                 # Mandatory artifact paths for flexible configuration
                 dev_metadata_file: Union[str, Path],
                 enroll_csv_path: Union[str, Path],
                 test_unique_csv_path: Union[str, Path],
                 dev_spk_file: Union[str, Path],
                 test_spk_file: Union[str, Path],
                 cnceleb2: Optional[str] = None,
                 verbose: bool = True, sep: str = "|", 
                 sample_rate: int = 16000, n_jobs: Optional[int] = None,
                 use_pre_segmentation: bool = False,
                 segment_duration: float = 3.0,
                 segment_overlap: float = 0.0,
                 min_segment_duration: float = 0.5):
        self.sep = sep
        self.verbose = verbose
        if not self.verbose:
            logger.setLevel(logging.WARNING)
        self.sample_rate = sample_rate
        self.root_dir = Path(root_dir)
        
        # Handle cnceleb1 (mandatory) and cnceleb2 (optional)
        if cnceleb1 is None:
            raise ValueError("cnceleb1 is mandatory and cannot be None")
        
        # Build sub_datasets list from cnceleb1 and cnceleb2
        self.sub_datasets = [cnceleb1]
        if cnceleb2 is not None:
            self.sub_datasets.append(cnceleb2)

        # Store the sub-dataset roots for verification
        self.sub_dataset_roots = [self.root_dir / d for d in self.sub_datasets]
        self._verify_sub_datasets()

        self.cnceleb1_dirpath = self.root_dir / cnceleb1
        self.cnceleb2_dirpath = self.root_dir / cnceleb2 if cnceleb2 is not None else None

        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Set number of parallel jobs
        self.n_jobs = n_jobs if n_jobs is not None else min(cpu_count(), 8)
        
        # Define file paths - now all mandatory
        self.dev_metadata_file = Path(dev_metadata_file)
        self.enroll_csv_path = Path(enroll_csv_path)
        self.test_unique_csv_path = Path(test_unique_csv_path)
        self.dev_spk_file = Path(dev_spk_file)
        self.test_spk_file = Path(test_spk_file)
        
        # Trial files - look in the first sub-dataset or root
        self.enroll_map_file = self.cnceleb1_dirpath / 'eval' / 'lists' / 'enroll.map'
        self.enroll_lst_file = self.cnceleb1_dirpath / 'eval' / 'lists' / 'enroll.lst'
        self.trials_lst_file = self.cnceleb1_dirpath / 'eval' / 'lists' / 'trials.lst'
        self.test_lst_file = self.cnceleb1_dirpath / 'eval' / 'lists' / 'test.lst'
        self.cnceleb1_dev_lst_file = self.cnceleb1_dirpath / 'dev' / 'dev.lst'
        self.cnceleb2_dev_lst_file = self.cnceleb2_dirpath / 'spk.lst' if self.cnceleb2_dirpath else None

        assert self.enroll_map_file.exists(), f"Enrollment map file not found: {self.enroll_map_file}"
        assert self.enroll_lst_file.exists(), f"Enrollment list file not found: {self.enroll_lst_file}"
        assert self.trials_lst_file.exists(), f"Trials list file not found: {self.trials_lst_file}"
        assert self.cnceleb1_dev_lst_file.exists(), f"dev.lst in CN-Celeb1 not found: {self.cnceleb1_dev_lst_file}"
        if self.cnceleb2_dev_lst_file:
            assert self.cnceleb2_dev_lst_file.exists(), f"spk.lst in CN-Celeb2 not found: {self.cnceleb2_dev_lst_file}"

        logger.info(f"Initialized CNCelebProcessor:")
        logger.info(f"  Root: {self.root_dir}")
        logger.info(f"  Sub-datasets: {[str(p) for p in self.sub_dataset_roots]}")
        logger.info(f"  Artifacts: {self.artifacts_dir}")
        
        # Segmentation parameters (passed from config)
        self.use_pre_segmentation = use_pre_segmentation
        self.segment_duration = segment_duration
        self.segment_overlap = segment_overlap
        self.min_segment_duration = min_segment_duration
    
    def _verify_sub_datasets(self):
        """Verify that the specified sub-datasets exist."""
        for sub_root in self.sub_dataset_roots:
            if not sub_root.exists() or not sub_root.is_dir():
                raise FileNotFoundError(f"Sub-dataset directory does not exist: {sub_root}")
        logger.info(f"All specified sub-datasets exist: {[str(p) for p in self.sub_dataset_roots]}")

    def generate_metadata(self) -> pd.DataFrame:
        """Generate metadata by scanning dev and data directories."""
        if self.dev_metadata_file.exists():
            logger.info(f"Loading existing metadata: {self.dev_metadata_file}")
            return pd.read_csv(self.dev_metadata_file, sep=self.sep)

        all_audio_files = []
        
        # Scan all sub-datasets
        for sub_root in self.sub_dataset_roots:
            logger.info(f"Scanning: {sub_root}")
            
            # Look for audio files in common directories
            scan_dirs = [
                sub_root / 'data',
            ]
            
            for scan_dir in scan_dirs:
                audio_files = list(scan_dir.rglob('*.flac'))
                assert audio_files, f"No audio files found in {scan_dir}"
                logger.info(f"Found {len(audio_files)} audio files in {scan_dir}")
                all_audio_files.extend([(f, sub_root) for f in audio_files])

        logger.info(f"Processing {len(all_audio_files)} audio files...")
        
        # Prepare tasks for multiprocessing - include cnceleb1 and cnceleb2 names
        tasks = [(str(f), str(r), self.sample_rate, self.sub_datasets[0], 
                  self.sub_datasets[1] if len(self.sub_datasets) > 1 else None) 
                for f, r in all_audio_files]
        
        utterances = []
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            future_to_path = {executor.submit(process_audio_file, task): task[0] for task in tasks}
            
            for future in tqdm(as_completed(future_to_path), total=len(tasks), desc="Processing files"):
                result = future.result()
                assert result, f"Processing failed for: {future_to_path[future]}"
                utterances.append(result)

        # Apply segmentation only if enabled
        if self.use_pre_segmentation:
            logger.info(f"Pre-segmentation enabled. Processing {len(utterances)} utterances...")
            
            all_segments = []
            for utt in tqdm(utterances, desc="Segmenting utterances"):
                segments = segment_utterance(
                    speaker_id=utt['speaker_id'],
                    rel_filepath=utt['rel_filepath'],
                    recording_duration=utt['recording_duration'],
                    segment_duration=self.segment_duration,
                    segment_overlap=self.segment_overlap,
                    min_segment_duration=self.min_segment_duration,
                    original_row=utt
                )
                all_segments.extend(segments)
            
            logger.info(f"Generated {len(all_segments)} segments from {len(utterances)} utterances")
            logger.info(f"Average segments per utterance: {len(all_segments)/len(utterances):.2f}")
            dev_df = pd.DataFrame(all_segments)
        else:
            logger.info(f"Pre-segmentation disabled. Using full file metadata with random cropping during training.")
            dev_df = pd.DataFrame(utterances)

        dev_df.to_csv(self.dev_metadata_file, sep=self.sep, index=False)
        logger.info(f"Saved metadata for {len(dev_df)} {'segments' if self.use_pre_segmentation else 'files'} to {self.dev_metadata_file}")
        
        return dev_df
    
    def _find_audio_trial_file(self, trial_path: str) -> str:
        """Find the actual audio file path from trial path - CNCeleb specific logic."""
        # CNCeleb1 trial paths are like "test/id00800-singing-01-001.wav"
        # But actual files are in "eval/test/id00800-singing-01-001.flac"
        
        # Add .flac extension if not present
        if not trial_path.endswith('.flac'):
            trial_path_flac = Path(trial_path).with_suffix('.flac')
        else:
            trial_path_flac = Path(trial_path)
                    
        path = self.cnceleb1_dirpath / 'eval' / trial_path_flac
        assert path.exists(), f"File not found: {path}"
        rel_path = self.cnceleb1_dirpath.name / path.relative_to(self.cnceleb1_dirpath)
        return str(rel_path)

    def _find_audio_map_files(self, trial_path: str) -> str:
        """Find the actual audio file path from trial path - CNCeleb specific logic."""        
        # Add .flac extension if not present
        if not trial_path.endswith('.flac'):
            trial_path_flac = Path(trial_path).with_suffix('.flac')
        else:
            trial_path_flac = Path(trial_path)

        path = self.cnceleb1_dirpath / 'data' / trial_path_flac
        assert path.exists(), f"File not found: {path}"
        rel_path = self.cnceleb1_dirpath.name / path.relative_to(self.cnceleb1_dirpath)
        return str(rel_path)

    def generate_trial_list(self) -> pd.DataFrame:
        """Generate verification trials."""
        output_path = self.artifacts_dir / "verification_trials.csv"
        if output_path.exists():
            logger.info(f"Loading existing trials: {output_path}")
            return pd.read_csv(output_path, sep=self.sep)

        if not self.trials_lst_file.exists():
            raise FileNotFoundError(f"Trials list file not found: {self.trials_lst_file}")

        logger.info(f"Reading trials from: {self.trials_lst_file}")
        
        trials_data = []
        with open(self.trials_lst_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in tqdm(lines, desc="Processing trials"):
            line = line.strip()
            if not line or line.startswith('#'):
                raise ValueError("Empty or comment line found in trials list")
            
            parts = line.split()
            assert len(parts) == 3, f"Invalid trial format: expected 3 parts, got {len(parts)} in '{line}'"            

            enroll_id, test_path_raw, label_str = parts            
            label = int(label_str)
            actual_test_path = self._find_audio_trial_file(test_path_raw)
            
            trials_data.append({
                'label': label,
                'enroll_id': enroll_id,
                'test_path': actual_test_path
            })


        if not trials_data:
            raise ValueError("No valid trials were generated. Check trial paths and audio file locations.")

        trials_df = pd.DataFrame(trials_data)
        logger.info(f"Generated {len(trials_df)} valid trials")
        
        trials_df.to_csv(output_path, sep=self.sep, index=False)
        logger.info(f"Saved trials to {output_path}")
        
        return trials_df
    
    def generate_enrollment_embeddings_list(self) -> pd.DataFrame:
        """Generate enrollment file list using enroll.lst (simplified approach)."""
        if self.enroll_csv_path.exists():
            logger.info(f"Loading existing enrollment list: {self.enroll_csv_path}")
            return pd.read_csv(self.enroll_csv_path, sep=self.sep)

        # Try enroll.lst first (simpler format), then enroll.map        
        enrollment_lst_files = []
        enrollment_map_files = []

        logger.info(f"Reading enrollment list: {self.enroll_lst_file}")
        with open(self.enroll_lst_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                parts = line.split()
                assert len(parts) >= 2, f"Invalid enroll.lst format: {line}"

                enroll_id = parts[0]
                enroll_file = parts[1]  # Usually enroll/id00800-enroll.wav
                actual_path = self._find_audio_trial_file(enroll_file)
                enrollment_lst_files.append({
                    'enroll_id': enroll_id,
                    'enroll_path': actual_path,
                })

        logger.info(f"Reading enrollment map: {self.enroll_map_file}")
        with open(self.enroll_map_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()                        
                parts = line.split()
                assert len(parts) >= 2, f"Invalid enroll.map format: {line}"

                enroll_id = parts[0]
                actual_paths = []
                for file_path in parts[1:]:
                    actual_path = self._find_audio_map_files(file_path)
                    actual_paths.append(actual_path)
                enrollment_map_files.append({
                    'enroll_id': enroll_id,
                    'map_path': ';'.join(actual_paths)  # Convert list to semicolon-delimited string
                })

        enrollment_lst_files = pd.DataFrame.from_dict(enrollment_lst_files)
        enrollment_map_files = pd.DataFrame.from_dict(enrollment_map_files)

        enroll_df = pd.merge(enrollment_lst_files, enrollment_map_files, on='enroll_id')
        logger.info(f"Generated {len(enroll_df)} enrollment mappings")
        
        enroll_df.to_csv(self.enroll_csv_path, sep=self.sep, index=False)
        logger.info(f"Saved enrollment list to {self.enroll_csv_path}")
        return enroll_df
    
    def generate_unique_test_csv(self, trials_df: pd.DataFrame) -> None:
        """Generate unique test CSV from trials (test paths only)."""
        if self.test_unique_csv_path.exists():
            logger.info(f"Test unique CSV already exists: {self.test_unique_csv_path}")
            return
                
        # Extract unique test paths
        unique_test_paths = pd.DataFrame({'test_path': trials_df['test_path'].unique()})
        logger.info(f"Found {len(unique_test_paths)} unique test paths")
        
        unique_test_paths.to_csv(self.test_unique_csv_path, sep=self.sep, index=False)
        logger.info(f"Saved test unique CSV to {self.test_unique_csv_path}")
    
    def save_split_speaker_lists(self):
        """Save speaker lists for dev and test splits."""
        # 1. Get dev speakers from CN-Celeb_flac/dev/dev.lst if it exists
        logger.info(f"Reading dev speakers: {self.cnceleb1_dev_lst_file}")
        with open(self.cnceleb1_dev_lst_file, 'r', encoding='utf-8') as f:
            cn1_dev_speakers = [line.strip() for line in f if line.strip()]

        if self.cnceleb2_dev_lst_file and self.cnceleb2_dev_lst_file.exists():
            logger.info(f"Reading additional dev speakers from: {self.cnceleb2_dev_lst_file}")
            with open(self.cnceleb2_dev_lst_file, 'r', encoding='utf-8') as f:
                cn2_dev_speakers = [line.strip() for line in f if line.strip()]
            dev_speakers = sorted(set(cn1_dev_speakers + cn2_dev_speakers))
        else:
            dev_speakers = sorted(set(cn1_dev_speakers))
        
        logger.info(f"Total dev speakers found: {len(dev_speakers)}")

        # Test speakers from test list
        test_speakers = set()

        # 1. Get test speakers from CN-Celeb_flac/eval/lists/test.lst if it exists
        if self.test_lst_file.exists():
            logger.info(f"Reading test speakers from: {self.test_lst_file}")
            with open(self.test_lst_file, 'r', encoding='utf-8') as f:
                for line in f:
                    file_path = line.strip()
                    # Extract speaker ID from path like 'test/id00800-singing-01-001.wav'
                    filename = file_path.split(os.sep)[-1]
                    assert filename.startswith('id') and '-' in filename, f"Unexpected test file format: {filename}"
                    speaker_id = filename.split('-')[0]
                    test_speakers.add(speaker_id)
                    
        # Save speaker lists
        with open(self.dev_spk_file, 'w') as f:
            f.write('\n'.join(dev_speakers) + '\n')
        logger.info(f"Saved {len(dev_speakers)} dev speakers to {self.dev_spk_file}")

        with open(self.test_spk_file, 'w') as f:
            f.write('\n'.join(sorted(test_speakers)) + '\n')
        logger.info(f"Saved {len(test_speakers)} test speakers to {self.test_spk_file}")

    def generate_speaker_metadata(self, dev_df: pd.DataFrame) -> pd.DataFrame:
        """Generate speaker-level metadata."""
        if 'speaker_id' not in dev_df.columns:
            raise ValueError("dev_df must contain 'speaker_id' column")
        
        # # Aggregate speaker statistics
        speaker_df = dev_df.groupby('speaker_id', as_index=False).agg({
            'rel_filepath': 'count',
            'recording_duration': 'sum'
        }).rename(columns={'rel_filepath': 'num_utterances', 'recording_duration': 'total_duration'})

        # Add gender info if available
        if 'gender' in dev_df.columns:
            speaker_info = dev_df[['speaker_id', 'gender']].drop_duplicates('speaker_id')
            speaker_df = pd.merge(speaker_df, speaker_info, on='speaker_id', how='left')
        
        # Add other speaker-level info if available
        for col in ['country', 'speaker_name', 'split']:
            if col in dev_df.columns:
                speaker_info = dev_df[['speaker_id', col]].drop_duplicates('speaker_id')
                speaker_df = pd.merge(speaker_df, speaker_info, on='speaker_id', how='left')
        
        logger.info(f"Generated speaker metadata for {len(speaker_df)} speakers")
        return speaker_df
    
    def get_speaker_list(self, from_metadata: bool = True) -> set:
        """Get the set of unique speaker IDs from the dataset."""
        if from_metadata and self.dev_metadata_file.exists():
            logger.info(f"Loading speaker IDs from metadata: {self.dev_metadata_file}")
            df = pd.read_csv(self.dev_metadata_file, sep=self.sep, usecols=['speaker_id'])
            speaker_ids = set(df['speaker_id'].unique())
            logger.info(f"Found {len(speaker_ids)} unique speakers in metadata")
            return speaker_ids
        
        # Scan directory structure directly
        logger.info("Scanning directory structure for speaker IDs...")
        speaker_ids = set()
        
        # Scan multiple directories for speaker folders
        scan_dirs = [
            self.root_dir / 'data',
            self.root_dir / 'dev', 
            self.root_dir / 'eval',
        ]
        
        for scan_dir in scan_dirs:
            assert scan_dir.exists(), f"Scan directory does not exist: {scan_dir}"
            for item in scan_dir.iterdir():
                assert item.is_dir() and item.name.startswith('id'), f"Unexpected directory structure: {item}"
                speaker_ids.add(item.name)
        
        # Also extract from trial files if available
        logger.info("Extracting speaker IDs from enrollment map...")
        with open(self.enroll_map_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                assert len(parts) >= 2, f"Invalid enroll.map format: {line}"

                enroll_id = parts[0]
                if '-enroll' in enroll_id:
                    speaker_id = enroll_id.replace('-enroll', '')
                    speaker_ids.add(speaker_id)
                
                for file_path in parts[1:]:
                    path_parts = file_path.split(os.sep)
                    if path_parts and path_parts[0].startswith('id'):
                        speaker_ids.add(path_parts[0])

        if self.test_lst_file.exists():
            logger.info("Extracting speaker IDs from test list...")
            with open(self.test_lst_file, 'r') as f:
                for line in f:
                    file_path = line.strip()
                    filename = file_path.split(os.sep)[-1]

                    assert filename.startswith('id') and '-' in filename, f"Unexpected test file format: {filename}"
                    speaker_id = filename.split('-')[0]
                    speaker_ids.add(speaker_id)
        
        logger.info(f"Found {len(speaker_ids)} unique speakers from directory scan")
        return speaker_ids
    
    def _get_enrollment_speakers(self) -> set:
        """Extract unique speaker IDs from enrollment files."""
        enroll_speakers = set()
        
        # From enroll.lst
        if self.enroll_lst_file.exists():
            with open(self.enroll_lst_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    enroll_id = line.split()[0]
                    speaker_id = enroll_id.split('-')[0]
                    enroll_speakers.add(speaker_id)

        # From enroll.map
        if self.enroll_map_file.exists():
            with open(self.enroll_map_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                        
                    enroll_id = line.split()[0]
                    speaker_id = enroll_id.split('-')[0]
                    enroll_speakers.add(speaker_id)
                    
        logger.info(f"Found {len(enroll_speakers)} unique enrollment speakers.")
        return enroll_speakers
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CNCeleb metadata")
    parser.add_argument("--root_dir", type=str, default="data/cnceleb", help="Root directory for CNCeleb")
    parser.add_argument("--artifacts_dir", type=str, default="data/cnceleb/metadata", help="Artifacts directory")
    parser.add_argument("--cnceleb1", type=str, default="CN-Celeb_flac", help="CNCeleb1 dataset directory (mandatory)")
    parser.add_argument("--cnceleb2", type=str, default="CN-Celeb2_flac", help="CNCeleb2 dataset directory (optional, set to empty string to skip)")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Expected sample rate")
    parser.add_argument("--sep", type=str, default="|", help="CSV separator character")
    parser.add_argument("--verbose", action="store_true", default=True, help="Enable verbose logging")
    parser.add_argument("--n_jobs", type=int, default=None, help="Number of parallel jobs")
    args = parser.parse_args()

    # Handle optional cnceleb2 parameter
    cnceleb2 = args.cnceleb2 if args.cnceleb2 and args.cnceleb2 != "" else None
    
    # Define artifact paths based on artifacts_dir
    artifacts_dir = Path(args.artifacts_dir)
    
    processor = CNCelebProcessor(
        root_dir=args.root_dir, 
        artifacts_dir=args.artifacts_dir,
        cnceleb1=args.cnceleb1,
        dev_metadata_file=artifacts_dir / "cnceleb_dev.csv",
        enroll_csv_path=artifacts_dir / "enroll.csv",
        test_unique_csv_path=artifacts_dir / "test_unique.csv",
        dev_spk_file=artifacts_dir / "dev_speakers.txt",
        test_spk_file=artifacts_dir / "test_speakers.txt",
        cnceleb2=cnceleb2,
        verbose=args.verbose,
        sep=args.sep,
        sample_rate=args.sample_rate,
        n_jobs=args.n_jobs
    )
    
    # Run the preprocessing pipeline
    dev_metadata = processor.generate_metadata()
    trials_df = processor.generate_trial_list()
    processor.save_split_speaker_lists()
    processor.generate_enrollment_embeddings_list()
    processor.generate_unique_test_csv(trials_df)
    
    logger.info(f"Preprocessing completed. Generated {len(trials_df)} trials.")
