import os
from pathlib import Path
from typing import Any, Dict, Optional, Union, Tuple, Set
from multiprocessing import cpu_count
import pandas as pd
import soundfile as sf
from dataclasses import dataclass
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from src import utils
from hydra.utils import instantiate
from src.datamodules.components.common import get_dataset_class, get_speaker_class, CNCelebDefaults
from src.datamodules.components.utils import segment_utterance
from src.datamodules.preparation.base import (
    build_config_snapshot_from_mapping,
    read_hydra_config,
    save_config_snapshot,
)
from src.datamodules.preparation.snapshot_keys import CNCELEB_COMPARABLE_KEYS


DATASET_DEFAULTS = CNCelebDefaults()
DATASET_CLS, DF_COLS = get_dataset_class(DATASET_DEFAULTS.dataset_name)
SPEAKER_CLS, _ = get_speaker_class(DATASET_DEFAULTS.dataset_name)

log = utils.get_pylogger(__name__)

@dataclass
class CNCelebUtterance:
    speaker_id: str
    rel_filepath: str
    recording_duration: float
    source: str  # Dataset origin: CN-Celeb_flac, CN-Celeb2_flac, etc.
    split: str   # Actual split: data, dev, eval, etc.
    class_id: Union[int, float] = None
    dataset_name: str = DATASET_DEFAULTS.dataset_name
    sample_rate: int = DATASET_DEFAULTS.sample_rate
    language: str = DATASET_DEFAULTS.language
    gender: Optional[str] = None
    country: Optional[str] = DATASET_DEFAULTS.country
    speaker_name: Optional[str] = None
    text: Optional[str] = None


def write_dataset_csv(df, path, sep='|', fillna_value='N/A'):
    """Save updated metadata"""
    df = df.fillna(fillna_value)
    df.to_csv(path, sep=sep, index=False)


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
    speaker_id = _format_global_speaker_id(speaker_id)

    # Determine dataset source and actual split
    source_name = _determine_source(audio_path, cnceleb1_name, cnceleb2_name)
    split_name = _determine_split(rel_path)
    
    # Create utterance data
    return {
        'speaker_id': speaker_id,
        'rel_filepath': str(Path(root_dir.name) / rel_path),
        'recording_duration': info.duration,
        'source': source_name,
        'split': split_name,
        'class_id': None,
        'dataset_name': DATASET_DEFAULTS.dataset_name,
        'sample_rate': info.samplerate,
        'language': DATASET_DEFAULTS.language,
        'gender': None,
        'country': DATASET_DEFAULTS.country,
        'speaker_name': None,
        'text': None,
        'is_concatenated': False
    }


def _determine_source(audio_path: Path, cnceleb1_name: str, cnceleb2_name: Optional[str] = None) -> str:
    """Determine the dataset source (CN-Celeb_flac, CN-Celeb2_flac) based on audio file path."""
    audio_path_str = str(audio_path)
    
    if cnceleb2_name and cnceleb2_name in audio_path_str:
        return cnceleb2_name
    elif cnceleb1_name in audio_path_str:
        return cnceleb1_name
    else:
        raise ValueError(f"Could not determine dataset source from path: {audio_path}")


def _determine_split(rel_path: Path) -> str:
    """Determine the actual split (data, dev, eval) from relative path."""
    parts = rel_path.parts
    if len(parts) >= 1:
        first_part = parts[0]
        if first_part in ['data', 'dev', 'eval']:
            return first_part
    return 'unknown'  # Default to 'data' if not determinable


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


def _format_global_speaker_id(local_id: str) -> str:
    """Attach dataset prefix to a speaker identifier if missing."""
    prefix = f"{DATASET_DEFAULTS.dataset_name}_"
    return local_id if local_id.startswith(prefix) else f"{prefix}{local_id}"


def _ensure_prefixed_identifier(raw_id: str) -> str:
    """Apply dataset prefix while preserving any suffix (e.g., '-enroll')."""
    if raw_id.startswith(f"{DATASET_DEFAULTS.dataset_name}_"):
        return raw_id

    if '-' in raw_id:
        speaker_token, remainder = raw_id.split('-', 1)
        return f"{_format_global_speaker_id(speaker_token)}-{remainder}"

    return _format_global_speaker_id(raw_id)


class CNCelebProcessor:
    """CNCeleb dataset processor."""

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
                 use_pre_segmentation: bool = True,
                 segment_duration: float = 3.0,
                 segment_overlap: float = 0.0,
                 min_segment_duration: float = 0.5,
                 vad: Any = None,
                 concat_mapping_file: Optional[Union[str, Path]] = None):
        self.sep = sep
        self.verbose = verbose
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

        log.info(f"Initialized CNCelebProcessor:")
        log.info(f"  Root: {self.root_dir}")
        log.info(f"  Sub-datasets: {[str(p) for p in self.sub_dataset_roots]}")
        log.info(f"  Artifacts: {self.artifacts_dir}")
        
        # Segmentation parameters (passed from config)
        self.use_pre_segmentation = use_pre_segmentation
        self.segment_duration = segment_duration
        self.segment_overlap = segment_overlap
        self.min_segment_duration = min_segment_duration

        # Optional VAD config (Hydra-instantiated when used)
        self.vad_cfg = vad
        
        # Load concatenation mapping if provided (CSV format: output_path|source_paths|duration|speaker_id|original_split)
        self.concat_mapping_df: Optional[pd.DataFrame] = None
        self.merged_source_files: Set[str] = set()
        self.concat_mapping_file = Path(concat_mapping_file) if concat_mapping_file else None
        if self.concat_mapping_file is not None and not self.concat_mapping_file.exists():
            raise FileNotFoundError(f"Concatenation mapping file not found: {self.concat_mapping_file}")
        
        log.info(f"Loading concatenation mapping from: {self.concat_mapping_file}")
        self.concat_mapping_df = pd.read_csv(self.concat_mapping_file, sep='|')
        # Build set of merged source files from semicolon-delimited source_paths column
        for source_paths in self.concat_mapping_df['source_paths']:
            for path in source_paths.split(';'):
                self.merged_source_files.add(path)
        log.info(f"Loaded mapping with {len(self.concat_mapping_df)} concatenated files")
        log.info(f"Will skip {len(self.merged_source_files)} merged source files")
    
    def _verify_sub_datasets(self):
        """Verify that the specified sub-datasets exist."""
        for sub_root in self.sub_dataset_roots:
            if not sub_root.exists() or not sub_root.is_dir():
                raise FileNotFoundError(f"Sub-dataset directory does not exist: {sub_root}")
        log.info(f"All specified sub-datasets exist: {[str(p) for p in self.sub_dataset_roots]}")
    
    def _collect_concatenated_utterances(self) -> list:
        """
        Collect utterances from concatenated files based on the mapping CSV.
        
        Returns:
            List of utterance dictionaries for concatenated files
        """
        concat_utterances = []
        
        if self.concat_mapping_df is None or len(self.concat_mapping_df) == 0:
            return concat_utterances
        
        log.info(f"Adding {len(self.concat_mapping_df)} concatenated files from mapping...")
        
        for _, row in self.concat_mapping_df.iterrows():
            concat_path = self.root_dir / row['output_path']
            if not concat_path.exists():
                log.warning(f"Concatenated file not found: {concat_path}")
                continue
            
            # Use dataset_source to preserve CN-Celeb1/CN-Celeb2 origin as 'source'
            # Fall back to first source file's path if dataset_source not in mapping
            original_source = row.get('dataset_source')
            if pd.isna(original_source) and row.get('source_paths'):
                # Extract from first source path: "CN-Celeb_flac/data/..." -> "CN-Celeb_flac"
                first_source = row['source_paths'].split(';')[0]
                original_source = first_source.split('/')[0] if '/' in first_source else None
            
            concat_utt = {
                'speaker_id': _format_global_speaker_id(row['speaker_id']),
                'rel_filepath': row['output_path'],
                'recording_duration': row['duration'],
                'source': original_source,  # Dataset origin: CN-Celeb_flac or CN-Celeb2_flac
                'split': 'data',  # Concatenated files are training data
                'class_id': None,
                'dataset_name': DATASET_DEFAULTS.dataset_name,
                'sample_rate': self.sample_rate,
                'language': DATASET_DEFAULTS.language,
                'gender': None,
                'country': DATASET_DEFAULTS.country,
                'speaker_name': None,
                'text': None,
                'is_concatenated': True,
            }
            concat_utterances.append(concat_utt)
        
        log.info(f"Added {len(concat_utterances)} concatenated utterances")
        return concat_utterances

    def generate_metadata(self) -> pd.DataFrame:
        """Generate metadata by scanning dev and data directories.
        
        If a concatenation mapping is provided, this method will:
        1. Skip original source files that have been merged into concatenated files
        2. Include the concatenated files in processing
        3. Apply VAD to concatenated files (as per requirement)
        """
        all_audio_files = []
        skipped_merged_count = 0
        
        # Scan all sub-datasets
        for sub_root in self.sub_dataset_roots:
            log.info(f"Scanning: {sub_root}")
            
            # Look for audio files in common directories
            scan_dirs = [
                sub_root / 'data',
            ]
            
            for scan_dir in scan_dirs:
                audio_files = list(scan_dir.rglob('*.flac'))
                assert audio_files, f"No audio files found in {scan_dir}"
                log.info(f"Found {len(audio_files)} audio files in {scan_dir}")
                
                for f in audio_files:
                    # Build relative path as stored in the mapping (sub_dataset/rel_path)
                    rel_to_root = f.relative_to(self.root_dir)
                    rel_path_str = str(rel_to_root)
                    
                    # Skip files that have been merged into concatenated files
                    if rel_path_str in self.merged_source_files:
                        skipped_merged_count += 1
                        continue
                    
                    all_audio_files.extend([(f, sub_root)])
        
        if self.merged_source_files:
            log.info(f"Skipped {skipped_merged_count} files that have been merged into concatenated utterances")
        
        # Add concatenated files if mapping exists
        concat_utterances = self._collect_concatenated_utterances()

        log.info(f"Processing {len(all_audio_files)} original audio files...")
        
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
        
        # Merge original utterances with concatenated utterances
        utterances.extend(concat_utterances)
        log.info(f"Total utterances (original + concatenated): {len(utterances)}")

        # Optional VAD (applied at file-level / before segmentation)
        # VAD is applied to ALL utterances including concatenated ones
        vad_cfg = self.vad_cfg
        vad = instantiate(vad_cfg) if vad_cfg and getattr(vad_cfg, '_target_', None) else None
        if vad is not None and vad.should_apply('dev'):
            utterances = vad.apply(
                utterances,
                audio_root=self.root_dir,
                split_name='dev',
                rel_filepath_key='rel_filepath',
                recording_duration_key='recording_duration',
                skip_list_path=self.artifacts_dir / f"vad_skipped_dev.txt",
            )

        # Apply segmentation only if enabled
        if self.use_pre_segmentation:
            log.info(f"Pre-segmentation enabled. Processing {len(utterances)} utterances...")
            
            all_segments = []
            skipped_utterances = 0
            for utt in tqdm(utterances, desc="Segmenting utterances"):
                segments = segment_utterance(
                    speaker_id=utt['speaker_id'],
                    rel_filepath=utt['rel_filepath'],
                    recording_duration=utt['recording_duration'],
                    segment_duration=self.segment_duration,
                    segment_overlap=self.segment_overlap,
                    min_segment_duration=self.min_segment_duration,
                    original_row=utt,
                    base_start_time=float(utt.get('vad_start', 0.0) or 0.0),
                    vad_speech_timestamps=utt.get('vad_speech_timestamps', None),
                    segment_max_silence_ratio=float(getattr(vad, 'segment_max_silence_ratio', 0.80)) if vad is not None and vad.enabled else None,
                )
                if not segments:
                    skipped_utterances += 1
                else:
                    all_segments.extend(segments)

            # Compute skipped utts percentage and average segment length (seconds)
            total_utterances = len(utterances)
            skipped_pct = (skipped_utterances / total_utterances) * 100 if total_utterances > 0 else 0.0
            avg_segment_length = sum(s.get('segment_duration', 0.0) for s in all_segments) / len(all_segments)

            log.info(f"Generated {len(all_segments)} segments from {total_utterances} utterances")
            log.info(f"Average segments per utterance: {len(all_segments)/total_utterances:.2f}")
            log.info(f"Average segment duration: {avg_segment_length:.3f}s")
            log.info(f"Skipped {skipped_utterances} utterances ({skipped_pct:.2f}%) due to duration < min_segment_duration={self.min_segment_duration}s or silence filtering")
            dev_df = pd.DataFrame(all_segments)
        else:
            log.info(f"Pre-segmentation disabled. Using full file metadata with random cropping during training.")
            dev_df = pd.DataFrame(utterances)

        # Ensure split speaker lists exist (dev/test) and filter out any test speakers
        try:
            # This will populate self.test_spk_file (and dev_spk_file) based on dataset lists
            self.save_split_speaker_lists()
            if self.test_spk_file.exists():
                with open(self.test_spk_file, 'r', encoding='utf-8') as f:
                    test_speakers = {line.strip() for line in f if line.strip()}
                if test_speakers:
                    before = len(dev_df)
                    dev_df = dev_df[~dev_df['speaker_id'].isin(test_speakers)]
                    log.info(f"Filtered {before - len(dev_df)} rows from dev metadata belonging to test speakers")
        except Exception as e:
            log.warning(f"Could not filter test speakers from dev metadata: {e}")

        write_dataset_csv(dev_df, self.dev_metadata_file, sep=self.sep)
        log.info(f"Saved metadata for {len(dev_df)} {'segments' if self.use_pre_segmentation else 'files'} to {self.dev_metadata_file}")

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
        if not self.trials_lst_file.exists():
            raise FileNotFoundError(f"Trials list file not found: {self.trials_lst_file}")

        log.info(f"Reading trials from: {self.trials_lst_file}")
        
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
            enroll_id = _ensure_prefixed_identifier(enroll_id)
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
        log.info(f"Generated {len(trials_df)} valid trials")

        trials_df.to_csv(output_path, sep=self.sep, index=False)
        log.info(f"Saved trials to {output_path}")

        return trials_df

    def generate_enrollment_embeddings_list(self) -> pd.DataFrame:
        """Generate enrollment file list using enroll.lst (simplified approach)."""
        # Try enroll.lst first (simpler format), then enroll.map        
        enrollment_lst_files = []
        enrollment_map_files = []

        log.info(f"Reading enrollment list: {self.enroll_lst_file}")
        with open(self.enroll_lst_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                parts = line.split()
                assert len(parts) >= 2, f"Invalid enroll.lst format: {line}"

                enroll_id = _ensure_prefixed_identifier(parts[0])
                enroll_file = parts[1]  # Usually enroll/id00800-enroll.wav
                actual_path = self._find_audio_trial_file(enroll_file)
                enrollment_lst_files.append({
                    'enroll_id': enroll_id,
                    'enroll_path': actual_path,
                })

        log.info(f"Reading enrollment map: {self.enroll_map_file}")
        with open(self.enroll_map_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                parts = line.split()
                assert len(parts) >= 2, f"Invalid enroll.map format: {line}"

                enroll_id = _ensure_prefixed_identifier(parts[0])
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
        log.info(f"Generated {len(enroll_df)} enrollment mappings")
        
        enroll_df.to_csv(self.enroll_csv_path, sep=self.sep, index=False)
        log.info(f"Saved enrollment list to {self.enroll_csv_path}")
        return enroll_df
    
    def generate_unique_test_csv(self, trials_df: pd.DataFrame) -> None:
        """Generate unique test CSV from trials (test paths only)."""
        unique_test_paths = pd.DataFrame({'test_path': trials_df['test_path'].unique()})
        log.info(f"Found {len(unique_test_paths)} unique test paths")

        unique_test_paths.to_csv(self.test_unique_csv_path, sep=self.sep, index=False)
        log.info(f"Saved test unique CSV to {self.test_unique_csv_path}")
    
    def save_split_speaker_lists(self):
        """Save speaker lists for dev and test splits."""
        # 1. Get dev speakers from CN-Celeb_flac/dev/dev.lst if it exists
        log.info(f"Reading dev speakers: {self.cnceleb1_dev_lst_file}")
        with open(self.cnceleb1_dev_lst_file, 'r', encoding='utf-8') as f:
            cn1_dev_speakers = [_format_global_speaker_id(line.strip()) for line in f if line.strip()]

        if self.cnceleb2_dev_lst_file and self.cnceleb2_dev_lst_file.exists():
            log.info(f"Reading additional dev speakers from: {self.cnceleb2_dev_lst_file}")
            with open(self.cnceleb2_dev_lst_file, 'r', encoding='utf-8') as f:
                cn2_dev_speakers = [_format_global_speaker_id(line.strip()) for line in f if line.strip()]
            dev_speakers = sorted(set(cn1_dev_speakers + cn2_dev_speakers))
        else:
            dev_speakers = sorted(set(cn1_dev_speakers))
        
        log.info(f"Total dev speakers found: {len(dev_speakers)}")

        # Test speakers from test list
        test_speakers = set()

        # 1. Get test speakers from CN-Celeb_flac/eval/lists/test.lst if it exists
        if self.test_lst_file.exists():
            log.info(f"Reading test speakers from: {self.test_lst_file}")
            with open(self.test_lst_file, 'r', encoding='utf-8') as f:
                for line in f:
                    file_path = line.strip()
                    # Extract speaker ID from path like 'test/id00800-singing-01-001.wav'
                    filename = file_path.split(os.sep)[-1]
                    assert filename.startswith('id') and '-' in filename, f"Unexpected test file format: {filename}"
                    speaker_id = filename.split('-')[0]
                    test_speakers.add(_format_global_speaker_id(speaker_id))
                    
        # Save speaker lists
        with open(self.dev_spk_file, 'w') as f:
            f.write('\n'.join(dev_speakers) + ('\n' if dev_speakers else ''))
        log.info(f"Saved {len(dev_speakers)} dev speakers to {self.dev_spk_file}")

        sorted_test_speakers = sorted(test_speakers)
        with open(self.test_spk_file, 'w') as f:
            f.write('\n'.join(sorted_test_speakers) + ('\n' if sorted_test_speakers else ''))
        log.info(f"Saved {len(test_speakers)} test speakers to {self.test_spk_file}")

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
        
        log.info(f"Generated speaker metadata for {len(speaker_df)} speakers")
        return speaker_df
    
    def get_speaker_list(self, from_metadata: bool = True) -> set:
        """Get the set of unique speaker IDs from the dataset."""
        if from_metadata and self.dev_metadata_file.exists():
            log.info(f"Loading speaker IDs from metadata: {self.dev_metadata_file}")
            df = pd.read_csv(self.dev_metadata_file, sep=self.sep, usecols=['speaker_id'])
            speaker_ids = set(df['speaker_id'].unique())
            log.info(f"Found {len(speaker_ids)} unique speakers in metadata")
            return speaker_ids
        
        # Scan directory structure directly
        log.info("Scanning directory structure for speaker IDs...")
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
                speaker_ids.add(_format_global_speaker_id(item.name))
        
        # Also extract from trial files if available
        log.info("Extracting speaker IDs from enrollment map...")
        with open(self.enroll_map_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                assert len(parts) >= 2, f"Invalid enroll.map format: {line}"

                enroll_id = parts[0]
                if '-enroll' in enroll_id:
                    speaker_id = enroll_id.replace('-enroll', '')
                    speaker_ids.add(_format_global_speaker_id(speaker_id))
                
                for file_path in parts[1:]:
                    path_parts = file_path.split(os.sep)
                    if path_parts and path_parts[0].startswith('id'):
                        speaker_ids.add(_format_global_speaker_id(path_parts[0]))

        if self.test_lst_file.exists():
            log.info("Extracting speaker IDs from test list...")
            with open(self.test_lst_file, 'r') as f:
                for line in f:
                    file_path = line.strip()
                    filename = file_path.split(os.sep)[-1]

                    assert filename.startswith('id') and '-' in filename, f"Unexpected test file format: {filename}"
                    speaker_id = filename.split('-')[0]
                    speaker_ids.add(_format_global_speaker_id(speaker_id))
        
        log.info(f"Found {len(speaker_ids)} unique speakers from directory scan")
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
                    enroll_speakers.add(_format_global_speaker_id(speaker_id))

        # From enroll.map
        if self.enroll_map_file.exists():
            with open(self.enroll_map_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                        
                    enroll_id = line.split()[0]
                    speaker_id = enroll_id.split('-')[0]
                    enroll_speakers.add(_format_global_speaker_id(speaker_id))
                    
        log.info(f"Found {len(enroll_speakers)} unique enrollment speakers.")
        return enroll_speakers        


if __name__ == "__main__":
    config = read_hydra_config(
        config_path='../../../configs',
        config_name='train.yaml',
        overrides=[
            f"paths.data_dir={os.environ['HOME']}/adversarial-robustness-for-sr/data",
            'datamodule=datasets/cnceleb',
            f"datamodule.dataset.artifacts_dir={os.environ['HOME']}/adversarial-robustness-for-sr/data/cnceleb/metadata"
        ]
    )
    config = config.datamodule.dataset
    
    # Resolve paths
    resolved_root = Path(config.data_dir).expanduser().resolve()
    resolved_artifacts = Path(config.artifacts_dir).expanduser().resolve()
    
    # Handle optional cnceleb2 parameter
    cnceleb2 = config.cnceleb2 if hasattr(config, 'cnceleb2') and config.cnceleb2 else None

    processor = CNCelebProcessor(
        root_dir=resolved_root,
        artifacts_dir=resolved_artifacts,
        cnceleb1=config.cnceleb1,
        dev_metadata_file=Path(config.dev_metadata_file),
        enroll_csv_path=Path(config.enroll_csv_path),
        test_unique_csv_path=Path(config.test_unique_csv_path),
        dev_spk_file=Path(config.dev_spk_file),
        test_spk_file=Path(config.test_spk_file),
        cnceleb2=cnceleb2,
        verbose=config.verbose,
        sep=config.sep,
        sample_rate=config.sample_rate,
        n_jobs=None,  # Use default
        use_pre_segmentation=config.use_pre_segmentation,
        segment_duration=config.segment_duration,
        segment_overlap=config.segment_overlap,
        min_segment_duration=config.min_segment_duration,
        vad=getattr(config, 'vad', None),
        concat_mapping_file=getattr(config, 'concat_mapping_file', None),
    )
    
    # Run the preprocessing pipeline
    dev_metadata = processor.generate_metadata()
    trials_df = processor.generate_trial_list()
    # Note: save_split_speaker_lists() is already called inside generate_metadata()
    processor.generate_enrollment_embeddings_list()
    processor.generate_unique_test_csv(trials_df)
    
    # Build snapshot by extracting only the comparable keys from config
    snapshot_source = {key: getattr(config, key) for key in CNCELEB_COMPARABLE_KEYS if hasattr(config, key)}
    snapshot = build_config_snapshot_from_mapping(snapshot_source, CNCELEB_COMPARABLE_KEYS)
    snapshot_path = save_config_snapshot(snapshot, resolved_artifacts)
    log.info(f"Saved configuration snapshot to {snapshot_path}")

    log.info(f"Preprocessing completed. Generated {len(trials_df)} trials.")
