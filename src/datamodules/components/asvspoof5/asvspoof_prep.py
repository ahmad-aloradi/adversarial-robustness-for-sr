"""
ASVSpoof5 dataset preparation
"""

import os
from pathlib import Path
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from typing import Optional, Literal

import pandas as pd
import soundfile as sf
from tqdm import tqdm

from src import utils
from src.datamodules.components.common import get_dataset_class, get_speaker_class, ASVSpoofDefaults
from src.datamodules.components.utils import segment_utterance
from hydra.utils import instantiate
from src.datamodules.preparation.base import (
    build_config_snapshot_from_mapping,
    read_hydra_config,
    save_config_snapshot,
)
from src.datamodules.preparation.snapshot_keys import ASVSPOOF_COMPARABLE_KEYS

DATASET_DEFAULTS = ASVSpoofDefaults()
DATESET_CLS, DF_COLS = get_dataset_class(DATASET_DEFAULTS.dataset_name)
SPEAKER_CLS, _ = get_speaker_class(DATASET_DEFAULTS.dataset_name)
TEST_PATH_COL = 'test_path'

log = utils.get_pylogger(__name__)


def process_audio_file(args_tuple):
    """Process a single audio file - optimized for multiprocessing."""
    (audio_path, root_dir_str, sample_rate, speaker_id, gender, attack_tag, attack_label, cm_label) = args_tuple

    assert sample_rate == DATASET_DEFAULTS.sample_rate, (
        f"Expected sample rate {DATASET_DEFAULTS.sample_rate}, got {sample_rate}"
    )
    audio_path = Path(audio_path)
    root_dir = Path(root_dir_str)

    info = sf.info(str(audio_path))
    rel_path = audio_path.relative_to(root_dir)

    return {
        DATESET_CLS.SPEAKER_ID: speaker_id,
        DATESET_CLS.REL_FILEPATH: str(rel_path),
        DATESET_CLS.REC_DURATION: info.duration,
        DATESET_CLS.GENDER: gender,
        DATESET_CLS.ATTACK_TAG: attack_tag,
        DATESET_CLS.ATTACK_LABEL: attack_label,
        DATESET_CLS.CM_LABEL: cm_label,
        DATESET_CLS.CLASS_ID: None,
        DATESET_CLS.DATASET: DATASET_DEFAULTS.dataset_name,
        DATESET_CLS.SR: info.samplerate,
        DATESET_CLS.LANGUAGE: DATASET_DEFAULTS.language,
        DATESET_CLS.NATIONALITY: None,
        DATESET_CLS.SPEAKER_NAME: None,
        DATESET_CLS.TEXT: None,
    }


def write_dataset_csv(df, path, sep='|', fillna_value='N/A'):
    """Save updated metadata"""
    df = df.fillna(fillna_value)
    df.to_csv(path, sep=sep, index=False)


class ASVSpoofProcessor:
    """ASVSpoof dataset processor."""

    def __init__(
            self,
            data_dir: str,
            artifacts_dir: str,
            protocol_dir: str,
            files_extension: str = '.wav',
            sample_rate: int = 16000,
            use_pre_segmentation: bool = True,
            segment_duration: float = 4.0,
            min_segment_duration: float = 1.0,
            segment_overlap: float = 0.0,
            sep: str = '|',
            n_jobs: Optional[int] = None,
        ):

        self.dataset_root = Path(data_dir)
        self.artifacts_dir = Path(artifacts_dir)
        self.files_extension = files_extension
        self.protocol_dir = Path(protocol_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        # optional args for metadata building and segmentation
        self.sample_rate = sample_rate
        self.use_pre_segmentation = use_pre_segmentation
        self.segment_duration = segment_duration
        self.min_segment_duration = min_segment_duration
        self.segment_overlap = segment_overlap
        self.sep = sep
        self.n_jobs = n_jobs or cpu_count()

        # Train protocol columns: T_5232 T_0000000010 M - - - AC3 A06 spoof -
        self.TRAIN_TSV_COLS = [
            'speaker_id', 'filename', 'gender', 'codec', 'codec_q', 'codec_seed',
            'attack_tag', 'attack_label', 'key', 'tmp',
        ]
        # Track1 protocol columns (identical to train)
        self.TRACK1_TSV_COLS = self.TRAIN_TSV_COLS.copy()

        # Track 2 protocol trials: D_0062 D_0000000001 F A11 spoof
        self.TRACK2_TRIAL_TSV_COLS = ['speaker_id', 'filename', 'gender', 'attack_label', 'key']

         # Track 2 protocol enroll: D_4205 D_A0000000562,D_A0000000898,D_A0000000328
        self.TRACK2_ENROLL_TSV_COLS = ['speaker_id', 'map_path']

        # Assume the original metadata files names, no need to pass them as args.
        METADATA_FILES = {
            'codec': 'ASVspoof5.codec.config.csv',
            'train': 'ASVspoof5.train.tsv',
            # Track 1
            'dev_track1': 'ASVspoof5.dev.track_1.tsv',
            'eval_track1': 'ASVspoof5.eval.track_1.tsv',
            # Track 2
            'dev_track2_trial': 'ASVspoof5.dev.track_2.trial.tsv',
            'dev_track2_enroll': 'ASVspoof5.dev.track_2.enroll.tsv',
            'eval_track2_trial': 'ASVspoof5.eval.track_2.trial.tsv',
            'eval_track2_enroll':'ASVspoof5.eval.track_2.enroll.tsv',
        }

        self.metadata_files = {}
        self.artifacts_files = {}
        for key, filename in METADATA_FILES.items():
            tsv_path = self.protocol_dir / filename
            if not tsv_path.is_file():
                raise FileNotFoundError(f"Metadata file {key} not found at {tsv_path}")
            self.metadata_files[key] = tsv_path
            self.artifacts_files[key] = self.artifacts_dir / filename.replace('.tsv', '.csv')     

        # Handle the cases of unique test paths outside the loop
        self.artifacts_files.update({
        'dev_track_2_trial_unique': self.artifacts_dir / 'ASVspoof5.dev_track_2.trial_unique.csv',
        'eval_track_2_trial_unique': self.artifacts_dir / 'ASVspoof5.eval_track_2.trial_unique.csv',
        })

        log.info(f"ASVSpoofProcessor initialized with dataset_root: {self.dataset_root}")


    def load_tsv(self, tsv_path, columns, sep=r'\s+', engine='python'):
        """Load TSV file with error handling."""
        return pd.read_csv(tsv_path, sep=sep, names=columns, engine=engine)
    

    def _get_dir_from_extension(self, split: Literal['train', 'dev', 'eval']):
        """Build audio file paths from the filename column"""
        flac_subdir_mapping = {
            'train': 'flac_T',
            'dev': 'flac_D',
            'eval': 'flac_E_eval'
        }
        wav_subdir_mapping = {
            'train': 'asvspoof5_wav/train',
            'dev': 'asvspoof5_wav/dev',
            'eval': 'asvspoof5_wav/eval'
        }
        if self.files_extension == '.flac':
            audio_dir_relapath = Path(flac_subdir_mapping[split])
            audio_dir_fullpath = self.dataset_root / flac_subdir_mapping[split]
        elif self.files_extension == '.wav':
            audio_dir_relapath = Path(wav_subdir_mapping[split])
            audio_dir_fullpath = self.dataset_root / wav_subdir_mapping[split]
        else:
            raise ValueError(f"Unsupported file extension: {self.files_extension}")
        return audio_dir_relapath, audio_dir_fullpath


    def build_train_metadata(self):
        """Building metadata for ASVspoof5.train.tsv. 
        It also appends segment_id, duration, etc. to the training metadadata
        """
        df_proto = self.load_tsv(tsv_path=self.metadata_files['train'], columns=self.TRAIN_TSV_COLS)

        _, audio_dir_fullpath = self._get_dir_from_extension(split='train')
        all_audio_files = [
            (
                audio_dir_fullpath / f"{row['filename']}{self.files_extension}",
                self.dataset_root,
                row['speaker_id'],
                row['gender'],
                row['attack_tag'],
                row['attack_label'],
                row['key'],
            )
            for _, row in df_proto.iterrows()
        ]

        log.info(f"Processing {len(all_audio_files)} original audio files...")

        tasks = [
            (str(f), str(r), self.sample_rate, sid, gender, attack_tag, attack_label, cm_label)
            for f, r, sid, gender, attack_tag, attack_label, cm_label in all_audio_files
        ]

        utterances = []
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            future_to_path = {executor.submit(process_audio_file, task): task[0] for task in tasks}

            for future in tqdm(as_completed(future_to_path), total=len(tasks), desc="Processing files"):
                result = future.result()
                assert result, f"Processing failed for: {future_to_path[future]}"
                utterances.append(result)

        # Apply segmentation if enabled
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
                )
                if not segments:
                    skipped_utterances += 1
                else:
                    all_segments.extend(segments)

            total_utterances = len(utterances)
            skipped_pct = (skipped_utterances / total_utterances) * 100 if total_utterances > 0 else 0.0
            avg_segment_length = sum(s.get('segment_duration', 0.0) for s in all_segments) / len(all_segments)

            log.info(f"Generated {len(all_segments)} segments from {total_utterances} utterances")
            log.info(f"Average segments per utterance: {len(all_segments)/total_utterances:.2f}")
            log.info(f"Average segment duration: {avg_segment_length:.3f}s")
            log.info(
                f"Skipped {skipped_utterances} utterances ({skipped_pct:.2f}%) "
                f"due to duration < min_segment_duration={self.min_segment_duration}s or silence filtering"
            )
            df_train = pd.DataFrame(all_segments)
        else:
            log.info("Pre-segmentation disabled. Using full file metadata with random cropping during training.")
            df_train = pd.DataFrame(utterances)

        write_dataset_csv(df_train, self.artifacts_files['train'])
        log.info(f"Saved metadata for {len(df_train)} to {self.artifacts_files['train']}")

        return df_train
    

    def build_track1_metadata(self):
        """Building metadata for ASVspoof5.dev.track_1.tsv and ASVspoof5.eval.track_1.tsv 
        Each file has 5 columns: SPEAKER_ID FLAC_FILE_NAME SPEAKER_GENDER CODEC CODEC_Q CODEC_SEED ATTACK_TAG ATTACK_LABEL KEY TMP
        """
        df_dev_track1 = self.load_tsv(self.metadata_files['dev_track1'], columns=self.TRACK1_TSV_COLS)
        df_eval_track1 = self.load_tsv(self.metadata_files['eval_track1'], columns=self.TRACK1_TSV_COLS)

        audio_dir_relpath_dev, _ = self._get_dir_from_extension(split='dev')
        audio_dir_relpath_eval, _ = self._get_dir_from_extension(split='eval')
        df_dev_track1['filename'] = df_dev_track1['filename'].apply(lambda x: str(audio_dir_relpath_dev / f"{x}{self.files_extension}"))
        df_eval_track1['filename'] = df_eval_track1['filename'].apply(lambda x: str(audio_dir_relpath_eval / f"{x}{self.files_extension}"))
        
        df_dev_track1.rename(columns={'filename': DATESET_CLS.REL_FILEPATH}, inplace=True)
        df_eval_track1.rename(columns={'filename': DATESET_CLS.REL_FILEPATH}, inplace=True)

        write_dataset_csv(df_dev_track1, self.artifacts_files['dev_track1'])
        write_dataset_csv(df_eval_track1, self.artifacts_files['eval_track1'])

        log.info(
            f"Saved {len(df_eval_track1)} to {self.artifacts_files['eval_track1']} "
            f"and {len(df_dev_track1)} to {self.artifacts_files['dev_track1']}"
            )


    def build_track2_trial_metadata(self):
        """ Building metadata for ASVspoof5.dev.track_2.trial.tsv and ASVspoof5.eval.track_2.trial.tsv"""
        df_dev_track2 = self.load_tsv(self.metadata_files['dev_track2_trial'], columns=self.TRACK2_TRIAL_TSV_COLS)
        df_eval_track2 = self.load_tsv(self.metadata_files['eval_track2_trial'], columns=self.TRACK2_TRIAL_TSV_COLS)

        audio_dir_relpath_dev, _ = self._get_dir_from_extension(split='dev')
        audio_dir_relpath_eval, _ = self._get_dir_from_extension(split='eval')
        df_dev_track2['filename'] = df_dev_track2['filename'].apply(lambda x: str(audio_dir_relpath_dev / f"{x}{self.files_extension}"))
        df_eval_track2['filename'] = df_eval_track2['filename'].apply(lambda x: str(audio_dir_relpath_eval / f"{x}{self.files_extension}"))
        
        df_dev_track2.rename(columns={'filename': DATESET_CLS.REL_FILEPATH}, inplace=True)
        df_eval_track2.rename(columns={'filename': DATESET_CLS.REL_FILEPATH}, inplace=True)


        write_dataset_csv(df_dev_track2, self.artifacts_files['dev_track2_trial'])
        write_dataset_csv(df_eval_track2, self.artifacts_files['eval_track2_trial'])

        log.info(
            f"Saved {len(df_eval_track2)} to {self.artifacts_files['eval_track2_trial']} "
            f"and {len(df_dev_track2)} to {self.artifacts_files['dev_track2_trial']}"
            )
        
        return df_dev_track2, df_eval_track2


    def build_track2_enroll_metadata(self):
        """ ASVspoof5.dev.enroll.tsv and ASVspoof5.eval.enroll.tsv"""
        """ Building metadata for ASVspoof5.dev.track_2.trial.tsv and ASVspoof5.eval.track_2.trial.tsv"""
        df_dev_track2 = self.load_tsv(self.metadata_files['dev_track2_enroll'], columns=self.TRACK2_ENROLL_TSV_COLS)
        df_eval_track2 = self.load_tsv(self.metadata_files['eval_track2_enroll'], columns=self.TRACK2_ENROLL_TSV_COLS)

        write_dataset_csv(df_dev_track2, self.artifacts_files['dev_track2_enroll'])
        write_dataset_csv(df_eval_track2, self.artifacts_files['eval_track2_enroll'])

        log.info(
            f"Saved {len(df_eval_track2)} to {self.artifacts_files['dev_track2_enroll']} "
            f"and {len(df_dev_track2)} to {self.artifacts_files['dev_track2_enroll']}"
            )


    def generate_unique_test_csv(
        self, 
        trials_df: pd.DataFrame, 
        phase: Literal['dev_track_2_trial_unique', 'eval_track_2_trial_unique']
        ) -> None:
        """Generate unique test CSV from trials (test paths only)."""

        # Track 2 protocol trials: D_0062 D_0000000001 F A11 spoof
        assert 'filename' in self.TRACK2_TRIAL_TSV_COLS, 'Expected "filename" to be in the set of columns'
        
        unique_test_paths = pd.DataFrame({TEST_PATH_COL: trials_df[DATESET_CLS.REL_FILEPATH].unique()})
        unique_test_paths.to_csv(self.artifacts_files[phase], sep=self.sep, index=False)
        log.info(f"Found {len(unique_test_paths)} unique test paths")
        log.info(f"Saved test unique CSV to {self.artifacts_files[phase]}")
    

if __name__ == "__main__":
    # Resolve data directory (check for HPC path first)
    data_dir = f"{os.environ['HOME']}/adversarial-robustness-for-sr/data"
    hpc_data_dir = Path(data_dir) / "datasets"
    if hpc_data_dir.is_dir() and (hpc_data_dir / "asvspoof5").is_dir():
        data_dir = str(hpc_data_dir)

    # Load Hydra config
    config = read_hydra_config(
        config_path='../../../configs',
        config_name='train.yaml',
        overrides=[
            f"paths.data_dir={data_dir}",
            "datamodule=datasets/asvspoof5",
        ]
    )
    config = config.datamodule.dataset
    config.artifacts_dir = f"{data_dir}/asvspoof5/metadata"    
    # Resolve paths
    resolved_root = Path(config.data_dir).expanduser().resolve()
    resolved_artifacts = Path(config.artifacts_dir).expanduser().resolve()
    
    processor = ASVSpoofProcessor(
        data_dir=config.data_dir,
        artifacts_dir=config.artifacts_dir,
        protocol_dir=config.protocol_dir,
        files_extension=config.files_extension,
        sample_rate=config.sample_rate,
        use_pre_segmentation=config.use_pre_segmentation,
        segment_duration=config.segment_duration,
        min_segment_duration=config.min_segment_duration,
        segment_overlap=config.segment_overlap,
        sep=config.sep,
        n_jobs=None,
        )

    # Run the preprocessing pipeline
    dev_metadata = processor.build_train_metadata()
    processor.build_track1_metadata()
    processor.build_track2_enroll_metadata()
    # Handle trial metadata and generate unique test CSV for track 2
    df_dev_trial_track2, df_eval_trial_track2 = processor.build_track2_trial_metadata()
    processor.generate_unique_test_csv(df_dev_trial_track2, 
                                       phase='dev_track_2_trial_unique')
    processor.generate_unique_test_csv(df_eval_trial_track2, 
                                       phase='eval_track_2_trial_unique')


    # Build snapshot by extracting only the comparable keys from config
    snapshot_source = {key: getattr(config, key) for key in ASVSPOOF_COMPARABLE_KEYS if hasattr(config, key)}
    snapshot = build_config_snapshot_from_mapping(snapshot_source, ASVSPOOF_COMPARABLE_KEYS)
    snapshot_path = save_config_snapshot(snapshot, resolved_artifacts)
    log.info(f"Saved configuration snapshot to {snapshot_path}")

    log.info(f"ASVSPoof Preprocessing completed")
