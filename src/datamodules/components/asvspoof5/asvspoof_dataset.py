"""
ASVSpoof5 dataset classes.

Covers all five data-loading needs for ASVSpoof5:
  1. Training data            → ASVSpoofDataset (pre-segmented via BaseDataset)
  2. Track 1 dev/eval (CM)    → ASVSpoofTrack1Dataset  (spoof/bonafide detection)
  3. Track 2 enrollment       → ASVSpoofEnrollMulti     (multi-utterance per speaker)
  4. Track 2 unique test      → ASVSpoofTest            (one audio per unique test path)
  5. Track 2 trial pairs      → ASVSpoofTrialList       (lightweight enroll_id ↔ test_path ↔ label)

CSV schemas (pipe-separated, from asvspoof_prep.py artifacts):
  train.csv          : segment_id|speaker_id|rel_filepath|start_time|end_time|num_frames|...
  *.track_1.csv      : speaker_id|rel_filepath|gender|codec|codec_q|codec_seed|attack_tag|attack_label|key|tmp
  *.track_2.trial.csv: speaker_id|rel_filepath|gender|attack_label|key
  *.track_2.enroll.csv: speaker_id|map_path  (map_path = comma-separated file IDs)
  *_track_2.trial_unique.csv: test_path
"""

from typing import List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd

from src.datamodules.components.utils import AudioProcessor, BaseCollate, BaseDataset
from src.datamodules.components.common import get_dataset_class, ASVSpoofDefaults, DatasetItem
from src import utils

log = utils.get_pylogger(__name__)

DATASET_DEFAULTS = ASVSpoofDefaults()
DATASET_CLS, DF_COLS = get_dataset_class(DATASET_DEFAULTS.dataset_name)


# ---------------------------------------------------------------------------
#  Dataclasses — Training
# ---------------------------------------------------------------------------

@dataclass
class ASVSpoofItem(DatasetItem):
    """Single training item (extends generic DatasetItem)."""
    sample_rate: float = 16000.0


# ---------------------------------------------------------------------------
#  Dataclasses — Track 1 (Countermeasure / spoofing detection)
# ---------------------------------------------------------------------------

@dataclass
class ASVSpoofTrack1Sample:
    """Single Track 1 item: audio with CM label and attack metadata."""
    audio: torch.Tensor
    audio_length: int
    audio_path: str
    speaker_id: str
    gender: str
    attack_tag: str
    attack_label: str
    cm_key: str          # 'spoof' or 'bonafide'
    sample_rate: int


@dataclass
class ASVSpoofTrack1Batch:
    """Batched Track 1 items."""
    audio: torch.Tensor
    audio_length: torch.Tensor
    audio_path: Tuple[str, ...]
    speaker_id: Tuple[str, ...]
    gender: Tuple[str, ...]
    attack_tag: Tuple[str, ...]
    attack_label: Tuple[str, ...]
    cm_key: Tuple[str, ...]
    sample_rate: int


# ---------------------------------------------------------------------------
#  Dataclasses — Track 2 enrollment (multi-utterance)
# ---------------------------------------------------------------------------

@dataclass
class ASVSpoofEnrollSampleMulti:
    """Single multi-utterance enrollment sample before collation."""
    audios: List[torch.Tensor]
    audio_lengths: List[int]
    enroll_id: str
    audio_paths: List[str]
    sample_rate: int


@dataclass
class ASVSpoofEnrollBatchMulti:
    """Batched multi-utterance enrollment data consumed by the model."""
    audio: torch.Tensor
    audio_length: torch.Tensor
    audio_path: Tuple[str, ...]
    enroll_id: Tuple[str, ...]
    utt_counts: Tuple[int, ...]
    sample_rate: int


# ---------------------------------------------------------------------------
#  Dataclasses — Track 2 unique test files
# ---------------------------------------------------------------------------

@dataclass
class ASVSpoofTestSample:
    """Single test item from ASVSpoof dataset."""
    audio: torch.Tensor
    audio_length: int
    audio_path: str
    sample_rate: int


@dataclass
class ASVSpoofTestBatch:
    """Batched unique test utterances for embedding computation."""
    audio: torch.Tensor
    audio_length: torch.Tensor
    audio_path: Tuple[str, ...]
    sample_rate: int


# ===================================================================
#  Collate functions
# ===================================================================

class TrainCollate(BaseCollate):
    """Collate for training batches (delegates to BaseCollate)."""
    pass


class Track1Collate(BaseCollate):
    """Collate for Track 1 countermeasure detection batches."""

    def __call__(self, batch) -> ASVSpoofTrack1Batch:
        if not batch:
            raise ValueError("ASVSpoof Track 1 batch is empty.")

        audios = [item.audio for item in batch]
        lengths = torch.tensor([item.audio_length for item in batch], dtype=torch.long)
        padded = pad_sequence(audios, batch_first=True, padding_value=self.pad_value)

        return ASVSpoofTrack1Batch(
            audio=padded,
            audio_length=lengths,
            audio_path=tuple(item.audio_path for item in batch),
            speaker_id=tuple(item.speaker_id for item in batch),
            gender=tuple(item.gender for item in batch),
            attack_tag=tuple(item.attack_tag for item in batch),
            attack_label=tuple(item.attack_label for item in batch),
            cm_key=tuple(item.cm_key for item in batch),
            sample_rate=batch[0].sample_rate,
        )


class EnrollCollateMulti(BaseCollate):
    """Collate for multi-utterance enrollment: flattens utterances, returns `utt_counts`."""

    def __init__(self, pad_value=0):
        super().__init__(pad_value)

    def __call__(self, batch) -> ASVSpoofEnrollBatchMulti:
        if not batch:
            raise ValueError("ASVSpoof enrollment batch is empty.")

        all_audios: List[torch.Tensor] = []
        all_lengths: List[int] = []
        all_paths: List[str] = []
        enroll_ids: List[str] = []
        utt_counts: List[int] = []

        for item in batch:
            enroll_ids.append(item.enroll_id)
            utt_counts.append(len(item.audios))
            for audio, length, path in zip(item.audios, item.audio_lengths, item.audio_paths):
                all_audios.append(audio)
                all_lengths.append(length)
                all_paths.append(path)

        lengths = torch.tensor(all_lengths, dtype=torch.long)
        padded_audios = pad_sequence(all_audios, batch_first=True, padding_value=self.pad_value)

        return ASVSpoofEnrollBatchMulti(
            audio=padded_audios,
            audio_length=lengths,
            audio_path=tuple(all_paths),
            enroll_id=tuple(enroll_ids),
            utt_counts=tuple(utt_counts),
            sample_rate=batch[0].sample_rate,
        )


class TestCollate(BaseCollate):
    """Collate for unique test utterances."""

    def __init__(self, pad_value=0):
        super().__init__(pad_value)

    def __call__(self, batch) -> ASVSpoofTestBatch:
        if not batch:
            raise ValueError("ASVSpoof test batch is empty.")

        audios = [item.audio for item in batch]
        lengths = torch.tensor([item.audio_length for item in batch], dtype=torch.long)
        padded_audios = pad_sequence(audios, batch_first=True, padding_value=self.pad_value)

        return ASVSpoofTestBatch(
            audio=padded_audios,
            audio_length=lengths,
            audio_path=tuple(item.audio_path for item in batch),
            sample_rate=batch[0].sample_rate,
        )


# ===================================================================
#  Datasets
# ===================================================================

# ---- 1. Training dataset (pre-segmented) ----

class ASVSpoofDataset(BaseDataset):
    """Training dataset. Reads the pre-segmented ``train.csv`` produced by
    ``asvspoof_prep.py`` and returns ``ASVSpoofItem`` instances through
    ``BaseDataset.__getitem__``.

    Args:
        cm_mode: When True, override ``class_id`` with binary CM labels
            derived from the ``key`` column (bonafide=1, spoof=0).
            This is required for countermeasure (Track 1) training where
            the target is spoofing detection rather than speaker classification.
    """

    # Mapping from CM key string to integer class label
    CM_LABEL_MAP = {"bonafide": 1, "spoof": 0}

    def __init__(self, *args, transforms=None, cm_mode: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.transforms = transforms
        self.cm_mode = cm_mode        

        if cm_mode:
            key_col = DATASET_CLS.CM_LABEL  # 'key'
            if key_col not in self.dataset.columns:
                raise ValueError(
                    f"cm_mode=True requires '{key_col}' column in training CSV. "
                    f"Found columns: {list(self.dataset.columns)}"
                )
            self.dataset["class_id"] = (
                self.dataset[key_col].map(self.CM_LABEL_MAP).astype(int)
            )
            counts = self.dataset["class_id"].value_counts()
            log.info(
                f"CM mode: mapped '{key_col}' → class_id. "
                f"Distribution: bonafide(1)={counts.get(1, 0)}, spoof(0)={counts.get(0, 0)}"
            )

    def __getitem__(self, index: int):
        item = super().__getitem__(index)
        if self.transforms:
            item.audio = self.transforms(item.audio.unsqueeze(0)).squeeze(0)
        return item


# ---- 2. Track 1: Countermeasure detection (dev / eval) ----

class ASVSpoofTrack1Dataset(Dataset):
    """Dataset for Track 1 anti-spoofing detection.

    Each row in the Track 1 CSV has:
        speaker_id | rel_filepath | gender | codec | codec_q | codec_seed |
        attack_tag | attack_label | key | tmp

    The ``key`` column is the CM label (``spoof`` or ``bonafide``).
    """

    def __init__(
        self,
        data_dir: str,
        data_filepath: str,
        sample_rate: int = 16000,
        max_duration: Union[None, float, int] = None,
        apply_preemphasis: bool = False,
        sep: str = "|",
    ):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.audio_processor = AudioProcessor(sample_rate, apply_preemphasis=apply_preemphasis)

        self.df = pd.read_csv(data_filepath, sep=sep)

        required_cols = {DATASET_CLS.REL_FILEPATH, DATASET_CLS.CM_LABEL}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(
                f"Track 1 CSV is missing columns: {missing}. "
                f"Found: {list(self.df.columns)}"
            )

        if max_duration is not None and max_duration > 0:
            self.max_samples = int(max_duration * sample_rate)
        else:
            self.max_samples = -1

        log.info(f"Loaded {len(self.df)} Track 1 utterances from {data_filepath}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> ASVSpoofTrack1Sample:
        row = self.df.iloc[idx]
        rel_path = row[DATASET_CLS.REL_FILEPATH]
        audio_path = self.data_dir / rel_path
        audio, _ = self.audio_processor.process_audio(str(audio_path))

        return ASVSpoofTrack1Sample(
            audio=audio,
            audio_length=len(audio),
            audio_path=rel_path,
            speaker_id=str(row[DATASET_CLS.SPEAKER_ID]),
            gender=str(row[DATASET_CLS.GENDER]),
            attack_tag=str(row.get(DATASET_CLS.ATTACK_TAG, 'N/A')),
            attack_label=str(row[DATASET_CLS.ATTACK_LABEL]),
            cm_key=str(row[DATASET_CLS.CM_LABEL]),
            sample_rate=self.sample_rate,
        )


# ---- 3. Track 2 enrollment: multi-utterance ----

class ASVSpoofEnrollMulti(Dataset):
    """Multi-utterance enrollment dataset for Track 2.

    The enrollment CSV has columns ``speaker_id | map_path`` where ``map_path``
    is a comma-separated list of file IDs (e.g. ``D_A0000000562,D_A0000000898``).
    These IDs live in the same audio directory as the dev/eval data and need
    ``audio_subdir`` and ``files_extension`` to be resolved to full paths.

    Args:
        data_dir: Root data directory (parent of ``asvspoof5/``).
        df: Enrollment DataFrame with ``speaker_id`` and ``map_path`` columns.
            ``map_path`` must already be split into a ``list[str]`` of file IDs.
        audio_subdir: Relative path from *data_dir* to the directory containing
            enrollment audio (e.g. ``asvspoof5/asvspoof5_wav/dev``).
        files_extension: Audio file extension including the dot (e.g. ``.wav``).
        sample_rate: Target sample rate.
        apply_preemphasis: Whether to apply pre-emphasis filtering.
    """

    def __init__(
        self,
        data_dir: str,
        df: pd.DataFrame,
        audio_subdir: str,
        files_extension: str = ".wav",
        sample_rate: int = 16000,
        apply_preemphasis: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.audio_subdir = Path(audio_subdir)
        self.files_extension = files_extension
        self.sample_rate = sample_rate
        self.audio_processor = AudioProcessor(sample_rate, apply_preemphasis=apply_preemphasis)

        self.enroll_df = df

        total_utterances = sum(
            len(paths) if isinstance(paths, list) else 1
            for paths in self.enroll_df["map_path"]
        )
        log.info(
            f"Loaded {len(self.enroll_df)} enrollment IDs "
            f"with {total_utterances} total utterances"
        )

    def _resolve_file_id(self, file_id: str) -> str:
        """Turn a bare file ID into a relative path from data_dir."""
        return str(self.audio_subdir / f"{file_id}{self.files_extension}")

    def __len__(self):
        return len(self.enroll_df)

    def __getitem__(self, idx) -> ASVSpoofEnrollSampleMulti:
        row = self.enroll_df.iloc[idx]
        enroll_id = str(row["speaker_id"])

        map_paths = row["map_path"]
        if not isinstance(map_paths, list):
            map_paths = [p.strip() for p in str(map_paths).split(",") if p.strip()]

        audios: List[torch.Tensor] = []
        audio_lengths: List[int] = []
        audio_paths: List[str] = []

        for file_id in map_paths:
            rel_path = self._resolve_file_id(file_id)
            audio_path = self.data_dir / rel_path
            audio, _ = self.audio_processor.process_audio(str(audio_path))
            audios.append(audio)
            audio_lengths.append(len(audio))
            audio_paths.append(rel_path)

        return ASVSpoofEnrollSampleMulti(
            audios=audios,
            audio_lengths=audio_lengths,
            enroll_id=enroll_id,
            audio_paths=audio_paths,
            sample_rate=self.sample_rate,
        )


# ---- 4. Track 2 unique test files ----

class ASVSpoofTest(Dataset):
    """Dataset for loading unique test audio files (Track 2).

    Reads the ``*_track_2.trial_unique.csv`` which has a single column
    ``test_path`` containing relative audio paths.
    """

    def __init__(
        self,
        data_dir: str,
        df: pd.DataFrame,
        sample_rate: int = 16000,
        apply_preemphasis: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.audio_processor = AudioProcessor(sample_rate, apply_preemphasis=apply_preemphasis)
        self.test_df = df
        log.info(f"Loaded {len(self.test_df)} unique test file paths")

    def __len__(self):
        return len(self.test_df)

    def __getitem__(self, idx) -> ASVSpoofTestSample:
        row = self.test_df.iloc[idx]
        test_path = row["test_path"]
        audio_path = self.data_dir / test_path
        audio, _ = self.audio_processor.process_audio(str(audio_path))

        return ASVSpoofTestSample(
            audio=audio,
            audio_length=len(audio),
            audio_path=test_path,
            sample_rate=self.sample_rate,
        )


# ---- 5. Track 2 trial list (enrollment ID ↔ test path ↔ label) ----

class ASVSpoofTrialList:
    """Lightweight wrapper around the Track 2 trial CSV.

    Provides structured access to ``(enroll_id, test_path, label)`` triples used
    for scoring after embeddings have been pre-computed.

    The trial CSV has columns:
        speaker_id | rel_filepath | gender | attack_label | key

    where:
        - ``speaker_id`` = target (enrollment) speaker ID
        - ``rel_filepath`` = relative path to the test utterance
        - ``key`` ∈ {``target``, ``nontarget``, ``spoof``}
    """

    # Column mapping from CSV column names to semantic names
    COL_ENROLL_ID = DATASET_CLS.SPEAKER_ID       # 'speaker_id'
    COL_TEST_PATH = DATASET_CLS.REL_FILEPATH      # 'rel_filepath'
    COL_LABEL = DATASET_CLS.CM_LABEL               # 'key'

    def __init__(self, data_filepath: str, sep: str = "|"):
        self.df = pd.read_csv(data_filepath, sep=sep)

        required = {self.COL_ENROLL_ID, self.COL_TEST_PATH, self.COL_LABEL}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(
                f"Trial CSV missing columns: {missing}. Found: {list(self.df.columns)}"
            )

        log.info(
            f"Loaded {len(self.df)} trials "
            f"({self.df[self.COL_LABEL].value_counts().to_dict()})"
        )

    def __len__(self) -> int:
        return len(self.df)

    @property
    def enroll_ids(self) -> List[str]:
        """All unique enrollment (target speaker) IDs in the trial list."""
        return self.df[self.COL_ENROLL_ID].unique().tolist()

    @property
    def test_paths(self) -> List[str]:
        """All unique test utterance paths in the trial list."""
        return self.df[self.COL_TEST_PATH].unique().tolist()

    def itertrials(self):
        """Yield ``(enroll_id, test_path, label)`` for every trial."""
        for _, row in self.df.iterrows():
            yield (
                str(row[self.COL_ENROLL_ID]),
                str(row[self.COL_TEST_PATH]),
                str(row[self.COL_LABEL]),
            )