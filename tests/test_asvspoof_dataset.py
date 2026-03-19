"""
Tests for ASVSpoof5 dataset classes in asvspoof_dataset.py.

Uses the real metadata CSVs in data/asvspoof5/metadata/ to verify that each
dataset class correctly parses column names, validates schemas, and returns
properly-typed samples.  Audio I/O is mocked to avoid needing actual .wav files.
"""

import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest
import torch

# ── Locate workspace root and metadata directory ──────────────────────
ROOT = Path(__file__).resolve().parents[1]
METADATA_DIR = ROOT / "data" / "asvspoof5" / "metadata"

# Skip the whole module when metadata is missing (e.g. CI without data)
pytestmark = pytest.mark.skipif(
    not METADATA_DIR.is_dir(),
    reason=f"ASVSpoof5 metadata not found at {METADATA_DIR}",
)

# ── Real CSV file paths ──────────────────────────────────────────────
TRAIN_CSV       = str(METADATA_DIR / "ASVspoof5.train.csv")
TRACK1_DEV_CSV  = str(METADATA_DIR / "ASVspoof5.dev.track_1.csv")
TRACK1_EVAL_CSV = str(METADATA_DIR / "ASVspoof5.eval.track_1.csv")
TRACK2_TRIAL_DEV_CSV  = str(METADATA_DIR / "ASVspoof5.dev.track_2.trial.csv")
TRACK2_TRIAL_EVAL_CSV = str(METADATA_DIR / "ASVspoof5.eval.track_2.trial.csv")
TRACK2_ENROLL_DEV_CSV  = str(METADATA_DIR / "ASVspoof5.dev.track_2.enroll.csv")
TRACK2_ENROLL_EVAL_CSV = str(METADATA_DIR / "ASVspoof5.eval.track_2.enroll.csv")
TEST_UNIQUE_DEV_CSV  = str(METADATA_DIR / "ASVspoof5.dev_track_2.trial_unique.csv")
TEST_UNIQUE_EVAL_CSV = str(METADATA_DIR / "ASVspoof5.eval_track_2.trial_unique.csv")

# Expected row counts (header excluded)
EXPECTED_ROWS = {
    "train": 634_758,
    "track1_dev": 140_950,
    "track1_eval": 680_774,
    "track2_trial_dev": 282_456,
    "track2_trial_eval": 834_536,
    "track2_enroll_dev": 398,
    "track2_enroll_eval": 367,
    "test_unique_dev": 140_950,
    "test_unique_eval": 496_632,
}

SEP = "|"
DATA_DIR = str(ROOT / "data")  # root passed to dataset classes


# ── Helpers ───────────────────────────────────────────────────────────

def _fake_audio(*_args, **_kwargs):
    """Return a dummy (waveform, sr) matching AudioProcessor.process_audio."""
    return torch.randn(16000), 16000


def _load_csv(path, nrows=None):
    return pd.read_csv(path, sep=SEP, nrows=nrows)


# ======================================================================
#  1. ASVSpoofTrack1Dataset
# ======================================================================

class TestASVSpoofTrack1Dataset:
    """Tests for Track 1 CM detection dataset."""

    def test_loads_dev_csv_columns(self):
        """Track 1 dev CSV must have the expected columns."""
        df = _load_csv(TRACK1_DEV_CSV, nrows=5)
        expected = {"speaker_id", "rel_filepath", "gender", "codec", "codec_q",
                    "codec_seed", "attack_tag", "attack_label", "key", "tmp"}
        assert expected == set(df.columns), f"Column mismatch: {set(df.columns)}"

    def test_loads_eval_csv_columns(self):
        df = _load_csv(TRACK1_EVAL_CSV, nrows=5)
        expected = {"speaker_id", "rel_filepath", "gender", "codec", "codec_q",
                    "codec_seed", "attack_tag", "attack_label", "key", "tmp"}
        assert expected == set(df.columns)

    def test_dev_row_count(self):
        df = _load_csv(TRACK1_DEV_CSV)
        assert len(df) == EXPECTED_ROWS["track1_dev"]

    def test_eval_row_count(self):
        df = _load_csv(TRACK1_EVAL_CSV)
        assert len(df) == EXPECTED_ROWS["track1_eval"]

    def test_cm_key_values_dev(self):
        """Track 1 CM key must be 'spoof' or 'bonafide'."""
        df = _load_csv(TRACK1_DEV_CSV)
        assert set(df["key"].unique()) == {"spoof", "bonafide"}

    def test_cm_key_values_eval(self):
        df = _load_csv(TRACK1_EVAL_CSV)
        assert set(df["key"].unique()) == {"spoof", "bonafide"}

    def test_rel_filepath_format_dev(self):
        """rel_filepath should start with 'asvspoof5/' and end with '.wav'."""
        df = _load_csv(TRACK1_DEV_CSV, nrows=100)
        assert all(p.startswith("asvspoof5/") for p in df["rel_filepath"])
        assert all(p.endswith(".wav") for p in df["rel_filepath"])

    @patch.object(
        __import__("src.datamodules.components.utils", fromlist=["AudioProcessor"]).AudioProcessor,
        "process_audio",
        side_effect=_fake_audio,
    )
    def test_getitem_returns_track1_sample(self, _mock):
        """Instantiate dataset and verify __getitem__ returns correct type."""
        from src.datamodules.components.asvspoof5.asvspoof_dataset import (
            ASVSpoofTrack1Dataset,
            ASVSpoofTrack1Sample,
        )
        ds = ASVSpoofTrack1Dataset(
            data_dir=DATA_DIR,
            data_filepath=TRACK1_DEV_CSV,
            sample_rate=16000,
            sep=SEP,
        )
        assert len(ds) == EXPECTED_ROWS["track1_dev"]
        sample = ds[0]
        assert isinstance(sample, ASVSpoofTrack1Sample)
        assert sample.cm_key in ("spoof", "bonafide")
        assert isinstance(sample.audio, torch.Tensor)
        assert sample.audio_length > 0
        assert sample.speaker_id.startswith("D_")
        assert sample.gender in ("F", "M")

    @patch.object(
        __import__("src.datamodules.components.utils", fromlist=["AudioProcessor"]).AudioProcessor,
        "process_audio",
        side_effect=_fake_audio,
    )
    def test_missing_required_column_raises(self, _mock):
        """If a required column is missing, constructor should raise ValueError."""
        from src.datamodules.components.asvspoof5.asvspoof_dataset import ASVSpoofTrack1Dataset
        import tempfile

        df = _load_csv(TRACK1_DEV_CSV, nrows=5).drop(columns=["key"])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f, sep=SEP, index=False)
            tmp = f.name
        try:
            with pytest.raises(ValueError, match="missing columns"):
                ASVSpoofTrack1Dataset(data_dir=DATA_DIR, data_filepath=tmp, sep=SEP)
        finally:
            os.unlink(tmp)


# ======================================================================
#  2. ASVSpoofTrialList  (Track 2 trials — lightweight, no audio)
# ======================================================================

class TestASVSpoofTrialList:
    """Tests for the Track 2 trial-list wrapper."""

    def test_dev_columns(self):
        df = _load_csv(TRACK2_TRIAL_DEV_CSV, nrows=5)
        assert {"speaker_id", "rel_filepath", "gender", "attack_label", "key"} == set(df.columns)

    def test_eval_columns(self):
        df = _load_csv(TRACK2_TRIAL_EVAL_CSV, nrows=5)
        assert {"speaker_id", "rel_filepath", "gender", "attack_label", "key"} == set(df.columns)

    def test_dev_row_count(self):
        df = _load_csv(TRACK2_TRIAL_DEV_CSV)
        assert len(df) == EXPECTED_ROWS["track2_trial_dev"]

    def test_eval_row_count(self):
        df = _load_csv(TRACK2_TRIAL_EVAL_CSV)
        assert len(df) == EXPECTED_ROWS["track2_trial_eval"]

    def test_track2_key_values_dev(self):
        """Track 2 key must be 'target', 'nontarget', or 'spoof'."""
        df = _load_csv(TRACK2_TRIAL_DEV_CSV)
        assert set(df["key"].unique()) == {"target", "nontarget", "spoof"}

    def test_track2_key_values_eval(self):
        df = _load_csv(TRACK2_TRIAL_EVAL_CSV)
        assert set(df["key"].unique()) == {"target", "nontarget", "spoof"}

    def test_trial_list_construction_dev(self):
        from src.datamodules.components.asvspoof5.asvspoof_dataset import ASVSpoofTrialList
        tl = ASVSpoofTrialList(data_filepath=TRACK2_TRIAL_DEV_CSV, sep=SEP)
        assert len(tl) == EXPECTED_ROWS["track2_trial_dev"]

    def test_trial_list_enroll_ids(self):
        from src.datamodules.components.asvspoof5.asvspoof_dataset import ASVSpoofTrialList
        tl = ASVSpoofTrialList(data_filepath=TRACK2_TRIAL_DEV_CSV, sep=SEP)
        ids = tl.enroll_ids
        assert len(ids) > 0
        assert all(isinstance(i, str) for i in ids)
        # Dev enroll IDs start with D_
        assert all(i.startswith("D_") for i in ids)

    def test_trial_list_test_paths(self):
        from src.datamodules.components.asvspoof5.asvspoof_dataset import ASVSpoofTrialList
        tl = ASVSpoofTrialList(data_filepath=TRACK2_TRIAL_DEV_CSV, sep=SEP)
        paths = tl.test_paths
        assert len(paths) > 0
        assert all(p.startswith("asvspoof5/") and p.endswith(".wav") for p in paths)

    def test_itertrials_yields_triples(self):
        from src.datamodules.components.asvspoof5.asvspoof_dataset import ASVSpoofTrialList
        tl = ASVSpoofTrialList(data_filepath=TRACK2_TRIAL_DEV_CSV, sep=SEP)
        first_triple = next(tl.itertrials())
        enroll_id, test_path, label = first_triple
        assert isinstance(enroll_id, str)
        assert isinstance(test_path, str)
        assert label in ("target", "nontarget", "spoof")

    def test_missing_column_raises(self):
        from src.datamodules.components.asvspoof5.asvspoof_dataset import ASVSpoofTrialList
        import tempfile

        df = _load_csv(TRACK2_TRIAL_DEV_CSV, nrows=5).drop(columns=["rel_filepath"])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f, sep=SEP, index=False)
            tmp = f.name
        try:
            with pytest.raises(ValueError, match="missing columns"):
                ASVSpoofTrialList(data_filepath=tmp, sep=SEP)
        finally:
            os.unlink(tmp)


# ======================================================================
#  3. ASVSpoofEnrollMulti  (Track 2 enrollment)
# ======================================================================

class TestASVSpoofEnrollMulti:
    """Tests for multi-utterance enrollment dataset."""

    def test_enroll_csv_columns_dev(self):
        df = _load_csv(TRACK2_ENROLL_DEV_CSV, nrows=5)
        assert set(df.columns) == {"speaker_id", "map_path"}

    def test_enroll_csv_columns_eval(self):
        df = _load_csv(TRACK2_ENROLL_EVAL_CSV, nrows=5)
        assert set(df.columns) == {"speaker_id", "map_path"}

    def test_dev_row_count(self):
        df = _load_csv(TRACK2_ENROLL_DEV_CSV)
        assert len(df) == EXPECTED_ROWS["track2_enroll_dev"]

    def test_eval_row_count(self):
        df = _load_csv(TRACK2_ENROLL_EVAL_CSV)
        assert len(df) == EXPECTED_ROWS["track2_enroll_eval"]

    def test_map_path_is_comma_separated(self):
        """map_path column holds comma-separated file IDs."""
        df = _load_csv(TRACK2_ENROLL_DEV_CSV)
        for _, row in df.head(20).iterrows():
            ids = row["map_path"].split(",")
            assert len(ids) >= 1
            for fid in ids:
                assert fid.startswith("D_A")

    def test_map_path_eval_prefix(self):
        df = _load_csv(TRACK2_ENROLL_EVAL_CSV)
        for _, row in df.head(20).iterrows():
            ids = row["map_path"].split(",")
            for fid in ids:
                assert fid.startswith("E_A")

    @patch.object(
        __import__("src.datamodules.components.utils", fromlist=["AudioProcessor"]).AudioProcessor,
        "process_audio",
        side_effect=_fake_audio,
    )
    def test_getitem_returns_multi_sample(self, _mock):
        from src.datamodules.components.asvspoof5.asvspoof_dataset import (
            ASVSpoofEnrollMulti,
            ASVSpoofEnrollSampleMulti,
        )
        df = _load_csv(TRACK2_ENROLL_DEV_CSV)
        ds = ASVSpoofEnrollMulti(
            data_dir=DATA_DIR,
            df=df,
            audio_subdir="asvspoof5/asvspoof5_wav/dev",
            files_extension=".wav",
            sample_rate=16000,
        )
        assert len(ds) == EXPECTED_ROWS["track2_enroll_dev"]
        sample = ds[0]
        assert isinstance(sample, ASVSpoofEnrollSampleMulti)
        assert isinstance(sample.audios, list)
        assert len(sample.audios) >= 1
        assert all(isinstance(a, torch.Tensor) for a in sample.audios)
        assert sample.enroll_id.startswith("D_")
        assert len(sample.audio_paths) == len(sample.audios)

    @patch.object(
        __import__("src.datamodules.components.utils", fromlist=["AudioProcessor"]).AudioProcessor,
        "process_audio",
        side_effect=_fake_audio,
    )
    def test_resolve_file_id(self, _mock):
        """File IDs should be resolved to full relative paths."""
        from src.datamodules.components.asvspoof5.asvspoof_dataset import ASVSpoofEnrollMulti

        df = _load_csv(TRACK2_ENROLL_DEV_CSV, nrows=2)
        ds = ASVSpoofEnrollMulti(
            data_dir=DATA_DIR,
            df=df,
            audio_subdir="asvspoof5/asvspoof5_wav/dev",
            files_extension=".wav",
        )
        resolved = ds._resolve_file_id("D_A0000000562")
        assert resolved == "asvspoof5/asvspoof5_wav/dev/D_A0000000562.wav"

    @patch.object(
        __import__("src.datamodules.components.utils", fromlist=["AudioProcessor"]).AudioProcessor,
        "process_audio",
        side_effect=_fake_audio,
    )
    def test_audio_paths_are_resolvable(self, _mock):
        """Each audio_path in the sample should start with audio_subdir."""
        from src.datamodules.components.asvspoof5.asvspoof_dataset import ASVSpoofEnrollMulti

        df = _load_csv(TRACK2_ENROLL_DEV_CSV, nrows=3)
        ds = ASVSpoofEnrollMulti(
            data_dir=DATA_DIR,
            df=df,
            audio_subdir="asvspoof5/asvspoof5_wav/dev",
            files_extension=".wav",
        )
        sample = ds[0]
        for p in sample.audio_paths:
            assert p.startswith("asvspoof5/asvspoof5_wav/dev/")
            assert p.endswith(".wav")

    @patch.object(
        __import__("src.datamodules.components.utils", fromlist=["AudioProcessor"]).AudioProcessor,
        "process_audio",
        side_effect=_fake_audio,
    )
    def test_pre_split_map_path(self, _mock):
        """If map_path is already a list, it should be used directly."""
        from src.datamodules.components.asvspoof5.asvspoof_dataset import ASVSpoofEnrollMulti

        df = _load_csv(TRACK2_ENROLL_DEV_CSV, nrows=2)
        # Pre-split the map_path
        df["map_path"] = df["map_path"].apply(lambda x: x.split(","))
        ds = ASVSpoofEnrollMulti(
            data_dir=DATA_DIR,
            df=df,
            audio_subdir="asvspoof5/asvspoof5_wav/dev",
            files_extension=".wav",
        )
        sample = ds[0]
        assert len(sample.audios) == len(df.iloc[0]["map_path"])


# ======================================================================
#  4. ASVSpoofTest  (Track 2 unique test files)
# ======================================================================

class TestASVSpoofTest:
    """Tests for unique test file dataset."""

    def test_unique_test_csv_columns_dev(self):
        df = _load_csv(TEST_UNIQUE_DEV_CSV, nrows=5)
        assert list(df.columns) == ["test_path"]

    def test_unique_test_csv_columns_eval(self):
        df = _load_csv(TEST_UNIQUE_EVAL_CSV, nrows=5)
        assert list(df.columns) == ["test_path"]

    def test_dev_row_count(self):
        df = _load_csv(TEST_UNIQUE_DEV_CSV)
        assert len(df) == EXPECTED_ROWS["test_unique_dev"]

    def test_eval_row_count(self):
        df = _load_csv(TEST_UNIQUE_EVAL_CSV)
        assert len(df) == EXPECTED_ROWS["test_unique_eval"]

    def test_paths_are_unique(self):
        df = _load_csv(TEST_UNIQUE_DEV_CSV)
        assert df["test_path"].is_unique

    def test_path_format(self):
        df = _load_csv(TEST_UNIQUE_DEV_CSV, nrows=100)
        assert all(p.startswith("asvspoof5/") for p in df["test_path"])
        assert all(p.endswith(".wav") for p in df["test_path"])

    @patch.object(
        __import__("src.datamodules.components.utils", fromlist=["AudioProcessor"]).AudioProcessor,
        "process_audio",
        side_effect=_fake_audio,
    )
    def test_getitem_returns_test_sample(self, _mock):
        from src.datamodules.components.asvspoof5.asvspoof_dataset import (
            ASVSpoofTest,
            ASVSpoofTestSample,
        )
        df = _load_csv(TEST_UNIQUE_DEV_CSV, nrows=10)
        ds = ASVSpoofTest(data_dir=DATA_DIR, df=df, sample_rate=16000)
        assert len(ds) == 10
        sample = ds[0]
        assert isinstance(sample, ASVSpoofTestSample)
        assert isinstance(sample.audio, torch.Tensor)
        assert sample.audio_length > 0
        assert sample.audio_path.startswith("asvspoof5/")
        assert sample.sample_rate == 16000


# ======================================================================
#  5. Collate functions
# ======================================================================

class TestCollates:
    """Tests for collate functions with synthetic data."""

    def test_track1_collate(self):
        from src.datamodules.components.asvspoof5.asvspoof_dataset import (
            ASVSpoofTrack1Sample,
            Track1Collate,
            ASVSpoofTrack1Batch,
        )
        samples = [
            ASVSpoofTrack1Sample(
                audio=torch.randn(sr := 16000),
                audio_length=sr,
                audio_path=f"asvspoof5/test_{i}.wav",
                speaker_id=f"D_{i:04d}",
                gender="F",
                attack_tag="AC1",
                attack_label="A11",
                cm_key="spoof",
                sample_rate=sr,
            )
            for i in range(4)
        ]
        batch = Track1Collate()(samples)
        assert isinstance(batch, ASVSpoofTrack1Batch)
        assert batch.audio.shape[0] == 4
        assert len(batch.speaker_id) == 4
        assert all(k == "spoof" for k in batch.cm_key)

    def test_track1_collate_empty_raises(self):
        from src.datamodules.components.asvspoof5.asvspoof_dataset import Track1Collate
        with pytest.raises(ValueError, match="empty"):
            Track1Collate()([])

    def test_enroll_collate_multi(self):
        from src.datamodules.components.asvspoof5.asvspoof_dataset import (
            ASVSpoofEnrollSampleMulti,
            EnrollCollateMulti,
            ASVSpoofEnrollBatchMulti,
        )
        samples = [
            ASVSpoofEnrollSampleMulti(
                audios=[torch.randn(16000), torch.randn(8000)],
                audio_lengths=[16000, 8000],
                enroll_id=f"D_{i:04d}",
                audio_paths=[f"a_{i}_0.wav", f"a_{i}_1.wav"],
                sample_rate=16000,
            )
            for i in range(3)
        ]
        batch = EnrollCollateMulti()(samples)
        assert isinstance(batch, ASVSpoofEnrollBatchMulti)
        # 3 speakers × 2 utterances each = 6 flattened
        assert batch.audio.shape[0] == 6
        assert batch.utt_counts == (2, 2, 2)
        assert len(batch.enroll_id) == 3

    def test_enroll_collate_multi_empty_raises(self):
        from src.datamodules.components.asvspoof5.asvspoof_dataset import EnrollCollateMulti
        with pytest.raises(ValueError, match="empty"):
            EnrollCollateMulti()([])

    def test_test_collate(self):
        from src.datamodules.components.asvspoof5.asvspoof_dataset import (
            ASVSpoofTestSample,
            TestCollate,
            ASVSpoofTestBatch,
        )
        samples = [
            ASVSpoofTestSample(
                audio=torch.randn(16000 + i * 1000),
                audio_length=16000 + i * 1000,
                audio_path=f"test_{i}.wav",
                sample_rate=16000,
            )
            for i in range(3)
        ]
        batch = TestCollate()(samples)
        assert isinstance(batch, ASVSpoofTestBatch)
        assert batch.audio.shape[0] == 3
        # Padded to max length
        assert batch.audio.shape[1] == 18000
        assert len(batch.audio_path) == 3

    def test_test_collate_empty_raises(self):
        from src.datamodules.components.asvspoof5.asvspoof_dataset import TestCollate
        with pytest.raises(ValueError, match="empty"):
            TestCollate()([])


# ======================================================================
#  6. Cross-consistency between CSVs
# ======================================================================

class TestCrossCSVConsistency:
    """Verify that related CSVs are mutually consistent."""

    def test_track2_unique_subset_of_trials_dev(self):
        """Every unique test path should appear in the trial list."""
        unique_df = _load_csv(TEST_UNIQUE_DEV_CSV)
        trial_df = _load_csv(TRACK2_TRIAL_DEV_CSV)
        trial_paths = set(trial_df["rel_filepath"])
        unique_paths = set(unique_df["test_path"])
        assert unique_paths.issubset(trial_paths), (
            f"{len(unique_paths - trial_paths)} unique paths not in trials"
        )

    def test_track2_unique_subset_of_trials_eval(self):
        unique_df = _load_csv(TEST_UNIQUE_EVAL_CSV)
        trial_df = _load_csv(TRACK2_TRIAL_EVAL_CSV)
        trial_paths = set(trial_df["rel_filepath"])
        unique_paths = set(unique_df["test_path"])
        assert unique_paths.issubset(trial_paths)

    def test_track2_unique_matches_trial_distinct_paths_dev(self):
        """Unique test CSV should have exactly the distinct rel_filepaths from trial CSV."""
        unique_df = _load_csv(TEST_UNIQUE_DEV_CSV)
        trial_df = _load_csv(TRACK2_TRIAL_DEV_CSV)
        assert set(unique_df["test_path"]) == set(trial_df["rel_filepath"])

    def test_track2_enroll_ids_match_trial_ids_dev(self):
        """All enrollment speaker IDs should appear in trial speaker_id column."""
        enroll_df = _load_csv(TRACK2_ENROLL_DEV_CSV)
        trial_df = _load_csv(TRACK2_TRIAL_DEV_CSV)
        enroll_ids = set(enroll_df["speaker_id"])
        trial_ids = set(trial_df["speaker_id"])
        assert enroll_ids.issubset(trial_ids), (
            f"{len(enroll_ids - trial_ids)} enrollment IDs not in trials"
        )

    def test_track1_dev_paths_cover_track2_dev_trial_paths(self):
        """Track 2 trial utterances should be a subset of Track 1 utterances (same audio pool)."""
        t1 = _load_csv(TRACK1_DEV_CSV)
        t2 = _load_csv(TRACK2_TRIAL_DEV_CSV)
        t1_paths = set(t1["rel_filepath"])
        t2_paths = set(t2["rel_filepath"])
        assert t2_paths.issubset(t1_paths), (
            f"{len(t2_paths - t1_paths)} Track 2 paths not in Track 1"
        )


# ======================================================================
#  7. Train CSV structure
# ======================================================================

class TestTrainCSV:
    """Verify the training CSV has the expected structure."""

    def test_train_columns(self):
        df = _load_csv(TRAIN_CSV, nrows=5)
        expected = {
            "segment_id", "speaker_id", "rel_filepath", "start_time", "end_time",
            "num_frames", "segment_duration", "recording_duration", "gender",
            "attack_tag", "attack_label", "key", "class_id", "dataset_name",
            "sample_rate", "language", "country", "speaker_name", "text",
        }
        assert expected == set(df.columns)

    def test_train_row_count(self):
        df = _load_csv(TRAIN_CSV)
        assert len(df) == EXPECTED_ROWS["train"]

    def test_train_segment_times_valid(self):
        """start_time < end_time for every row."""
        df = _load_csv(TRAIN_CSV, nrows=1000)
        assert (df["start_time"] < df["end_time"]).all()

    def test_train_segment_duration_positive(self):
        df = _load_csv(TRAIN_CSV, nrows=1000)
        assert (df["segment_duration"] > 0).all()

    def test_train_sample_rate_uniform(self):
        df = _load_csv(TRAIN_CSV, nrows=1000)
        assert (df["sample_rate"] == 16000).all()

    def test_train_key_values(self):
        df = _load_csv(TRAIN_CSV)
        assert set(df["key"].unique()) == {"spoof", "bonafide"}

    def test_train_rel_filepath_format(self):
        df = _load_csv(TRAIN_CSV, nrows=100)
        assert all(p.startswith("asvspoof5/") for p in df["rel_filepath"])
        assert all(p.endswith(".wav") for p in df["rel_filepath"])
