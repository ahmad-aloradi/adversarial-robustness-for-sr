"""
Tests for ASVSpoofDataModule — structural correctness of setup() routing.

Verifies that after setup("fit") and setup("test"):
  - the right datasets land in the right dicts (val_* vs test_*)
  - dict keys follow the asvspoof5_T{1,2}_{dev,eval} naming convention
  - dataloaders are returned with the right types/collates
  - track-selection config ("1", "2", "both") is respected

Audio I/O is not exercised; setup() is pure CSV parsing so no mocking needed.
"""

from pathlib import Path

import pytest
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

# ── Locate workspace root and metadata directory ──────────────────────
ROOT = Path(__file__).resolve().parents[1]
METADATA_DIR = ROOT / "data" / "asvspoof5" / "metadata"

pytestmark = pytest.mark.skipif(
    not METADATA_DIR.is_dir(),
    reason=f"ASVSpoof5 metadata not found at {METADATA_DIR}",
)

# ── Config factory ────────────────────────────────────────────────────

def _make_cfg(eval_tracks="both"):
    """Build a minimal OmegaConf config for ASVSpoofDataModule tests."""
    return OmegaConf.create({
        "dataset": {
            "data_dir": str(ROOT / "data"),
            "base_search_dir": str(METADATA_DIR),
            # Intentionally non-existent so _resolve_artifact_path falls back to base_search_dir
            "artifacts_dir": str(ROOT / "tmp_test_artifacts_nonexistent"),
            "protocol_dir": str(ROOT / "data" / "asvspoof5" / "protocols"),
            "files_extension": ".wav",
            "sep": "|",
            "sample_rate": 16000,
            "use_pre_segmentation": True,
            "segment_duration": 3.0,
            "segment_overlap": 0.0,
            "min_segment_duration": 1.0,
            "max_duration": 3.0,
            "eval_tracks": eval_tracks,
        },
        "loaders": {
            "train":      {"batch_size": 4, "shuffle": True,  "num_workers": 0, "drop_last": False, "pin_memory": False},
            "valid":      {"batch_size": 4, "shuffle": False, "num_workers": 0, "drop_last": False, "pin_memory": False},
            "test":       {"batch_size": 4, "shuffle": False, "num_workers": 0, "drop_last": False, "pin_memory": False},
            "enrollment": {"batch_size": 2, "shuffle": False, "num_workers": 0, "drop_last": False, "pin_memory": False},
        },
    })


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def dm_both():
    from src.datamodules.asvspoof_datamodule import ASVSpoofDataModule
    return ASVSpoofDataModule(**_make_cfg("both"))

@pytest.fixture
def dm_track1():
    from src.datamodules.asvspoof_datamodule import ASVSpoofDataModule
    return ASVSpoofDataModule(**_make_cfg("1"))

@pytest.fixture
def dm_track2():
    from src.datamodules.asvspoof_datamodule import ASVSpoofDataModule
    return ASVSpoofDataModule(**_make_cfg("2"))


# ======================================================================
#  1. setup("fit") — validation split uses "dev"
# ======================================================================

class TestSetupFit:

    def test_train_data_populated(self, dm_both):
        dm_both.setup("fit")
        assert dm_both.train_data is not None

    def test_val_track1_key_is_dev(self, dm_both):
        dm_both.setup("fit")
        assert "asvspoof5_T1_dev" in dm_both.val_data_dict

    def test_val_track2_key_is_dev(self, dm_both):
        dm_both.setup("fit")
        assert "asvspoof5_T2_dev" in dm_both.val_data_dict

    def test_val_track2_enrollment_populated(self, dm_both):
        dm_both.setup("fit")
        assert "asvspoof5_T2_dev" in dm_both.val_enrollment_data_dict

    def test_val_track2_unique_test_populated(self, dm_both):
        dm_both.setup("fit")
        assert "asvspoof5_T2_dev" in dm_both.val_test_unique_data_dict

    def test_val_track2_trial_list_populated(self, dm_both):
        dm_both.setup("fit")
        assert "asvspoof5_T2_dev" in dm_both.val_trial_list_dict

    def test_val_track2_data_dict_points_to_unique_test(self, dm_both):
        """val_data_dict[T2] must be the same object as val_test_unique_data_dict[T2]."""
        dm_both.setup("fit")
        assert dm_both.val_data_dict["asvspoof5_T2_dev"] is dm_both.val_test_unique_data_dict["asvspoof5_T2_dev"]

    def test_fit_does_not_populate_test_dicts(self, dm_both):
        dm_both.setup("fit")
        assert len(dm_both.test_data_dict) == 0
        assert len(dm_both.enrollment_data_dict) == 0
        assert len(dm_both.test_unique_data_dict) == 0
        assert len(dm_both.trial_list_dict) == 0

    def test_val_no_eval_keys(self, dm_both):
        """Validation must not contain the eval split."""
        dm_both.setup("fit")
        assert not any("eval" in k for k in dm_both.val_data_dict)


# ======================================================================
#  2. setup("test") — test split uses "eval"
# ======================================================================

class TestSetupTest:

    def test_test_track1_key_is_eval(self, dm_both):
        dm_both.setup("test")
        assert "asvspoof5_T1_eval" in dm_both.test_data_dict

    def test_test_track2_key_is_eval(self, dm_both):
        dm_both.setup("test")
        assert "asvspoof5_T2_eval" in dm_both.test_data_dict

    def test_test_track2_enrollment_populated(self, dm_both):
        dm_both.setup("test")
        assert "asvspoof5_T2_eval" in dm_both.enrollment_data_dict

    def test_test_track2_unique_test_populated(self, dm_both):
        dm_both.setup("test")
        assert "asvspoof5_T2_eval" in dm_both.test_unique_data_dict

    def test_test_track2_trial_list_populated(self, dm_both):
        dm_both.setup("test")
        assert "asvspoof5_T2_eval" in dm_both.trial_list_dict

    def test_test_track2_data_dict_points_to_unique_test(self, dm_both):
        dm_both.setup("test")
        assert dm_both.test_data_dict["asvspoof5_T2_eval"] is dm_both.test_unique_data_dict["asvspoof5_T2_eval"]

    def test_test_does_not_populate_val_dicts(self, dm_both):
        dm_both.setup("test")
        assert len(dm_both.val_data_dict) == 0
        assert len(dm_both.val_enrollment_data_dict) == 0
        assert len(dm_both.val_trial_list_dict) == 0

    def test_test_no_dev_keys(self, dm_both):
        """Test dicts must not contain the dev split."""
        dm_both.setup("test")
        assert not any("dev" in k for k in dm_both.test_data_dict)


# ======================================================================
#  3. Dataloaders
# ======================================================================

class TestDataloaders:

    def test_val_dataloader_returns_dict(self, dm_both):
        dm_both.setup("fit")
        loaders = dm_both.val_dataloader()
        assert isinstance(loaders, dict)
        assert "asvspoof5_T1_dev" in loaders
        assert "asvspoof5_T2_dev" in loaders

    def test_val_dataloader_entries_are_dataloaders(self, dm_both):
        dm_both.setup("fit")
        for dl in dm_both.val_dataloader().values():
            assert isinstance(dl, DataLoader)

    def test_val_dataloader_empty_when_not_set_up(self, dm_both):
        # setup("fit") not called — val_data_dict is empty
        assert dm_both.val_dataloader() == {}

    def test_test_dataloader_returns_dict(self, dm_both):
        dm_both.setup("test")
        loaders = dm_both.test_dataloader()
        assert isinstance(loaders, dict)
        assert "asvspoof5_T1_eval" in loaders
        assert "asvspoof5_T2_eval" in loaders

    def test_test_dataloader_entries_are_dataloaders(self, dm_both):
        dm_both.setup("test")
        for dl in dm_both.test_dataloader().values():
            assert isinstance(dl, DataLoader)

    def test_get_enroll_and_trial_dataloaders(self, dm_both):
        dm_both.setup("test")
        enroll_dl, test_dl = dm_both.get_enroll_and_trial_dataloaders("asvspoof5_T2_eval")
        assert isinstance(enroll_dl, DataLoader)
        assert isinstance(test_dl, DataLoader)

    def test_get_enroll_dataloaders_unknown_key_raises(self, dm_both):
        dm_both.setup("test")
        with pytest.raises(ValueError, match="Unknown test set"):
            dm_both.get_enroll_and_trial_dataloaders("nonexistent_key")

    def test_get_trial_list(self, dm_both):
        from src.datamodules.components.asvspoof5.asvspoof_dataset import ASVSpoofTrialList
        dm_both.setup("test")
        tl = dm_both.get_trial_list("asvspoof5_T2_eval")
        assert isinstance(tl, ASVSpoofTrialList)

    def test_get_trial_list_unknown_key_raises(self, dm_both):
        dm_both.setup("test")
        with pytest.raises(ValueError, match="No trial list"):
            dm_both.get_trial_list("nonexistent_key")


# ======================================================================
#  4. Track-selection config
# ======================================================================

class TestTrackSelection:

    def test_track1_only_fit_has_only_t1_in_val(self, dm_track1):
        dm_track1.setup("fit")
        assert "asvspoof5_T1_dev" in dm_track1.val_data_dict
        assert not any("T2" in k for k in dm_track1.val_data_dict)
        assert len(dm_track1.val_enrollment_data_dict) == 0

    def test_track1_only_test_has_only_t1_in_test(self, dm_track1):
        dm_track1.setup("test")
        assert "asvspoof5_T1_eval" in dm_track1.test_data_dict
        assert not any("T2" in k for k in dm_track1.test_data_dict)
        assert len(dm_track1.enrollment_data_dict) == 0

    def test_track2_only_fit_has_only_t2_in_val(self, dm_track2):
        dm_track2.setup("fit")
        assert "asvspoof5_T2_dev" in dm_track2.val_data_dict
        assert not any("T1" in k for k in dm_track2.val_data_dict)

    def test_track2_only_test_has_only_t2_in_test(self, dm_track2):
        dm_track2.setup("test")
        assert "asvspoof5_T2_eval" in dm_track2.test_data_dict
        assert not any("T1" in k for k in dm_track2.test_data_dict)

    def test_both_tracks_fit_has_two_val_entries(self, dm_both):
        dm_both.setup("fit")
        assert len(dm_both.val_data_dict) == 2

    def test_both_tracks_test_has_two_test_entries(self, dm_both):
        dm_both.setup("test")
        assert len(dm_both.test_data_dict) == 2
