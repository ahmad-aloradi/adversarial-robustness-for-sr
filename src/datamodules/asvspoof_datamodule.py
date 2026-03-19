from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from src import utils
from src.datamodules.components.asvspoof5.asvspoof_prep import ASVSpoofProcessor
from src.datamodules.components.asvspoof5.asvspoof_dataset import (
    ASVSpoofDataset,
    ASVSpoofEnrollMulti,
    ASVSpoofTest,
    ASVSpoofTrack1Dataset,
    ASVSpoofTrialList,
    EnrollCollateMulti,
    TestCollate,
    Track1Collate,
    TrainCollate,
)
from src.datamodules.components.common import (
    ASVSpoofDefaults,
    get_dataset_class,
)
from src.datamodules.components.utils import make_dataloader

log = utils.get_pylogger(__name__)
DATASET_DEFAULTS = ASVSpoofDefaults()
DATASET_CLS, DF_COLS = get_dataset_class(DATASET_DEFAULTS.dataset_name)

# Artifact file names produced by ASVSpoofProcessor (all pipe-separated CSVs).
# Keys mirror ASVSpoofProcessor.artifacts_files.
_ARTIFACT_NAMES = {
    "train": "ASVspoof5.train.csv",
    # Track 1
    "dev_track1": "ASVspoof5.dev.track_1.csv",
    "eval_track1": "ASVspoof5.eval.track_1.csv",
    # Track 2 trials
    "dev_track2_trial": "ASVspoof5.dev.track_2.trial.csv",
    "eval_track2_trial": "ASVspoof5.eval.track_2.trial.csv",
    # Track 2 enrollment
    "dev_track2_enroll": "ASVspoof5.dev.track_2.enroll.csv",
    "eval_track2_enroll": "ASVspoof5.eval.track_2.enroll.csv",
    # Track 2 unique test paths
    "dev_track2_unique": "ASVspoof5.dev_track_2.trial_unique.csv",
    "eval_track2_unique": "ASVspoof5.eval_track_2.trial_unique.csv",
}


class ASVSpoofDataModule(LightningDataModule):
    """ASVSpoof5 DataModule with standardized interface for sv.py.

    Supports two evaluation tracks controlled by the ``eval_tracks`` config key:
      - **Track 1** — Countermeasure (CM) detection: spoof vs bonafide.
        Uses ``ASVSpoofTrack1Dataset`` with ``Track1Collate``.
      - **Track 2** — Spoofing-robust ASV: multi-utterance enrollment +
        unique test files + trial list with 3-class labels (target / nontarget / spoof).

    ``eval_tracks`` can be ``"1"``, ``"2"``, or ``"both"``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.train_data: Optional[ASVSpoofDataset] = None

        # Validation data structures (mirrors test_* below but for the dev split).
        # For Track 1 (CM): cm.py accumulates per-utterance scores and computes
        # official ASVSpoof5 metrics (minDCF, EER, actDCF, CLLR) at epoch end.
        # For Track 2 (SASV): sv.py must implement embedding-based validation
        # (t-EER/a-DCF) to consume these.
        self.val_data_dict: Dict[str, Any] = {}
        self.val_enrollment_data_dict: Dict[str, ASVSpoofEnrollMulti] = {}
        self.val_test_unique_data_dict: Dict[str, ASVSpoofTest] = {}
        self.val_trial_list_dict: Dict[str, ASVSpoofTrialList] = {}

        # test_data_dict: key → Dataset used by sv.py's test loop
        # For Track 1 the datasets are ASVSpoofTrack1Dataset (dev/eval).
        # For Track 2 the datasets are also Track1-style stand-ins whose sole
        # purpose is to drive the test-step loop; actual scoring goes through
        # get_enroll_and_trial_dataloaders().
        self.test_data_dict: Dict[str, Any] = {}
        self.enrollment_data_dict: Dict[str, ASVSpoofEnrollMulti] = {}
        self.test_unique_data_dict: Dict[str, ASVSpoofTest] = {}
        self.trial_list_dict: Dict[str, ASVSpoofTrialList] = {}

        self.eval_tracks_cfg = self.hparams.dataset.get("eval_tracks", "both")

    # ------------------------------------------------------------------
    #  Artifact path helpers
    # ------------------------------------------------------------------

    def _artifact(self, key: str) -> Path:
        """Resolve an artifact path by key (see ``_ARTIFACT_NAMES``)."""
        return Path(self.hparams.dataset.artifacts_dir) / _ARTIFACT_NAMES[key]

    def _base_artifact(self, key: str) -> Path:
        """Look up a pre-generated artifact in ``base_search_dir``."""
        return Path(self.hparams.dataset.base_search_dir) / _ARTIFACT_NAMES[key]

    # ------------------------------------------------------------------
    #  Data preparation
    # ------------------------------------------------------------------

    def prepare_data(self):
        """Run ASVSpoofProcessor to create metadata CSVs if not already present."""
        if self._artifacts_ready():
            log.info("Skipping ASVSpoof5 data preparation — all artifacts present.")
            return

        log.info("Preparing ASVSpoof5 data...")
        cfg = self.hparams.dataset
        processor = ASVSpoofProcessor(
            data_dir=cfg.data_dir,
            artifacts_dir=cfg.artifacts_dir,
            protocol_dir=cfg.protocol_dir,
            files_extension=cfg.get("files_extension", ".wav"),
            sample_rate=cfg.get("sample_rate", 16000),
            use_pre_segmentation=cfg.get("use_pre_segmentation", True),
            segment_duration=cfg.get("segment_duration", 3.0),
            min_segment_duration=cfg.get("min_segment_duration", 1.0),
            segment_overlap=cfg.get("segment_overlap", 0.0),
            sep=cfg.get("sep", "|"),
        )
        processor.build_train_metadata()
        processor.build_track1_metadata()
        processor.build_track2_enroll_metadata()
        df_dev_trial, df_eval_trial = processor.build_track2_trial_metadata()
        processor.generate_unique_test_csv(df_dev_trial, phase="dev_track_2_trial_unique")
        processor.generate_unique_test_csv(df_eval_trial, phase="eval_track_2_trial_unique")
        log.info("ASVSpoof5 data preparation complete.")

    def _artifacts_ready(self) -> bool:
        """Check whether all required CSV artifacts already exist."""
        # Always need the training CSV
        required = ["train"]
        tracks = self._resolve_eval_tracks()
        if "1" in tracks:
            required += ["dev_track1", "eval_track1"]
        if "2" in tracks:
            required += [
                "dev_track2_trial", "eval_track2_trial",
                "dev_track2_enroll", "eval_track2_enroll",
                "dev_track2_unique", "eval_track2_unique",
            ]
        # First check base_search_dir (pre-generated), then artifacts_dir
        for key in required:
            if not (self._base_artifact(key).exists() or self._artifact(key).exists()):
                return False
        return True

    def _resolve_artifact_path(self, key: str) -> Path:
        """Return the base_search_dir copy if it exists, else artifacts_dir."""
        base = self._base_artifact(key)
        return base if base.exists() else self._artifact(key)

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _resolve_eval_tracks(self) -> List[str]:
        return ["1", "2"] if self.eval_tracks_cfg == "both" else [str(self.eval_tracks_cfg)]

    @property
    def _sep(self) -> str:
        return self.hparams.dataset.get("sep", "|")

    @property
    def _data_dir(self) -> str:
        return self.hparams.dataset.data_dir

    @property
    def _sample_rate(self) -> int:
        return self.hparams.dataset.sample_rate

    # ------------------------------------------------------------------
    #  setup
    # ------------------------------------------------------------------

    def setup(self, stage: Optional[str] = None):
        """Instantiate PyTorch Datasets for the requested stage."""
        if not self._artifacts_ready():
            self.prepare_data()

        if stage == "fit" or stage is None:
            self._setup_train()
            self._setup_validation()

        if stage == "test" or stage is None:
            self._setup_test()

        # --- Uncomment the block below to subsample all splits for quick pipeline testing ---
        # self._subsample_for_testing(n_train=2000, n_val=500, n_test_speakers=10)

    def _setup_train(self):
        train_csv = self._resolve_artifact_path("train")
        max_duration = (
            -1
            if self.hparams.dataset.use_pre_segmentation
            else self.hparams.dataset.max_duration
        )

        self.train_data = ASVSpoofDataset(
            data_filepath=str(train_csv),
            data_dir=self._data_dir,
            sample_rate=self._sample_rate,
            max_duration=max_duration,
            sep=self._sep,
            cm_mode=self.hparams.dataset.get("cm_mode", False),
            apply_preemphasis=self.hparams.dataset.get("apply_preemphasis", False),
        )

    def _setup_validation(self):
        # Uses the official dev splits for validation.
        # Track 1 (CM): val_data_dict holds ASVSpoofTrack1Dataset instances
        # consumed by cm.py for per-utterance scoring (minDCF/EER/actDCF/CLLR).
        # Track 2 (SASV): requires sv.py to implement embedding-based validation
        # (t-EER / a-DCF) to consume val_enrollment_data_dict / val_trial_list_dict.
        eval_tracks = self._resolve_eval_tracks()
        for track in eval_tracks:
            if track == "1":
                self._setup_track1(dest="dev")
            elif track == "2":
                self._setup_track2(dest="dev")

    def _setup_test(self):
        eval_tracks = self._resolve_eval_tracks()
        for track in eval_tracks:
            if track == "1":
                self._setup_track1(dest="eval")
            elif track == "2":
                self._setup_track2(dest="eval")

    # ---- Track 1 (CM detection) ------------------------------------------

    def _setup_track1(self, dest: str):
        """Populate Track 1 dataset for dest ('dev' → val dicts, 'eval' → test dicts)."""
        log.info(f"Setting up Track 1 (CM detection) [split={dest}]")
        data_dict = self.val_data_dict if dest == "dev" else self.test_data_dict

        csv_path = self._resolve_artifact_path(f"{dest}_track1")
        ds = ASVSpoofTrack1Dataset(
            data_dir=self._data_dir,
            data_filepath=str(csv_path),
            sample_rate=self._sample_rate,
            sep=self._sep,
            apply_preemphasis=self.hparams.dataset.get("apply_preemphasis", False),
        )
        data_dict[f"asvspoof5_T1_{dest}"] = ds

    # ---- Track 2 (spoofing-robust ASV) -----------------------------------

    def _setup_track2(self, dest: str):
        """Populate Track 2 datasets for dest ('dev' → val dicts, 'eval' → test dicts)."""
        log.info(f"Setting up Track 2 (spoofing-robust ASV) [split={dest}]")
        audio_subdir_map = self._audio_subdir_map()

        enrollment_dict = self.val_enrollment_data_dict if dest == "dev" else self.enrollment_data_dict
        test_unique_dict = self.val_test_unique_data_dict if dest == "dev" else self.test_unique_data_dict
        trial_list_dict = self.val_trial_list_dict if dest == "dev" else self.trial_list_dict
        data_dict = self.val_data_dict if dest == "dev" else self.test_data_dict

        dict_key = f"asvspoof5_T2_{dest}"

        # --- Enrollment (multi-utterance) ---
        enroll_csv = self._resolve_artifact_path(f"{dest}_track2_enroll")
        enroll_df = pd.read_csv(enroll_csv, sep=self._sep)
        enroll_df["map_path"] = enroll_df["map_path"].apply(
            lambda x: [p.strip() for p in str(x).split(",") if p.strip()] if pd.notna(x) else []
        )
        enrollment_dict[dict_key] = ASVSpoofEnrollMulti(
            data_dir=self._data_dir,
            df=enroll_df,
            audio_subdir=audio_subdir_map.get(dest, f"asvspoof5/asvspoof5_wav/{dest}"),
            files_extension=self.hparams.dataset.get("files_extension", ".wav"),
            sample_rate=self._sample_rate,
        )

        # --- Unique test files ---
        unique_csv = self._resolve_artifact_path(f"{dest}_track2_unique")
        unique_df = pd.read_csv(unique_csv, sep=self._sep)
        test_unique_dict[dict_key] = ASVSpoofTest(
            data_dir=self._data_dir,
            df=unique_df,
            sample_rate=self._sample_rate,
        )

        # --- Trial list (lightweight, no audio) ---
        trial_csv = self._resolve_artifact_path(f"{dest}_track2_trial")
        trial_list_dict[dict_key] = ASVSpoofTrialList(
            data_filepath=str(trial_csv), sep=self._sep,
        )

        # Register unique-test dataset as the driver for the dataloader loop;
        # actual scoring happens via get_enroll_and_trial_dataloaders().
        data_dict[dict_key] = test_unique_dict[dict_key]

    def _audio_subdir_map(self) -> Dict[str, str]:
        """Map split name → relative audio subdirectory (from data_dir)."""
        ext = self.hparams.dataset.get("files_extension", ".wav")
        if ext == ".flac":
            return {"dev": "asvspoof5/flac_D", "eval": "asvspoof5/flac_E_eval", "train": "asvspoof5/flac_T"}
        return {"dev": "asvspoof5/asvspoof5_wav/dev", "eval": "asvspoof5/asvspoof5_wav/eval", "train": "asvspoof5/asvspoof5_wav/train"}

    # ------------------------------------------------------------------
    #  Dataloaders
    # ------------------------------------------------------------------

    def train_dataloader(self):
        assert self.hparams.get("loaders") is not None, "ASVSpoofDataModule requires 'loaders' config"
        return make_dataloader(
            dataset=self.train_data,
            loader_kwargs=dict(self.hparams.loaders.train),
            collate_fn=TrainCollate(),
            batch_sampler_cfg=self.hparams.dataset.get("batch_sampler", None),
        )

    def val_dataloader(self):
        # Track 1: cm.py consumes these via validation_step → _cm_eval_step.
        # Track 2: sv.py must implement embedding-based validation to use these.
        assert self.hparams.get("loaders") is not None, "ASVSpoofDataModule requires 'loaders' config"
        if not self.val_data_dict:
            return {}
        loaders = {}
        for key, dataset in self.val_data_dict.items():
            collate = Track1Collate() if "_T1_" in key else TestCollate()
            loaders[key] = DataLoader(
                dataset,
                **self.hparams.loaders.valid,
                collate_fn=collate,
            )
        return loaders

    def test_dataloader(self):
        """Return one test dataloader per registered test key.

        Track 1 keys use ``Track1Collate``; Track 2 keys use ``TestCollate``
        (the unique-test dataset stands in for the test loop).
        """
        loaders = {}
        for key, dataset in self.test_data_dict.items():
            if "_T1_" in key:
                collate = Track1Collate()
            else:
                collate = TestCollate()
            loaders[key] = DataLoader(
                dataset,
                **self.hparams.loaders.test,
                collate_fn=collate,
            )
        return loaders

    def get_enroll_and_trial_dataloaders(
        self, test_filename: str = None, *args, **kwargs
    ) -> Tuple[DataLoader, DataLoader]:
        """Return enrollment and unique-test dataloaders for sv.py.

        Args:
            test_filename: Key matching one of the Track 2 entries
                (e.g. ``'asvspoof5_T2_dev'``).

        Returns:
            ``(enrollment_dataloader, test_unique_dataloader)``
        """
        if not self.enrollment_data_dict or not self.test_unique_data_dict:
            raise ValueError("Enrollment / test data not prepared. Call setup('test') first.")

        if test_filename is None:
            test_filename = next(iter(self.enrollment_data_dict.keys()))

        if test_filename not in self.enrollment_data_dict:
            raise ValueError(
                f"Unknown test set: {test_filename}. "
                f"Available: {list(self.enrollment_data_dict.keys())}"
            )

        assert "enrollment" in self.hparams.loaders, "Enrollment loader config is required"
        loader_cfg = dict(self.hparams.loaders.get("enrollment"))

        enrollment_dataloader = DataLoader(
            self.enrollment_data_dict[test_filename],
            **loader_cfg,
            collate_fn=EnrollCollateMulti(),
        )
        test_unique_dataloader = DataLoader(
            self.test_unique_data_dict[test_filename],
            **loader_cfg,
            collate_fn=TestCollate(),
        )
        return enrollment_dataloader, test_unique_dataloader

    def get_trial_list(self, test_filename: str) -> ASVSpoofTrialList:
        """Return the trial list for a Track 2 test set (used for scoring)."""
        if test_filename not in self.trial_list_dict:
            raise ValueError(
                f"No trial list for '{test_filename}'. "
                f"Available: {list(self.trial_list_dict.keys())}"
            )
        return self.trial_list_dict[test_filename]
