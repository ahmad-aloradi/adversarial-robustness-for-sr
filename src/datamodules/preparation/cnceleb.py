"""Metadata preparation flow for the CNCeleb datamodule."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional

import pandas as pd

from src import utils
from src.datamodules.components.cnceleb.cnceleb_prep import CNCelebProcessor
from src.datamodules.components.common import CNCelebDefaults, get_dataset_class
from src.datamodules.components.utils import CsvProcessor

from .base import BaseMetadataPreparer, PreparationResult, SplitArtifacts
from .snapshot_keys import CNCELEB_COMPARABLE_KEYS

log = utils.get_pylogger(__name__)
DATASET_DEFAULTS = CNCelebDefaults()
DATASET_CLS, DF_COLS = get_dataset_class(DATASET_DEFAULTS.dataset_name)


class CNCelebMetadataPreparer(BaseMetadataPreparer):
    """Encapsulates metadata preparation for CNCeleb datasets."""

    COMPARABLE_KEYS: tuple[str, ...] = CNCELEB_COMPARABLE_KEYS

    def __init__(self, dataset_cfg: Any, csv_processor: CsvProcessor):
        super().__init__(dataset_cfg=dataset_cfg, csv_processor=csv_processor)

    def prepare(self) -> PreparationResult:
        dataset = self.dataset_cfg
        artifacts_dir = Path(dataset.artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        expected_snapshot = self.build_config_snapshot()

        base_search_dir_value: Optional[str] = dataset.get("base_search_dir", None)
        base_search_dir = Path(base_search_dir_value).expanduser().resolve() if base_search_dir_value else None

        core_files_to_copy: List[tuple[str, str]] = [
            ("cnceleb_dev.csv", dataset.dev_metadata_file),
            ("enroll.csv", dataset.enroll_csv_path),
            ("test_unique.csv", dataset.test_unique_csv_path),
            ("verification_trials.csv", dataset.veri_test_output_path),
            ("dev_speakers.txt", dataset.dev_spk_file),
            ("test_speakers.txt", dataset.test_spk_file),
        ]

        processor = self._build_processor(dataset)

        can_copy = False
        if base_search_dir and base_search_dir.exists():
            missing_sources = [
                source for source, _ in core_files_to_copy if not (base_search_dir / source).exists()
            ]
            config_path = base_search_dir / self.CONFIG_SNAPSHOT_FILENAME
            if missing_sources:
                log.info(f"Skipping cached artifacts in {base_search_dir} due to missing files: {', '.join(missing_sources)}")
            elif not config_path.is_file():
                log.info(f"Cached artifacts in {base_search_dir} lack {self.CONFIG_SNAPSHOT_FILENAME}; regenerating instead.")
            else:
                observed_snapshot = self.load_config_snapshot(config_path)
                if self.snapshots_match(expected_snapshot, observed_snapshot):
                    can_copy = True
                else:
                    diff = self.diff_config_snapshots(expected_snapshot, observed_snapshot)
                    log.info(f"Cached CNCeleb artifacts config mismatch detected in {base_search_dir}; regenerating. Differences: {diff}")

        if can_copy:
            log.info(f"Found matching pre-generated files in {base_search_dir}. Copying to experiment directory...")
            for source_file, target_path in core_files_to_copy:
                source_path = base_search_dir / source_file
                resolved_target = Path(target_path)
                resolved_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, resolved_target)
                log.info(f"Copied {source_path} -> {resolved_target}")

            dev_df = pd.read_csv(dataset.dev_metadata_file, sep=dataset.sep)
            log.info(f"Loaded pre-generated dev metadata with {len(dev_df)} rows")
        else:
            log.info("Generating CNCeleb metadata from scratch...")

            dev_df = processor.generate_metadata()

            trials_df = processor.generate_trial_list()
            trials_df.to_csv(dataset.veri_test_output_path, sep=dataset.get("sep", "|"), index=False)
            log.info(f"Saved verification trials to {dataset.veri_test_output_path}")

            processor.save_split_speaker_lists()

            enroll_df = processor.generate_enrollment_embeddings_list()
            if not Path(processor.enroll_csv_path).exists():
                raise FileNotFoundError(f"Failed to generate enrollment CSV: {processor.enroll_csv_path}")

            processor.generate_unique_test_csv(trials_df)
            if not Path(processor.test_unique_csv_path).exists():
                raise FileNotFoundError(f"Failed to generate test unique CSV: {processor.test_unique_csv_path}")

            enroll_speakers = set(enroll_df["enroll_id"].str.split("-").str[0])
            test_speakers = set(pd.read_csv(processor.test_spk_file, header=None)[0])
            dev_speakers = set(pd.read_csv(processor.dev_spk_file, header=None)[0])

            original_dev_rows = len(dev_df)
            dev_df = dev_df[~dev_df["speaker_id"].isin(test_speakers)]
            excluded_count = original_dev_rows - len(dev_df)
            log.info(f"Excluded {excluded_count} utterances belonging to test speakers from dev set.")

            dev_enroll_overlap = dev_speakers.intersection(enroll_speakers)
            test_enroll_overlap = test_speakers.intersection(enroll_speakers)
            test_dev_overlap = test_speakers.intersection(dev_speakers)

            assert not dev_enroll_overlap, (
                "Dev and enrollment speakers should not overlap! "
                f"Found {len(dev_enroll_overlap)} overlapping speakers."
            )
            assert test_enroll_overlap, (
                "Enrollment and test speakers should overlap! "
                f"Found {len(test_enroll_overlap)} overlapping speakers, expected more."
            )
            assert not test_dev_overlap, (
                "Test and dev speakers should not overlap! "
                f"Found {len(test_dev_overlap)} overlapping speakers."
            )

            log.info(f"Speaker set validation passed: {len(test_enroll_overlap)} enrollment speakers overlap with test set")

            dev_df.to_csv(processor.dev_metadata_file, sep=dataset.sep, index=False)

        speaker_lookup_df = processor.generate_speaker_metadata(dev_df)
        speaker_lookup_df.to_csv(dataset.speaker_lookup, sep=dataset.sep, index=False)

        dev_df, speaker_lookup_df = self.csv_processor.process(
            dataset_files=[str(processor.dev_metadata_file)],
            spks_metadata_paths=[dataset.speaker_lookup],
            verbose=dataset.verbose,
        )

        dev_df.to_csv(processor.dev_metadata_file, sep=dataset.sep, index=False)
        log.info(f"Saved metadata to {processor.dev_metadata_file}")
        speaker_lookup_df.to_csv(dataset.speaker_lookup, sep=dataset.sep, index=False)
        log.info(f"Saved speaker lookup to {dataset.speaker_lookup}")

        CsvProcessor.split_dataset(
            df=dev_df,
            train_ratio=dataset.train_ratio,
            save_csv=True,
            speaker_overlap=dataset.speaker_overlap,
            speaker_id_col=DATASET_CLS.SPEAKER_ID,
            train_csv=dataset.train_csv_file,
            val_csv=dataset.val_csv_file,
            sep=dataset.sep,
            seed=dataset.seed,
        )
        log.info(f"Saved train and val csvs to {dataset.train_csv_file} and {dataset.val_csv_file}")

        extras: Dict[str, Any] = {"core_files_copied": can_copy}

        splits = SplitArtifacts(
            train_csv=Path(dataset.train_csv_file),
            val_csv=Path(dataset.val_csv_file),
            speaker_lookup_csv=Path(dataset.speaker_lookup),
            dev_csv=Path(dataset.dev_metadata_file),
        )

        return PreparationResult(splits=splits, extras=extras)

    @staticmethod
    def _build_processor(dataset) -> CNCelebProcessor:
        processor = CNCelebProcessor(
            root_dir=dataset.data_dir,
            artifacts_dir=dataset.artifacts_dir,
            cnceleb1=dataset.cnceleb1,
            dev_metadata_file=dataset.dev_metadata_file,
            enroll_csv_path=dataset.enroll_csv_path,
            test_unique_csv_path=dataset.test_unique_csv_path,
            dev_spk_file=dataset.dev_spk_file,
            test_spk_file=dataset.test_spk_file,
            cnceleb2=dataset.get("cnceleb2", None),
            verbose=dataset.verbose,
            sep=dataset.sep,
            sample_rate=dataset.get("sample_rate", DATASET_DEFAULTS.sample_rate),
            use_pre_segmentation=dataset.use_pre_segmentation,
            segment_duration=dataset.segment_duration,
            segment_overlap=dataset.segment_overlap,
            min_segment_duration=dataset.min_segment_duration,
            vad=dataset.get("vad", None),
        )
            
        return processor

