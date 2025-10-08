"""Metadata preparation flow for the CNCeleb datamodule."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any, Dict

import pandas as pd

from src import utils
from src.datamodules.components.cnceleb.cnceleb_prep import CNCelebProcessor
from src.datamodules.components.common import CNCelebDefaults, get_dataset_class
from src.datamodules.components.utils import CsvProcessor

from .base import BaseMetadataPreparer, PreparationResult, SplitArtifacts

log = utils.get_pylogger(__name__)
DATASET_DEFAULTS = CNCelebDefaults()
DATASET_CLS, DF_COLS = get_dataset_class(DATASET_DEFAULTS.dataset_name)


class CNCelebMetadataPreparer(BaseMetadataPreparer):
    """Encapsulates metadata preparation for CNCeleb datasets."""

    def __init__(self, dataset_cfg: Any, csv_processor: CsvProcessor):
        super().__init__(dataset_cfg=dataset_cfg, csv_processor=csv_processor)

    def prepare(self) -> PreparationResult:
        dataset = self.dataset_cfg
        artifacts_dir = Path(dataset.artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        base_search_dir = Path(dataset.get("base_search_dir", ""))
        core_files_to_copy = [
            ("cnceleb_dev.csv", dataset.dev_metadata_file),
            ("enroll.csv", dataset.enroll_csv_path),
            ("test_unique.csv", dataset.test_unique_csv_path),
            ("verification_trials.csv", dataset.veri_test_output_path),
            ("dev_speakers.txt", dataset.dev_spk_file),
            ("test_speakers.txt", dataset.test_spk_file),
        ]

        core_files_exist = base_search_dir.exists() and all(
            (base_search_dir / source_file).exists() for source_file, _ in core_files_to_copy
        )

        if core_files_exist:
            log.info(
                "Found core pre-generated files in %s. Copying to experiment directory...",
                base_search_dir,
            )
            for source_file, target_path in core_files_to_copy:
                source_path = base_search_dir / source_file
                resolved_target = Path(target_path)
                resolved_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, resolved_target)
                log.info("Copied %s -> %s", source_path, resolved_target)

            dev_df = pd.read_csv(dataset.dev_metadata_file, sep=dataset.sep)
            log.info("Loaded pre-generated dev metadata with %d rows", len(dev_df))
        else:
            log.info("Pre-generated core files not found. Generating metadata from scratch...")
            processor = self._build_processor(dataset)

            # Generate dev metadata first (needed for speaker statistics)
            dev_df = processor.generate_metadata()
            
            # Generate trials list
            trials_df = processor.generate_trial_list()
            trials_df.to_csv(dataset.veri_test_output_path, sep=dataset.get("sep", "|"), index=False)
            log.info("Saved verification trials to %s", dataset.veri_test_output_path)

            # Save speaker lists (creates dev_speakers.txt and test_speakers.txt)
            processor.save_split_speaker_lists()
            
            # Generate enrollment CSV
            enroll_df = processor.generate_enrollment_embeddings_list()
            if not Path(processor.enroll_csv_path).exists():
                raise FileNotFoundError(f"Failed to generate enrollment CSV: {processor.enroll_csv_path}")

            # Generate unique test CSV
            processor.generate_unique_test_csv(trials_df)
            if not Path(processor.test_unique_csv_path).exists():
                raise FileNotFoundError(f"Failed to generate test unique CSV: {processor.test_unique_csv_path}")

            # Load speaker sets for validation
            enroll_speakers = set(enroll_df["enroll_id"].str.split("-").str[0])
            test_speakers = set(pd.read_csv(processor.test_spk_file, header=None)[0])
            dev_speakers = set(pd.read_csv(processor.dev_spk_file, header=None)[0])

            # Exclude test speakers from dev set
            original_dev_rows = len(dev_df)
            dev_df = dev_df[~dev_df["speaker_id"].isin(test_speakers)]
            excluded_count = original_dev_rows - len(dev_df)
            log.info("Excluded %d utterances belonging to test speakers from dev set.", excluded_count)

            # Validate speaker set relationships
            dev_enroll_overlap = dev_speakers.intersection(enroll_speakers)
            test_enroll_overlap = test_speakers.intersection(enroll_speakers)
            test_dev_overlap = test_speakers.intersection(dev_speakers)
            
            assert not dev_enroll_overlap, f"Dev and enrollment speakers should not overlap! Found {len(dev_enroll_overlap)} overlapping speakers."
            assert test_enroll_overlap, f"Enrollment and test speakers should overlap! Found {len(test_enroll_overlap)} overlapping speakers, expected more."
            assert not test_dev_overlap, f"Test and dev speakers should not overlap! Found {len(test_dev_overlap)} overlapping speakers."

            log.info(f"Speaker set validation passed: {len(test_enroll_overlap)} enrollment speakers overlap with test set")

            dev_df.to_csv(processor.dev_metadata_file, sep=dataset.sep, index=False)
        
        if core_files_exist:
            processor = self._build_processor(dataset)
            dev_df = pd.read_csv(dataset.dev_metadata_file, sep=dataset.sep)
        
        speaker_lookup_df = processor.generate_speaker_metadata(dev_df)
        speaker_lookup_df.to_csv(dataset.speaker_lookup, sep=dataset.sep, index=False)

        dev_df, speaker_lookup_df = self.csv_processor.process(
            dataset_files=[str(processor.dev_metadata_file)],
            spks_metadata_paths=[dataset.speaker_lookup],
            verbose=dataset.verbose,
        )

        dev_df.to_csv(processor.dev_metadata_file, sep=dataset.sep, index=False)
        log.info("Saved metadata to %s", processor.dev_metadata_file)
        speaker_lookup_df.to_csv(dataset.speaker_lookup, sep=dataset.sep, index=False)
        log.info("Saved speaker lookup to %s", dataset.speaker_lookup)

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
        log.info(
            "Saved train and val csvs to %s and %s",
            dataset.train_csv_file,
            dataset.val_csv_file,
        )

        extras: Dict[str, Any] = {"core_files_copied": core_files_exist}

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
            use_pre_segmentation=dataset.use_pre_segmentation,
            segment_duration=dataset.segment_duration,
            segment_overlap=dataset.segment_overlap,
            min_segment_duration=dataset.min_segment_duration,
        )
            
        return processor

