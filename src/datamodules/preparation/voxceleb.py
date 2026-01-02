"""Metadata preparation flow for the VoxCeleb datamodule."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, Literal

import pandas as pd

from src import utils
from src.datamodules.components.voxceleb.voxceleb_prep import (
    VoxCelebProcessor,
    VoxCelebTestFilter,
)
from src.datamodules.components.utils import CsvProcessor
from src.datamodules.components.common import VoxcelebDefaults, get_dataset_class

from .base import BaseMetadataPreparer, PreparationResult, SplitArtifacts, TestArtifacts
from .snapshot_keys import VOXCELEB_COMPARABLE_KEYS

log = utils.get_pylogger(__name__)
DATASET_DEFAULTS = VoxcelebDefaults()
DATASET_CLS, DF_COLS = get_dataset_class(DATASET_DEFAULTS.dataset_name)


def _extract_enroll_test(df: pd.DataFrame, mode: Literal["enroll", "test"]):
    """Vectorized pandas implementation with speaker consistency validation."""
    path_col = f"{mode}_path"
    enroll_columns = [col for col in df.columns if col.startswith(f"{mode}_") and col != path_col]

    if not enroll_columns:
        return df[[path_col]].drop_duplicates().reset_index(drop=True)

    grouped = df.groupby(path_col)

    nunique_per_group = grouped[enroll_columns].nunique()
    is_constant = nunique_per_group == 1

    non_constant_mask = ~is_constant.all(axis=1)
    if non_constant_mask.any():
        problematic_paths = non_constant_mask[non_constant_mask].index.tolist()
        first_path = problematic_paths[0]
        inconsistent_cols = (
            nunique_per_group.loc[first_path][nunique_per_group.loc[first_path] > 1].index.tolist()
        )
        raise ValueError(
            f"Inconsistent data found for {mode}_path '{first_path}'. "
            "Expected all rows with the same path to have identical values, "
            f"but found multiple values in columns: {inconsistent_cols}. "
            "This violates the assumption that same path = same speaker."
        )

    results_df = grouped[enroll_columns].first().reset_index()
    return results_df


class VoxCelebMetadataPreparer(BaseMetadataPreparer):
    """Encapsulates metadata preparation for VoxCeleb datasets."""

    COMPARABLE_KEYS: tuple[str, ...] = VOXCELEB_COMPARABLE_KEYS

    def __init__(self, dataset_cfg, csv_processor: CsvProcessor):
        super().__init__(dataset_cfg=dataset_cfg, csv_processor=csv_processor)

    def prepare(self) -> PreparationResult:
        dataset = self.dataset_cfg
        artifacts_dir = Path(dataset.voxceleb_artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        expected_snapshot = self.build_config_snapshot()

        processor = VoxCelebProcessor(
            root_dir=dataset.data_dir,
            verbose=dataset.verbose,
            artifcats_dir=artifacts_dir,
            sep=dataset.sep,
            use_pre_segmentation=dataset.use_pre_segmentation,
            segment_duration=dataset.segment_duration,
            segment_overlap=dataset.segment_overlap,
            min_segment_duration=dataset.min_segment_duration,
            vad=dataset.get("vad", None),
        )

        required_files = [
            processor.dev_metadata_file,
            processor.vox_metadata,
            processor.preprocess_stats_file,
            Path(dataset.speaker_lookup),
            Path(dataset.train_csv_file),
            Path(dataset.val_csv_file),
        ] + [Path(path) for path in dataset.veri_test_output_paths.values()]

        snapshot_path = artifacts_dir / self.CONFIG_SNAPSHOT_FILENAME
        reuse_artifacts = False
        if all(path.exists() for path in required_files) and snapshot_path.is_file():
            cached_snapshot = self.load_config_snapshot(snapshot_path)
            if self.snapshots_match(expected_snapshot, cached_snapshot):
                log.info(f"Reusing existing VoxCeleb artifacts in {artifacts_dir}")
                reuse_artifacts = True
            else:
                diff = self.diff_config_snapshots(expected_snapshot, cached_snapshot)
                log.info(f"VoxCeleb config mismatch detected; regenerating artifacts. Differences: {diff}")

        if reuse_artifacts:
            verification_csvs = {
                name: Path(path) for name, path in dataset.veri_test_output_paths.items()
            }
            enroll_data_dict: Dict[str, pd.DataFrame] = {}
            unique_trial_data_dict: Dict[str, pd.DataFrame] = {}
            for name, csv_path in verification_csvs.items():
                veri_df = pd.read_csv(csv_path, sep=dataset.sep)
                enroll_data_dict[name] = _extract_enroll_test(veri_df, mode="enroll")
                unique_trial_data_dict[name] = _extract_enroll_test(veri_df, mode="test")

            splits = SplitArtifacts(
                train_csv=Path(dataset.train_csv_file),
                val_csv=Path(dataset.val_csv_file),
                speaker_lookup_csv=Path(dataset.speaker_lookup),
                dev_csv=Path(processor.dev_metadata_file),
            )

            test_artifacts = TestArtifacts(
                verification_csvs=verification_csvs,
                enroll_frames=enroll_data_dict,
                unique_trial_frames=unique_trial_data_dict,
            )

            return PreparationResult(splits=splits, test=test_artifacts, extras={})

        copied_from_base = False
        base_search_dir = getattr(dataset, "base_search_dir", None)
        if base_search_dir:
            base_dir = Path(base_search_dir).expanduser().resolve()
            base_required = [
                base_dir / processor.dev_metadata_file.name,
                base_dir / processor.vox_metadata.name,
                base_dir / processor.preprocess_stats_file.name,
            ]
            base_snapshot_path = base_dir / self.CONFIG_SNAPSHOT_FILENAME
            if base_dir.exists() and all(path.exists() for path in base_required) and base_snapshot_path.is_file():
                cached_snapshot = self.load_config_snapshot(base_snapshot_path)
                if self.snapshots_match(expected_snapshot, cached_snapshot):
                    log.info(f"Copying VoxCeleb cached metadata from {base_dir}")
                    for src_path in base_required:
                        dest_path = artifacts_dir / src_path.name
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src_path, dest_path)
                    copied_from_base = True
                else:
                    diff = self.diff_config_snapshots(expected_snapshot, cached_snapshot)
                    log.info(f"Cached VoxCeleb artifacts in {base_dir} have mismatched config; regenerating. Differences: {diff}")
            elif base_dir.exists():
                missing = [src.name for src in base_required if not src.exists()]
                if missing:
                    log.info(f"Skipping cached VoxCeleb artifacts in {base_dir} due to missing files: {', '.join(missing)}")

        if not copied_from_base:
            processor.generate_metadata(
                min_duration=dataset.min_duration,
                save_df=dataset.save_csv,
            )

        test_filter = VoxCelebTestFilter(root_dir=dataset.data_dir, verbose=dataset.verbose)

        # Single loop: extract speakers and enrich verification files
        all_test_speakers = set()
        verification_csvs: Dict[str, Path] = {}
        enroll_data_dict: Dict[str, pd.DataFrame] = {}
        unique_trial_data_dict: Dict[str, pd.DataFrame] = {}

        for test_filename in dataset.veri_test_filenames:
            # Download and extract speakers (reads file once)
            test_speakers, veri_df = test_filter.get_test_speakers(test_filename)
            all_test_speakers.update(test_speakers)
            
            # Enrich with metadata (reuses already-loaded DataFrame)
            test_df = VoxCelebProcessor.enrich_verification_file(
                veri_test_path=None,
                metadata_path=dataset.metadata_csv_file,
                output_path=dataset.veri_test_output_paths[test_filename],
                sep=dataset.sep,
                veri_df=veri_df
            )

            # Store results
            verification_csvs[test_filename] = Path(dataset.veri_test_output_paths[test_filename])
            enroll_data_dict[test_filename] = _extract_enroll_test(test_df, mode="enroll")
            unique_trial_data_dict[test_filename] = _extract_enroll_test(test_df, mode="test")

        # Filter dev metadata to exclude all test speakers
        dev_metadata = pd.read_csv(str(processor.dev_metadata_file), sep=dataset.sep)
        filtered_dev_metadata = test_filter.filter_dev_metadata(dev_metadata, all_test_speakers)

        VoxCelebProcessor.save_csv(
            filtered_dev_metadata,
            str(processor.dev_metadata_file),
            sep=dataset.sep,
        )

        # Process and update metadata with speaker lookup
        updated_filtered_dev_metadata, speaker_lookup_csv = self.csv_processor.process(
            dataset_files=[str(processor.dev_metadata_file)],
            spks_metadata_paths=[dataset.metadata_csv_file],
            verbose=dataset.verbose,
        )

        VoxCelebProcessor.save_csv(
            updated_filtered_dev_metadata,
            str(processor.dev_metadata_file),
            sep=dataset.sep,
        )
        VoxCelebProcessor.save_csv(speaker_lookup_csv, dataset.speaker_lookup)

        # Split into train/val
        CsvProcessor.split_dataset(
            df=updated_filtered_dev_metadata,
            train_ratio=dataset.train_ratio,
            save_csv=dataset.save_csv,
            speaker_overlap=dataset.speaker_overlap,
            speaker_id_col=DATASET_CLS.SPEAKER_ID,
            train_csv=dataset.train_csv_file,
            val_csv=dataset.val_csv_file,
            sep=dataset.sep,
            seed=dataset.seed,
        )

        splits = SplitArtifacts(
            train_csv=Path(dataset.train_csv_file),
            val_csv=Path(dataset.val_csv_file),
            speaker_lookup_csv=Path(dataset.speaker_lookup),
            dev_csv=Path(processor.dev_metadata_file),
        )

        test_artifacts = TestArtifacts(
            verification_csvs=verification_csvs,
            enroll_frames=enroll_data_dict,
            unique_trial_frames=unique_trial_data_dict,
        )

        return PreparationResult(splits=splits, test=test_artifacts, extras={})
