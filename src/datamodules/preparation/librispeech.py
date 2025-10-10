"""Metadata preparation flow for the LibriSpeech datamodule."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from src import utils

from src.datamodules.components.librispeech.librispeech_prep import (
    generate_csvs,
    write_dataset_csv,
)
from src.datamodules.components.utils import CsvProcessor

from .base import BaseMetadataPreparer, PreparationResult, SplitArtifacts
from .snapshot_keys import LIBRISPEECH_COMPARABLE_KEYS

log = utils.get_pylogger(__name__)


class LibrispeechMetadataPreparer(BaseMetadataPreparer):
    """Encapsulates metadata preparation for LibriSpeech datasets."""

    COMPARABLE_KEYS: tuple[str, ...] = LIBRISPEECH_COMPARABLE_KEYS

    def __init__(self, dataset_cfg: Any, csv_processor: CsvProcessor):
        super().__init__(dataset_cfg=dataset_cfg, csv_processor=csv_processor)

    def prepare(self) -> PreparationResult:
        dataset = self.dataset_cfg
        artifacts_dir = Path(dataset.artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        expected_snapshot = self.build_config_snapshot()

        required_files = [
            Path(dataset.train_csv_exp_filepath),
            Path(dataset.dev_csv_exp_filepath),
            Path(dataset.test_csv_exp_filepath),
            Path(dataset.speaker_csv_exp_filepath),
        ]

        snapshot_path = artifacts_dir / self.CONFIG_SNAPSHOT_FILENAME
        reuse_artifacts = False
        if all(path.exists() for path in required_files) and snapshot_path.is_file():
            cached_snapshot = self.load_config_snapshot(snapshot_path)
            if cached_snapshot == expected_snapshot:
                log.info(f"Reusing existing LibriSpeech metadata in {artifacts_dir}")
                reuse_artifacts = True
            else:
                diff = self.diff_config_snapshots(expected_snapshot, cached_snapshot)
                log.info(f"LibriSpeech config mismatch detected; regenerating artifacts. Differences: {diff}")

        if reuse_artifacts:
            splits = SplitArtifacts(
                train_csv=Path(dataset.train_csv_exp_filepath),
                val_csv=Path(dataset.dev_csv_exp_filepath),
                speaker_lookup_csv=Path(dataset.speaker_csv_exp_filepath),
                dev_csv=Path(dataset.dev_csv_exp_filepath),
            )
            extras: Dict[str, Path] = {"test_csv": Path(dataset.test_csv_exp_filepath)}
            return PreparationResult(splits=splits, extras=extras)

        dfs_data, df_speaker = generate_csvs(
            dataset,
            delimiter=dataset.sep,
            save_csv=dataset.save_csv,
        )

        write_dataset_csv(df_speaker, dataset.speaker_csv_exp_filepath, sep=dataset.sep)
        for df_path, df in dfs_data.items():
            write_dataset_csv(df, df_path, sep=dataset.sep)

        updated_dev_csv, speaker_lookup_csv = self.csv_processor.process(
            dataset_files=[dataset.train_csv_exp_filepath, dataset.dev_csv_exp_filepath],
            spks_metadata_paths=[dataset.speaker_csv_path],
            verbose=dataset.verbose,
        )

        write_dataset_csv(speaker_lookup_csv, dataset.speaker_csv_exp_filepath, sep=dataset.sep)

        split_map = {
            dataset.train_csv_exp_filepath: "train",
            dataset.dev_csv_exp_filepath: "dev",
            dataset.test_csv_exp_filepath: "test",
        }
        for path, split_name in split_map.items():
            filtered = updated_dev_csv[updated_dev_csv.split == split_name]
            write_dataset_csv(filtered, path, sep=dataset.sep)

        splits = SplitArtifacts(
            train_csv=Path(dataset.train_csv_exp_filepath),
            val_csv=Path(dataset.dev_csv_exp_filepath),
            speaker_lookup_csv=Path(dataset.speaker_csv_exp_filepath),
            dev_csv=Path(dataset.dev_csv_exp_filepath),
        )

        extras: Dict[str, Path] = {"test_csv": Path(dataset.test_csv_exp_filepath)}

        return PreparationResult(splits=splits, extras=extras)
