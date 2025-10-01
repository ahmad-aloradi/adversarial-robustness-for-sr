"""Metadata preparation flow for the LibriSpeech datamodule."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from src.datamodules.components.librispeech.librispeech_prep import (
    generate_csvs,
    write_dataset_csv,
)
from src.datamodules.components.utils import CsvProcessor

from .base import BaseMetadataPreparer, PreparationResult, SplitArtifacts


class LibrispeechMetadataPreparer(BaseMetadataPreparer):
    """Encapsulates metadata preparation for LibriSpeech datasets."""

    def __init__(self, dataset_cfg: Any, csv_processor: CsvProcessor):
        super().__init__(dataset_cfg=dataset_cfg, csv_processor=csv_processor)

    def prepare(self) -> PreparationResult:
        dataset = self.dataset_cfg

        dfs_data, df_speaker = generate_csvs(
            dataset,
            delimiter=dataset.sep,
            save_csv=dataset.save_csv,
        )

        os.makedirs(dataset.artifacts_dir, exist_ok=True)
        write_dataset_csv(df_speaker, dataset.speaker_csv_exp_filepath, sep=dataset.sep)
        for df_path, df in dfs_data.items():
            write_dataset_csv(df, df_path, sep=dataset.sep)

        updated_dev_csv, speaker_lookup_csv = self.csv_processor.process(
            dataset_files=list(dfs_data.keys()),
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
