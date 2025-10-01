"""Shared interfaces for dataset metadata preparation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import pandas as pd


@dataclass
class SplitArtifacts:
    """Paths to the primary dataset splits and aggregate metadata."""

    train_csv: Path
    val_csv: Path
    speaker_lookup_csv: Path
    dev_csv: Path


@dataclass
class TestArtifacts:
    """Artifacts required for evaluation/evaluation-time dataloaders."""

    verification_csvs: Dict[str, Path] = field(default_factory=dict)
    enroll_frames: Dict[str, pd.DataFrame] = field(default_factory=dict)
    unique_trial_frames: Dict[str, pd.DataFrame] = field(default_factory=dict)


@dataclass
class PreparationResult:
    """Container returned by metadata preparers."""

    splits: SplitArtifacts
    test: TestArtifacts = field(default_factory=TestArtifacts)
    extras: Dict[str, Any] = field(default_factory=dict)


class BaseMetadataPreparer(ABC):
    """Abstract interface for generating metadata artifacts used by datamodules."""

    def __init__(self, dataset_cfg: Any, csv_processor: Any):
        self.dataset_cfg = dataset_cfg
        self.csv_processor = csv_processor

    @abstractmethod
    def prepare(self) -> PreparationResult:
        """Generate or load metadata and return relevant artifacts."""
        raise NotImplementedError
