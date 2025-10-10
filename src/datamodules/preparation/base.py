"""Shared interfaces for dataset metadata preparation."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from omegaconf import OmegaConf
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

    CONFIG_SNAPSHOT_FILENAME = "prep_config.json"
    COMPARABLE_KEYS: Sequence[str] = ()

    def __init__(self, dataset_cfg: Any, csv_processor: Any):
        self.dataset_cfg = dataset_cfg
        self.csv_processor = csv_processor

    @abstractmethod
    def prepare(self) -> PreparationResult:
        """Generate or load metadata and return relevant artifacts."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Configuration snapshot helpers
    # ------------------------------------------------------------------

    def build_config_snapshot(self) -> Dict[str, Any]:
        """Create the canonical snapshot of the dataset configuration."""
        mapping = _to_plain_dict(self.dataset_cfg)
        snapshot = build_config_snapshot_from_mapping(mapping, self.COMPARABLE_KEYS)
        return snapshot

    @classmethod
    def load_config_snapshot(cls, path: Path) -> Dict[str, Any]:
        return load_config_snapshot(path)

    @classmethod
    def diff_config_snapshots(
        cls, expected: Mapping[str, Any], observed: Mapping[str, Any]
    ) -> Dict[str, Any]:
        return diff_config_snapshots(expected, observed)

    def save_config_snapshot(self, target: Path, snapshot: Mapping[str, Any]) -> Path:
        target_path = Path(target)
        if target_path.is_dir():
            target_path = target_path / self.CONFIG_SNAPSHOT_FILENAME
        return save_config_snapshot(snapshot, target_path)


CONFIG_SNAPSHOT_FILENAME = BaseMetadataPreparer.CONFIG_SNAPSHOT_FILENAME


# ----------------------------------------------------------------------
# Module-level helpers used across preparers and standalone prep scripts
# ----------------------------------------------------------------------

def _normalize_config_value(value: Any) -> Any:
    if isinstance(value, Path):
        return value.name
    if isinstance(value, str):
        # Skip URLs
        if "://" in value:
            return value
        # Try to detect if it's a path-like string
        try:
            path_obj = Path(value)
            # If it has multiple parts or exists as a path, extract basename
            if len(path_obj.parts) > 1 or path_obj.exists():
                return path_obj.name
        except (ValueError, OSError):
            # Not a valid path, keep as-is
            pass
        return value
    if isinstance(value, dict):
        return {str(k): _normalize_config_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_config_value(v) for v in value]
    if isinstance(value, set):
        return sorted(_normalize_config_value(v) for v in value)
    return value


def _to_plain_dict(cfg: Any) -> Dict[str, Any]:
    """Convert config object to plain dict.
    
    Supports OmegaConf configs, dataclasses, and dict-like objects.
    """
    if OmegaConf.is_config(cfg):
        container = OmegaConf.to_container(cfg, resolve=True)
    elif is_dataclass(cfg):
        container = asdict(cfg)
    elif isinstance(cfg, Mapping):
        container = dict(cfg)
    else:
        raise TypeError(
            "Unsupported configuration object type for serialization: "
            f"{type(cfg)!r}"
        )
    return container


def build_config_snapshot_from_mapping(
    mapping: Mapping[str, Any], keys: Sequence[str]
) -> Dict[str, Any]:
    normalized = _normalize_config_value(dict(mapping))
    if not keys:
        return normalized  # type: ignore[return-value]

    snapshot: Dict[str, Any] = {}
    for key in keys:
        if key in normalized:
            snapshot[key] = normalized[key]
    return snapshot


def save_config_snapshot(snapshot: Mapping[str, Any], target_path: Path) -> Path:
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as fp:
        json.dump(snapshot, fp, indent=2, sort_keys=True)
    return target_path


def load_config_snapshot(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    return _normalize_config_value(data)


def diff_config_snapshots(
    expected: Mapping[str, Any], observed: Mapping[str, Any]
) -> Dict[str, Any]:
    diff: Dict[str, Any] = {}
    all_keys = sorted(set(expected) | set(observed))
    for key in all_keys:
        if expected.get(key) != observed.get(key):
            diff[key] = {"expected": expected.get(key), "observed": observed.get(key)}
    return diff


def build_snapshot_from_locals(comparable_keys: Sequence[str], **kwargs) -> Dict[str, Any]:
    """Build snapshot dict from local variables, extracting only comparable keys.
    
    This is a convenience helper for CLI scripts that need to build snapshots
    from mixed sources (args, local vars, processor attributes).
    
    Args:
        comparable_keys: Sequence of keys to extract (e.g., DATASET_COMPARABLE_KEYS)
        **kwargs: Local variables to extract from
        
    Returns:
        dict: Snapshot containing only keys defined in comparable_keys
    """
    return {key: kwargs[key] for key in comparable_keys if key in kwargs}


def read_hydra_config(config_path: str = "conf", config_name: str = "config", overrides: list = None):
    """Load Hydra configuration from YAML files.
    
    Args:
        config_path: Relative path to config directory
        config_name: Name of the config file (without .yaml extension)
        overrides: List of config overrides (e.g., ["paths.data_dir=/path/to/data"])
        
    Returns:
        OmegaConf configuration object
    """
    from hydra import initialize, compose
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=config_name, overrides=overrides)
        return cfg
