"""Canonical definition of snapshot comparison keys for each dataset.

This module serves as the single source of truth for which configuration parameters
should be compared when deciding whether to reuse cached metadata artifacts.

Both CLI scripts (components/*_prep.py) and preparers (preparation/*.py) import
from here to ensure consistency.
"""

from typing import Tuple

# CNCeleb dataset snapshot keys
CNCELEB_COMPARABLE_KEYS: Tuple[str, ...] = (
    "cnceleb1",
    "cnceleb2",
    "sep",
    "sample_rate",
    "use_pre_segmentation",
    "segment_duration",
    "segment_overlap",
    "min_segment_duration",
)

# VoxCeleb dataset snapshot keys
VOXCELEB_COMPARABLE_KEYS: Tuple[str, ...] = (
    "wav_dir",
    "voxceleb_metadata",
    "sep",
    "use_pre_segmentation",
    "segment_duration",
    "segment_overlap",
    "min_segment_duration",
    "min_duration",
    "veri_test_filenames",
)

# LibriSpeech dataset snapshot keys
LIBRISPEECH_COMPARABLE_KEYS: Tuple[str, ...] = (
    "metadata_path",
    "train_dir",
    "dev_dir",
    "test_dir",
    "speaker_filepath",
    "audio_file_type",
    "annotation_format",
    "sep",
    "use_pre_segmentation",
    "segment_duration",
    "segment_overlap",
    "min_segment_duration",
    "max_duration",
)
