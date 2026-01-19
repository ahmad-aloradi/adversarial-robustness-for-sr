"""GPU-native audio augmentation using torch-audiomentations.

Simplified augmentation pipeline leveraging built-in torch-audiomentations transforms
for GPU-native execution with significant speedup over SpeechBrain's CPU-based Augmenter.

Key design decisions:
- Uses built-in torch-audiomentations transforms where available (BandStopFilter, SpliceOut)
- Assumes mono audio (single channel) for simplicity and performance
- Supports RIR amplitude normalization without amplifying noise levels
- Uses per_example mode by default for independent sample augmentation
"""

import csv
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from torch import nn

from torch_audiomentations import (
    AddBackgroundNoise,
    ApplyImpulseResponse,
)

from src import utils
log = utils.get_pylogger(__name__)


def _check_mono(waveforms: torch.Tensor, context: str = "") -> None:
    """Verify waveforms are mono (single channel).

    Args:
        waveforms: (batch, channels, time) tensor
        context: Optional context string for error message
    """
    if waveforms.ndim == 3 and waveforms.shape[1] != 1:
        raise ValueError(
            f"{context}Expected mono audio (1 channel), got {waveforms.shape[1]} channels. "
            f"Shape: {waveforms.shape}"
        )


def load_paths_from_csv(csv_file: str | Path) -> list[str]:
    """Parse SpeechBrain-style annotation CSV and extract audio file paths.

    Looks for 'wav' or 'filepath' columns, or any column ending with .wav/.flac.

    Args:
        csv_file: Path to CSV file

    Returns:
        List of audio file paths
    """
    csv_file = Path(csv_file)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV not found: {csv_file}")

    paths = []
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "wav" in row:
                paths.append(row["wav"])
            elif "filepath" in row:
                paths.append(row["filepath"])
            else:
                for v in row.values():
                    if v.endswith((".wav", ".flac")):
                        paths.append(v)
                        break

    if not paths:
        raise ValueError(f"No audio paths found in {csv_file}")

    return paths


class SpeedPerturb(nn.Module):
    """Speed perturbation via resampling (GPU-native).

    Applies random speed factors by resampling audio. Commonly used factors
    are 0.9, 1.0, 1.1 (wespeaker-style) for speaker verification.

    Args:
        sample_rate: Audio sample rate
        speed_factors: Speed factors to randomly choose from
        p: Probability of applying perturbation per sample
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        speed_factors: list[float] | None = None,
        p: float = 0.5,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.speed_factors = speed_factors or [0.9, 1.0, 1.1]
        self.p = p

        # Pre-create resamplers for each non-unity speed factor
        self._resamplers: dict[float, torchaudio.transforms.Resample] = {}
        for speed in self.speed_factors:
            if speed != 1.0:
                self._resamplers[speed] = torchaudio.transforms.Resample(
                    orig_freq=int(sample_rate * speed),
                    new_freq=sample_rate,
                )

    def forward(self, waveforms: torch.Tensor, sample_rate: int | None = None) -> torch.Tensor:
        """Apply speed perturbation.

        Args:
            waveforms: (batch, 1, time) mono audio tensor
            sample_rate: Unused, for API compatibility

        Returns:
            Speed-perturbed waveforms (same shape as input)
        """
        _check_mono(waveforms, "SpeedPerturb: ")
        batch_size, channels, length = waveforms.shape
        assert channels == 1, "SpeedPerturb only supports mono audio."
        device = waveforms.device

        # Generate random decisions for entire batch
        apply_mask = torch.rand(batch_size, device=device) <= self.p
        if not apply_mask.any():
            return waveforms

        speed_indices = torch.randint(len(self.speed_factors), (batch_size,), device=device)
        output = waveforms.clone()

        # Process samples grouped by speed factor for efficiency
        for speed_idx, speed in enumerate(self.speed_factors):
            if speed == 1.0:
                continue

            mask = apply_mask & (speed_indices == speed_idx)
            if not mask.any():
                continue

            indices = mask.nonzero(as_tuple=True)[0]
            resampler = self._resamplers[speed].to(device)
            resampled = resampler(waveforms[indices])

            # Adjust to original length (pad or truncate)
            new_length = resampled.shape[-1]
            if new_length < length:
                resampled = F.pad(resampled, (0, length - new_length))
            elif new_length > length:
                resampled = resampled[..., :length]

            output[indices] = resampled

        return output


class NormalizedReverb(nn.Module):
    """Reverb with peak amplitude normalization.

    Wraps ApplyImpulseResponse to prevent energy boost from RIR convolution.
    Normalizes output to match input peak amplitude.

    Args:
        ir_csv: Path to CSV with IR paths
        sample_rate: Audio sample rate
        p: Probability of applying reverb
        normalize: Whether to normalize amplitude after convolution
    """

    def __init__(
        self,
        ir_csv: str | None = None,
        sample_rate: int = 16000,
        p: float = 0.5,
        normalize: bool = True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.normalize = normalize
        ir_paths = load_paths_from_csv(ir_csv)

        self._reverb = ApplyImpulseResponse(
            ir_paths=ir_paths,
            p=p,
            sample_rate=sample_rate,
            mode="per_example",
            output_type="tensor",
        )

    def forward(self, waveforms: torch.Tensor, sample_rate: int | None = None) -> torch.Tensor:
        """Apply reverb with optional amplitude normalization."""
        sr = sample_rate or self.sample_rate

        if not self.normalize:
            return self._reverb(waveforms, sample_rate=sr)

        # Store original peak amplitudes
        orig_peaks = waveforms.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)

        # Apply reverb
        output = self._reverb(waveforms, sample_rate=sr)

        # Normalize to original peak amplitude
        new_peaks = output.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        return output * (orig_peaks / new_peaks)


class NoiseFromCSV(nn.Module):
    """Background noise augmentation from CSV file paths.

    Wrapper around AddBackgroundNoise that loads paths from a SpeechBrain-style CSV.

    Args:
        noise_csv: Path to CSV with noise file paths
        min_snr_db: Minimum SNR in dB
        max_snr_db: Maximum SNR in dB
        p: Probability of applying noise
    """

    def __init__(
        self,
        noise_csv: str,
        min_snr_db: float = 0.0,
        max_snr_db: float = 15.0,
        p: float = 0.5,
    ):
        super().__init__()
        noise_paths = load_paths_from_csv(noise_csv)

        self._noise = AddBackgroundNoise(
            background_paths=noise_paths,
            min_snr_in_db=min_snr_db,
            max_snr_in_db=max_snr_db,
            p=p,
            mode="per_example",
        )

    def forward(self, waveforms: torch.Tensor, sample_rate: int | None = None) -> torch.Tensor:
        """Apply background noise."""
        return self._noise(waveforms, sample_rate=sample_rate)


class GPUAugmenter(nn.Module):
    """GPU-native audio augmentation pipeline for speaker verification.

    Generic pipeline that applies an ordered list of transforms.

    Args:
        sample_rate: Audio sample rate (default: 16000)
        transforms: Ordered list of pre-instantiated transforms (via Hydra _target_)
        concat_original: Concatenate original with augmented (doubles batch)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        transforms: list[nn.Module] | None = None,
        concat_original: bool = True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.concat_original = concat_original

        # Validate and reorder transforms if needed
        if transforms:
            transforms = self._validate_and_reorder_transforms(transforms)

        self.transforms = nn.ModuleList(transforms) if transforms else nn.ModuleList()

    def _validate_and_reorder_transforms(
        self, transforms: list[nn.Module]
    ) -> list[nn.Module]:
        """Check transform order and reorder if SpeedPerturb/NormalizedReverb are misplaced.

        Correct order:
        1. SpeedPerturb (first - changes signal length)
        2. NormalizedReverb (before noise - to avoid amplifying it)
        3. Everything else (noise, filters, etc.)
        """
        speed_perturb = None
        normalized_reverb = None
        others = []

        for t in transforms:
            if isinstance(t, SpeedPerturb):
                speed_perturb = t
            elif isinstance(t, NormalizedReverb):
                normalized_reverb = t
            else:
                others.append(t)

        # Build correct order
        correct_order = []
        if speed_perturb is not None:
            correct_order.append(speed_perturb)
        if normalized_reverb is not None:
            correct_order.append(normalized_reverb)
        correct_order.extend(others)

        # Check if reordering occurred
        if correct_order != transforms:
            log.warning(
                "GPUAugmenter: Transforms were reordered. "
                "SpeedPerturb must be first (changes signal length), "
                "NormalizedReverb must come before noise (to avoid amplifying it). "
                f"New order: {[type(t).__name__ for t in correct_order]}"
            )

        return correct_order

    def forward(
        self, waveforms: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply augmentation pipeline.

        Args:
            waveforms: (batch, time) or (batch, 1, time) mono audio tensor
            lengths: Normalized lengths (0-1) for each sample

        Returns:
            Tuple of (augmented_waveforms, lengths). If concat_original=True,
            both are doubled in batch dimension.
        """
        # Ensure 3D: (batch, 1, time)
        needs_squeeze = False
        if waveforms.ndim == 2:
            waveforms = waveforms.unsqueeze(1)
            needs_squeeze = True

        _check_mono(waveforms, "GPUAugmenter: ")

        # Store original for potential concatenation
        original = waveforms

        # Apply transforms in order
        for transform in self.transforms:
            waveforms = transform(waveforms, sample_rate=self.sample_rate)

        if needs_squeeze:
            waveforms = waveforms.squeeze(1)
            original = original.squeeze(1)

        if self.concat_original:
            waveforms = torch.cat([original, waveforms], dim=0)
            lengths = torch.cat([lengths, lengths], dim=0)

        return waveforms, lengths

    def replicate_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Double labels when concat_original=True."""
        if self.concat_original:
            return torch.cat([labels, labels], dim=0)
        return labels
