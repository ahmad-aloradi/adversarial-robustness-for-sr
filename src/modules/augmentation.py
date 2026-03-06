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

    Randomly selects a speed factor per sample with probability ``p``.

    When ``virtual_speakers=True``, returns ``(waveforms, speed_indices)``
    so the caller can remap class labels at runtime — each (speaker, speed)
    pair becomes a unique virtual class.  1.0 is always moved to index 0
    internally so that ``speed_index * N + class_id`` keeps labels [0, N)
    for unperturbed audio, matching validation labels regardless of the
    ordering in the config.

    Args:
        sample_rate: Audio sample rate
        speed_factors: Speed factors to choose from (must include 1.0
            when virtual_speakers=True; order does not matter)
        p: Probability of applying perturbation
        virtual_speakers: If True, return speed indices for label remapping
        num_base_classes: Number of original speaker classes (required
            when virtual_speakers=True)
    """

    def __init__(
        self,
        sample_rate: int,
        speed_factors: list[float],
        virtual_speakers: bool,
        num_base_classes: int,
        p: float = 1.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.speed_factors = list(speed_factors or [0.9, 1.0, 1.1])
        self.p = p
        self.virtual_speakers = virtual_speakers

        if virtual_speakers:
            assert (
                num_base_classes is not None
            ), "num_base_classes is required when virtual_speakers=True"
            assert (
                1.0 in self.speed_factors
            ), "speed_factors must include 1.0 for virtual speakers"
            # IMPORTANT: Move 1.0 to index 0 so that speed_index=0 → label
            # offset 0 → validation labels [0, N) stay aligned.
            unity_pos = self.speed_factors.index(1.0)
            if unity_pos != 0:
                self.speed_factors[0], self.speed_factors[unity_pos] = (
                    self.speed_factors[unity_pos],
                    self.speed_factors[0],
                )
            self.num_base_classes = num_base_classes
            self._unity_index = 0

        # Pre-create resamplers for each non-unity speed factor
        self._resamplers: dict[float, torchaudio.transforms.Resample] = {}
        for speed in self.speed_factors:
            if speed != 1.0:
                self._resamplers[speed] = torchaudio.transforms.Resample(
                    orig_freq=int(sample_rate * speed),
                    new_freq=sample_rate,
                )

    def forward(
        self,
        waveforms: torch.Tensor,
        sample_rate: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply speed perturbation.

        Args:
            waveforms: (batch, 1, time) mono audio tensor
            sample_rate: Unused, for API compatibility

        Returns:
            Speed-perturbed waveforms (same shape as input), or
            tuple of (waveforms, speed_indices) when virtual_speakers=True.
        """
        _check_mono(waveforms, "SpeedPerturb: ")
        batch_size, channels, length = waveforms.shape
        assert channels == 1, "SpeedPerturb only supports mono audio."
        device = waveforms.device

        # TODO: maybe deprecate p and apply_mask (?) -> control through speed factors
        apply_mask = torch.rand(batch_size, device=device) <= self.p
        speed_indices = torch.randint(
            len(self.speed_factors), (batch_size,), device=device
        )

        # Samples not selected for perturbation get unity speed
        if self.virtual_speakers:
            speed_indices[~apply_mask] = self._unity_index

        if not apply_mask.any():
            if self.virtual_speakers:
                return waveforms, speed_indices
            return waveforms

        output = waveforms.clone()

        for speed_idx, speed in enumerate(self.speed_factors):
            if speed == 1.0:
                continue

            mask = apply_mask & (speed_indices == speed_idx)
            if not mask.any():
                continue

            indices = mask.nonzero(as_tuple=True)[0]
            resampler = self._resamplers[speed].to(device)
            resampled = resampler(waveforms[indices])

            new_length = resampled.shape[-1]
            if new_length < length:
                resampled = F.pad(resampled, (0, length - new_length))
            elif new_length > length:
                resampled = resampled[..., :length]

            output[indices] = resampled

        if self.virtual_speakers:
            return output, speed_indices
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
        ir_csv: str,
        sample_rate: int,
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

    def forward(
        self, waveforms: torch.Tensor, sample_rate: int | None = None
    ) -> torch.Tensor:
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

    def forward(
        self, waveforms: torch.Tensor, sample_rate: int | None = None
    ) -> torch.Tensor:
        """Apply background noise."""
        return self._noise(waveforms, sample_rate=sample_rate)


class MutuallyExclusive(nn.Module):
    """Apply exactly one randomly-selected transform per sample.

    Ensures transforms are mutually exclusive — e.g., noise OR reverb,
    never both. Inner transforms should have p=1.0 since probability
    is controlled by this wrapper.

    Args:
        transforms: List of transforms to choose from
        p: Probability of applying any transform per sample
    """

    def __init__(
        self,
        transforms: list[nn.Module],
        p: float = 1.0,
    ):
        super().__init__()
        if not transforms:
            raise ValueError("MutuallyExclusive requires at least one transform")
        self.transforms = nn.ModuleList(transforms)
        self.p = p

    def forward(
        self, waveforms: torch.Tensor, sample_rate: int | None = None
    ) -> torch.Tensor:
        """Apply exactly one transform per sample.

        Args:
            waveforms: (batch, 1, time) mono audio tensor
            sample_rate: Sample rate passed to inner transforms

        Returns:
            Augmented waveforms (same shape as input)
        """
        batch_size = waveforms.shape[0]
        device = waveforms.device
        n_transforms = len(self.transforms)

        # Decide which samples get augmented
        apply_mask = torch.rand(batch_size, device=device) <= self.p
        if not apply_mask.any():
            return waveforms

        # Assign each active sample a random transform index
        transform_indices = torch.randint(n_transforms, (batch_size,), device=device)

        output = waveforms.clone()

        # Process samples grouped by transform index (batched pattern)
        for t_idx, transform in enumerate(self.transforms):
            mask = apply_mask & (transform_indices == t_idx)
            if not mask.any():
                continue

            indices = mask.nonzero(as_tuple=True)[0]
            augmented = transform(waveforms[indices], sample_rate=sample_rate)
            output[indices] = augmented

        return output


class GPUAugmenter(nn.Module):
    """GPU-native audio augmentation pipeline for speaker verification.

    Generic pipeline that applies an ordered list of transforms.

    Args:
        sample_rate: Audio sample rate (default: 16000)
        transforms: Ordered list of pre-instantiated transforms (via Hydra _target_)
        concat_original: Concatenate original with augmented (doubles batch)
        speed_perturb: Optional SpeedPerturb applied before saving the
            clean copy for concat_original so both copies share the same
            virtual speaker identity.
    """

    def __init__(
        self,
        sample_rate: int,
        transforms: list[nn.Module] | None = None,
        concat_original: bool = True,
        speed_perturb: SpeedPerturb | None = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.concat_original = concat_original
        self.speed_perturb = speed_perturb
        self._speed_indices: torch.Tensor | None = None

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
        2. NormalizedReverb or MutuallyExclusive (before standalone noise)
        3. Everything else (noise, filters, etc.)
        """
        speed_perturb = None
        normalized_reverb = None
        others = []

        for t in transforms:
            if isinstance(t, SpeedPerturb):
                speed_perturb = t
            elif isinstance(t, (NormalizedReverb, MutuallyExclusive)):
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
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
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

        # Speed perturbation applied before saving original so both
        # concat copies share the same speed (virtual speaker identity).
        self._speed_indices = None
        if self.speed_perturb is not None:
            result = self.speed_perturb(waveforms)
            if self.speed_perturb.virtual_speakers:
                waveforms, self._speed_indices = result
            else:
                waveforms = result

        # Store original for potential concatenation
        original = waveforms

        # Apply augmentation transforms in order
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
        """Remap labels for virtual speakers then double for concat_original.

        When speed_perturb has virtual_speakers=True, each (speaker, speed)
        pair gets a unique class:
            virtual_class = speed_index * num_base_classes + original_class
        Because 1.0 is always at index 0, unperturbed audio keeps labels
        in [0, N) — matching validation labels.
        """
        if self._speed_indices is not None:
            labels = (
                self._speed_indices.to(labels.device)
                * self.speed_perturb.num_base_classes
                + labels
            )
        if self.concat_original:
            return torch.cat([labels, labels], dim=0)
        return labels
