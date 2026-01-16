"""GPU-native audio augmentation wrapper using torch-audiomentations.

Drop-in replacement for SpeechBrain's Augmenter with identical augmentations
but GPU-native execution for significant speedup.
"""
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from torch_audiomentations import Compose, AddBackgroundNoise, ApplyImpulseResponse


class DropChunk(nn.Module):
    """Drop random chunks of audio (GPU-native, matches SpeechBrain's DropChunk)."""

    def __init__(
        self,
        drop_length_low: int = 1000,
        drop_length_high: int = 2000,
        drop_count_low: int = 1,
        drop_count_high: int = 5,
        p: float = 0.5,
    ):
        super().__init__()
        self.drop_length_low = drop_length_low
        self.drop_length_high = drop_length_high
        self.drop_count_low = drop_count_low
        self.drop_count_high = drop_count_high
        self.p = p

    def forward(self, waveforms: torch.Tensor, sample_rate: int | None = None) -> torch.Tensor:
        """Apply random chunk dropping.

        Args:
            waveforms: (batch, channels, time) tensor
            sample_rate: unused, for API compatibility
        """
        if torch.rand(1).item() > self.p:
            return waveforms

        batch_size, channels, length = waveforms.shape
        output = waveforms.clone()

        for i in range(batch_size):
            n_drops = torch.randint(self.drop_count_low, self.drop_count_high + 1, (1,)).item()
            for _ in range(n_drops):
                drop_len = torch.randint(self.drop_length_low, min(self.drop_length_high + 1, length), (1,)).item()
                start = torch.randint(0, max(1, length - drop_len), (1,)).item()
                output[i, :, start : start + drop_len] = 0.0

        return output


class DropFreq(nn.Module):
    """Drop random frequency bands (GPU-native, matches SpeechBrain's DropFreq)."""

    def __init__(
        self,
        drop_freq_low: float = 0.0,
        drop_freq_high: float = 1.0,
        drop_freq_count_low: int = 1,
        drop_freq_count_high: int = 3,
        drop_freq_width: float = 0.05,
        p: float = 0.5,
    ):
        super().__init__()
        self.drop_freq_low = drop_freq_low
        self.drop_freq_high = drop_freq_high
        self.drop_freq_count_low = drop_freq_count_low
        self.drop_freq_count_high = drop_freq_count_high
        self.drop_freq_width = drop_freq_width
        self.p = p

    def forward(self, waveforms: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        """Apply random frequency band dropping via notch filters.

        Args:
            waveforms: (batch, channels, time) tensor
            sample_rate: sample rate in Hz
        """
        if torch.rand(1).item() > self.p:
            return waveforms

        batch_size = waveforms.shape[0]
        output = waveforms.clone()

        for i in range(batch_size):
            n_drops = torch.randint(self.drop_freq_count_low, self.drop_freq_count_high + 1, (1,)).item()
            for _ in range(n_drops):
                # Random center frequency (normalized 0-1)
                center = torch.empty(1).uniform_(self.drop_freq_low, self.drop_freq_high).item()
                low = max(0.0, center - self.drop_freq_width / 2)
                high = min(1.0, center + self.drop_freq_width / 2)

                # Apply notch filter in frequency domain
                output[i] = self._apply_notch(output[i], low, high)

        return output

    def _apply_notch(self, waveform: torch.Tensor, low: float, high: float) -> torch.Tensor:
        """Zero out frequency band [low, high] (normalized 0-1 = 0-Nyquist)."""
        # FFT
        spec = torch.fft.rfft(waveform, dim=-1)
        n_freqs = spec.shape[-1]

        # Create mask
        low_bin = int(low * n_freqs)
        high_bin = int(high * n_freqs)
        mask = torch.ones(n_freqs, device=waveform.device)
        mask[low_bin:high_bin] = 0.0

        # Apply and inverse FFT
        spec = spec * mask
        return torch.fft.irfft(spec, n=waveform.shape[-1], dim=-1)


def load_paths_from_speechbrain_csv(csv_file: str | Path) -> list[str]:
    """Parse SpeechBrain's annotation CSV and return list of audio file paths."""
    csv_file = Path(csv_file)
    if not csv_file.exists():
        raise FileNotFoundError(f"Annotation CSV not found: {csv_file}")

    paths = []
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # SpeechBrain CSVs typically have 'wav' or 'filepath' column
            if "wav" in row:
                paths.append(row["wav"])
            elif "filepath" in row:
                paths.append(row["filepath"])
            else:
                # Try first column that looks like a path
                for v in row.values():
                    if v.endswith(".wav") or v.endswith(".flac"):
                        paths.append(v)
                        break

    if not paths:
        raise ValueError(f"No audio paths found in {csv_file}")

    return paths


class GPUAugmenter(nn.Module):
    """Drop-in replacement for SpeechBrain Augmenter with GPU-native operations.

    Matches SpeechBrain's augmentation pipeline:
    - AddNoise: Add background noise at random SNR
    - AddReverb: Convolve with room impulse responses
    - DropFreq: Zero out random frequency bands
    - DropChunk: Zero out random time chunks

    Args:
        sample_rate: Audio sample rate
        noise_csv: Path to SpeechBrain noise annotation CSV (or null to disable)
        rir_csv: Path to SpeechBrain RIR annotation CSV (or null to disable)
        enable_noise: Explicitly enable/disable noise augmentation
        enable_reverb: Explicitly enable/disable reverb augmentation
        enable_drop_freq: Enable/disable frequency dropout
        enable_drop_chunk: Enable/disable chunk dropout
        snr_low: Minimum SNR in dB for noise addition
        snr_high: Maximum SNR in dB for noise addition
        noise_prob: Probability of applying noise (per sample)
        reverb_prob: Probability of applying reverb (per sample)
        drop_freq_prob: Probability of applying DropFreq (per sample)
        drop_chunk_prob: Probability of applying DropChunk (per sample)
        concat_original: If True, concatenate original with augmented (doubles batch)
        augment_prob: Probability of applying augmentation pipeline
        drop_freq_low: Minimum normalized frequency for DropFreq
        drop_freq_high: Maximum normalized frequency for DropFreq
        drop_freq_count_low: Minimum number of frequency bands to drop
        drop_freq_count_high: Maximum number of frequency bands to drop
        drop_freq_width: Width of each dropped frequency band
        drop_length_low: Minimum samples to drop in DropChunk
        drop_length_high: Maximum samples to drop in DropChunk
        drop_count_low: Minimum number of chunks to drop
        drop_count_high: Maximum number of chunks to drop
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        noise_csv: str | None = None,
        rir_csv: str | None = None,
        # Enable/disable individual augmentations
        enable_noise: bool = True,
        enable_reverb: bool = True,
        enable_drop_freq: bool = True,
        enable_drop_chunk: bool = True,
        # Per-augmentation probabilities
        noise_prob: float = 0.5,
        reverb_prob: float = 0.5,
        drop_freq_prob: float = 0.5,
        drop_chunk_prob: float = 0.5,
        # Noise params
        snr_low: float = 0.0,
        snr_high: float = 15.0,
        concat_original: bool = True,
        augment_prob: float = 1.0,
        # DropFreq params (matching SpeechBrain defaults)
        drop_freq_low: float = 0.0,
        drop_freq_high: float = 1.0,
        drop_freq_count_low: int = 1,
        drop_freq_count_high: int = 3,
        drop_freq_width: float = 0.05,
        # DropChunk params (matching SpeechBrain defaults)
        drop_length_low: int = 1000,
        drop_length_high: int = 2000,
        drop_count_low: int = 1,
        drop_count_high: int = 5,
    ):
        super().__init__()
        self.concat_original = concat_original
        self.sample_rate = sample_rate
        self.augment_prob = augment_prob

        transforms = []

        # AddNoise - load from SpeechBrain CSV
        if enable_noise and noise_csv is not None:
            noise_paths = load_paths_from_speechbrain_csv(noise_csv)
            transforms.append(
                AddBackgroundNoise(
                    background_paths=noise_paths,
                    min_snr_in_db=snr_low,
                    max_snr_in_db=snr_high,
                    p=noise_prob,
                )
            )

        # AddReverb - load from SpeechBrain CSV
        if enable_reverb and rir_csv is not None:
            rir_paths = load_paths_from_speechbrain_csv(rir_csv)
            transforms.append(
                ApplyImpulseResponse(
                    ir_paths=rir_paths,
                    p=reverb_prob,
                )
            )

        # DropFreq
        if enable_drop_freq:
            transforms.append(
                DropFreq(
                    drop_freq_low=drop_freq_low,
                    drop_freq_high=drop_freq_high,
                    drop_freq_count_low=drop_freq_count_low,
                    drop_freq_count_high=drop_freq_count_high,
                    drop_freq_width=drop_freq_width,
                    p=drop_freq_prob,
                )
            )

        # DropChunk
        if enable_drop_chunk:
            transforms.append(
                DropChunk(
                    drop_length_low=drop_length_low,
                    drop_length_high=drop_length_high,
                    drop_count_low=drop_count_low,
                    drop_count_high=drop_count_high,
                    p=drop_chunk_prob,
                )
            )

        self.augment = Compose(transforms, p=augment_prob) if transforms else None

    def forward(
        self, waveforms: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply augmentation pipeline matching SpeechBrain's Augmenter interface.

        Args:
            waveforms: (batch, time) or (batch, 1, time) audio tensor
            lengths: Normalized lengths (0-1) for each sample

        Returns:
            Tuple of (augmented_waveforms, lengths). If concat_original=True,
            both are doubled in batch dimension.
        """
        # Ensure 3D: (batch, channels, time)
        needs_squeeze = False
        if waveforms.ndim == 2:
            waveforms = waveforms.unsqueeze(1)
            needs_squeeze = True

        # Store original for potential concatenation
        original = waveforms

        # Apply augmentation pipeline (if any augmentations enabled)
        if self.augment is not None:
            augmented = self.augment(waveforms, sample_rate=self.sample_rate)
        else:
            augmented = waveforms

        if needs_squeeze:
            augmented = augmented.squeeze(1)
            original = original.squeeze(1)

        if self.concat_original:
            augmented = torch.cat([original, augmented], dim=0)
            lengths = torch.cat([lengths, lengths], dim=0)

        return augmented, lengths

    def replicate_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Double labels when concat_original=True (matches SpeechBrain interface)."""
        if self.concat_original:
            return torch.cat([labels, labels], dim=0)
        return labels
