"""GPU-native audio augmentation.

Audio augmentation that runs entirely on-device during the training forward
pass.  Noise and RIR audio are preloaded into in-memory tensor banks once at
construction; mixing and convolution then run natively in torch with no
per-batch disk I/O (the previous torch-audiomentations path reloaded and
CPU-resampled files every batch, which dominated training time).

Key design decisions:
- Preload noise/RIR into flat in-memory buffers; sample and mix on-device.
- Assumes mono audio (single channel) for simplicity and performance.
- Supports RIR amplitude normalization without amplifying noise levels.
- Applies each transform to a per-example random subset (independent samples).
"""

import csv
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchaudio.functional import fftconvolve

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


def load_audio_mono_resampled(
    path: str,
    target_sr: int,
    resampler_cache: dict[int, torchaudio.transforms.Resample],
) -> torch.Tensor:
    """Load an audio file as a mono, target-SR, 1-D float32 CPU tensor.

    Resamples only when the source rate differs from ``target_sr``, reusing one
    ``Resample`` per source rate via ``resampler_cache`` (the noise/RIR corpora
    are single-rate in practice, so at most one resampler is built).

    Args:
        path: Audio file path.
        target_sr: Desired sample rate.
        resampler_cache: Mutable dict mapping source SR -> Resample module.
    """
    assert target_sr > 0, f"target_sr must be positive, got {target_sr}"
    wav, sr = torchaudio.load(path)
    assert wav.ndim == 2, f"Expected (channels, time) from {path}, got {wav.shape}"

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != target_sr:
        if sr not in resampler_cache:
            resampler_cache[sr] = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=target_sr
            )
        wav = resampler_cache[sr](wav)

    wav = wav.squeeze(0).float()
    assert wav.ndim == 1, f"Expected 1-D mono audio, got {wav.shape}"
    return wav


def build_rir_bank(
    ir_paths: list[str],
    target_sr: int,
    max_files: int | None = None,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Preload RIRs into one flat buffer with per-RIR offsets.

    Concatenating into a single 1-D buffer keeps memory at ~sum-of-lengths
    instead of ``num_rirs * max_len`` (important with tens of thousands of
    variable-length RIRs).  RIR ``i`` is ``flat[offsets[i]:offsets[i + 1]]``.

    Args:
        ir_paths: RIR file paths.
        target_sr: Sample rate to resample to.
        max_files: Optional cap; a seeded random subset is kept when the corpus
            is larger.  ``None`` keeps every RIR.
        seed: RNG seed for the subset selection.

    Returns:
        Tuple of (flat float32 buffer [total], int64 offsets [N + 1]).
    """
    if len(ir_paths) < 1:
        raise ValueError("build_rir_bank received an empty RIR path list")

    if max_files is not None and len(ir_paths) > max_files:
        ir_paths = random.Random(seed).sample(ir_paths, max_files)

    resampler_cache: dict[int, torchaudio.transforms.Resample] = {}
    pieces = [
        load_audio_mono_resampled(p, target_sr, resampler_cache) for p in ir_paths
    ]

    lengths = torch.tensor([p.numel() for p in pieces], dtype=torch.long)
    offsets = torch.zeros(len(pieces) + 1, dtype=torch.long)
    offsets[1:] = torch.cumsum(lengths, dim=0)
    flat = torch.cat(pieces).float()

    assert flat.numel() >= 1, "RIR bank is empty"
    assert offsets.numel() == len(pieces) + 1
    return flat, offsets


def build_noise_bank(
    noise_paths: list[str],
    target_sr: int,
    max_total_seconds: float = 1200.0,
) -> torch.Tensor:
    """Preload noise into one flat RMS-normalized buffer for offset sampling.

    Each file is RMS-normalized before concatenation (matching the per-piece
    normalization in the replaced torch-audiomentations path), so any
    random-offset slice has RMS ~= 1.  Capped at ``max_total_seconds``.

    Args:
        noise_paths: Noise file paths.
        target_sr: Sample rate to resample to.
        max_total_seconds: Maximum total noise duration to retain.
    """
    if len(noise_paths) < 1:
        raise ValueError("build_noise_bank received an empty noise path list")

    target_len = int(max_total_seconds * target_sr)
    resampler_cache: dict[int, torchaudio.transforms.Resample] = {}

    pieces = []
    total = 0
    for p in noise_paths:
        wav = load_audio_mono_resampled(p, target_sr, resampler_cache)
        wav = wav / (wav.square().mean().sqrt() + 1e-8)
        pieces.append(wav)
        total += wav.numel()
        if total >= target_len:
            break

    flat = torch.cat(pieces).float()[:target_len]
    assert flat.ndim == 1 and flat.numel() >= 1, "Noise bank is empty"
    return flat


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

        # One resampler per speed factor, registered so they move to the
        # module's device once. Unity uses a no-op placeholder to keep the
        # list index aligned with speed_factors (unity is skipped in forward).
        self._resamplers = nn.ModuleList()
        for speed in self.speed_factors:
            if speed == 1.0:
                self._resamplers.append(nn.Identity())
            else:
                self._resamplers.append(
                    torchaudio.transforms.Resample(
                        orig_freq=int(sample_rate * speed),
                        new_freq=sample_rate,
                    )
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
            resampler = self._resamplers[speed_idx]
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
    """Reverb via RIR convolution with peak amplitude normalization.

    Preloads every RIR into a flat in-memory buffer once (no per-batch disk
    I/O) and convolves with ``fftconvolve``. Output length matches the input
    (the ``full`` convolution is truncated to the signal length). When
    ``normalize=True`` the output peak is rescaled to the input peak so reverb
    does not boost energy.

    The RIR bank is kept on CPU (the full corpus is multi-GB — ~4.7 GB for the
    60k-RIR set) and only the few RIRs sampled for the current batch are copied
    to the GPU. That copy is a few MB per step, negligible next to the
    per-batch disk reads it replaces, and keeps GPU memory free for the model.

    Args:
        ir_csv: Path to CSV with RIR paths
        sample_rate: Audio sample rate
        p: Per-example probability of applying reverb
        normalize: Whether to match the input peak amplitude after convolution
        max_files: Optional cap on preloaded RIRs (None = all). Larger keeps
            marginally more diversity at linearly more memory.
        seed: RNG seed for the RIR subset when capped
    """

    def __init__(
        self,
        ir_csv: str,
        sample_rate: int,
        p: float = 0.5,
        normalize: bool = True,
        max_files: int | None = None,
        seed: int = 0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.p = p

        ir_paths = load_paths_from_csv(ir_csv)
        # Plain attributes (not buffers): the bank stays on CPU and must not be
        # moved to the GPU with the module nor saved into checkpoints.
        self._rir_flat, self._rir_offsets = build_rir_bank(
            ir_paths, sample_rate, max_files=max_files, seed=seed
        )

    def forward(
        self, waveforms: torch.Tensor, sample_rate: int | None = None
    ) -> torch.Tensor:
        """Apply reverb to a per-example random subset."""
        _check_mono(waveforms, "NormalizedReverb: ")
        batch_size, channels, length = waveforms.shape
        assert channels == 1, "NormalizedReverb only supports mono audio."
        assert waveforms.dtype == torch.float32
        device = waveforms.device

        apply_mask = torch.rand(batch_size, device=device) <= self.p
        if not apply_mask.any():
            return waveforms

        # Sample RIRs on CPU (where the bank lives), pad, then ship just this
        # batch's RIRs to the device.
        num_rirs = self._rir_offsets.numel() - 1
        idx = torch.randint(num_rirs, (batch_size,))
        starts = self._rir_offsets[idx].tolist()
        ends = self._rir_offsets[idx + 1].tolist()
        selected = [self._rir_flat[s:e] for s, e in zip(starts, ends)]
        ir = pad_sequence(selected, batch_first=True).unsqueeze(1)
        ir = ir.to(device, non_blocking=True)

        conv = fftconvolve(waveforms, ir, mode="full")[..., :length]
        assert conv.shape[-1] == length

        if self.normalize:
            orig_peaks = waveforms.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
            new_peaks = conv.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
            conv = conv * (orig_peaks / new_peaks)

        return torch.where(apply_mask.view(batch_size, 1, 1), conv, waveforms)


class NoiseFromCSV(nn.Module):
    """Background-noise augmentation from a preloaded in-memory noise bank.

    Preloads noise into one flat RMS-normalized buffer once (no per-batch disk
    I/O) and mixes on-device. For each example a random contiguous slice (with
    wrap-around tiling) is scaled to a per-example SNR drawn uniformly in
    ``[min_snr_db, max_snr_db]`` and added to the signal.

    Args:
        noise_csv: Path to CSV with noise file paths
        min_snr_db: Minimum SNR in dB
        max_snr_db: Maximum SNR in dB
        p: Per-example probability of adding noise
        sample_rate: Audio sample rate (the bank is built at this rate)
        max_total_seconds: Cap on preloaded noise duration
    """

    def __init__(
        self,
        noise_csv: str,
        min_snr_db: float = 0.0,
        max_snr_db: float = 15.0,
        p: float = 0.5,
        sample_rate: int = 16000,
        max_total_seconds: float = 1200.0,
    ):
        super().__init__()
        if min_snr_db > max_snr_db:
            raise ValueError(
                f"min_snr_db ({min_snr_db}) must not exceed "
                f"max_snr_db ({max_snr_db})"
            )
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db
        self.p = p
        self.sample_rate = sample_rate

        noise_paths = load_paths_from_csv(noise_csv)
        bank = build_noise_bank(noise_paths, sample_rate, max_total_seconds)
        self.register_buffer("_noise_bank", bank, persistent=False)

    def forward(
        self, waveforms: torch.Tensor, sample_rate: int | None = None
    ) -> torch.Tensor:
        """Add background noise to a per-example random subset."""
        _check_mono(waveforms, "NoiseFromCSV: ")
        batch_size, channels, length = waveforms.shape
        assert channels == 1, "NoiseFromCSV only supports mono audio."
        assert waveforms.dtype == torch.float32
        assert self._noise_bank.device == waveforms.device
        assert (sample_rate or self.sample_rate) == self.sample_rate, (
            f"NoiseFromCSV bank built at {self.sample_rate} Hz, " f"got {sample_rate}"
        )
        device = waveforms.device

        # Modulo wrap-around below tiles the bank, so a bank shorter than the
        # signal is fine (it simply repeats).
        bank_len = self._noise_bank.numel()

        apply_mask = torch.rand(batch_size, device=device) <= self.p
        if not apply_mask.any():
            return waveforms

        # Random contiguous slice per example; modulo wrap-around tiles noise.
        offsets = torch.randint(bank_len, (batch_size, 1), device=device)
        ar = torch.arange(length, device=device).view(1, length)
        gather = (offsets + ar) % bank_len
        noise = self._noise_bank[gather].unsqueeze(1)

        noise_rms = noise.square().mean(-1, keepdim=True).sqrt()
        noise = noise / (noise_rms + 1e-8)

        snr = torch.empty(batch_size, device=device).uniform_(
            self.min_snr_db, self.max_snr_db
        )
        sig_rms = waveforms.square().mean(-1, keepdim=True).sqrt()
        scale = sig_rms / (10 ** (snr.view(batch_size, 1, 1) / 20))
        noised = waveforms + scale * noise

        return torch.where(apply_mask.view(batch_size, 1, 1), noised, waveforms)


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


class CodecAugmentation(nn.Module):
    """Random codec augmentation to simulate transmission channel effects.

    Wraps SpeechBrain's CodecAugment which applies random audio codecs
    (mp3, ogg/vorbis, etc.) via torchaudio+ffmpeg.

    Note: inherently CPU-bound — the per-sample ffmpeg round-trip dominates
    this transform. Keep ``p`` low; consider moving it to the dataloader
    workers if it becomes a bottleneck.

    Args:
        sample_rate: Audio sample rate
        p: Probability of applying codec augmentation per sample
    """

    def __init__(self, sample_rate: int = 16000, p: float = 0.3):
        super().__init__()
        self.p = p
        self.sample_rate = sample_rate

        from speechbrain.augment.codec import CodecAugment

        self._codec = CodecAugment(sample_rate=sample_rate)

    def forward(
        self, waveforms: torch.Tensor, sample_rate: int | None = None
    ) -> torch.Tensor:
        """Apply random codec to a subset of samples.

        Args:
            waveforms: (batch, 1, time) mono audio tensor
            sample_rate: Unused, uses self.sample_rate

        Returns:
            Codec-augmented waveforms (same shape)
        """
        batch_size = waveforms.shape[0]
        device = waveforms.device
        apply_mask = torch.rand(batch_size) <= self.p

        if not apply_mask.any():
            return waveforms

        output = waveforms.clone()
        indices = apply_mask.nonzero(as_tuple=True)[0]
        subset = waveforms[indices].cpu()  # move the selected rows once
        orig_len = waveforms.shape[-1]

        for i, idx in enumerate(indices):
            # CodecAugment expects (batch, time) on CPU
            sample = subset[i].squeeze(0).unsqueeze(0)
            coded = self._codec(sample)
            # Handle length changes from codec round-trip
            if coded.shape[-1] < orig_len:
                coded = F.pad(coded, (0, orig_len - coded.shape[-1]))
            elif coded.shape[-1] > orig_len:
                coded = coded[..., :orig_len]
            output[idx] = (
                coded.unsqueeze(1).to(device)
                if waveforms.ndim == 3
                else coded.to(device)
            )

        return output


class FreqMask(nn.Module):
    """Frequency masking augmentation for codec robustness.

    Applies STFT, zeros out frequency bins above a randomly selected cutoff,
    and converts back via ISTFT. Simulates bandwidth-limited channels and
    helps models generalize to codec-processed audio.

    Reference: "Temporal Variability and Multi-Viewed Self-Supervised
    Representations to Tackle the ASVspoof5 Deepfake Challenge" (2024).

    Args:
        sample_rate: Audio sample rate
        cutoff_freqs: List of possible cutoff frequencies in Hz
        p: Probability of applying the mask per sample
        n_fft: FFT size for STFT
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        cutoff_freqs: list[int] | None = None,
        p: float = 0.3,
        n_fft: int = 512,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.cutoff_freqs = cutoff_freqs or [4000, 5000, 6000, 7000]
        self.p = p
        self.n_fft = n_fft
        self.hop_length = n_fft // 4

        # Pre-compute the bin index for each cutoff frequency
        freq_resolution = sample_rate / n_fft
        cutoff_bins = torch.tensor(
            [int(f / freq_resolution) for f in self.cutoff_freqs],
            dtype=torch.long,
        )
        self.register_buffer("_cutoff_bins", cutoff_bins, persistent=False)

    def forward(
        self, waveforms: torch.Tensor, sample_rate: int | None = None
    ) -> torch.Tensor:
        """Apply frequency masking to a subset of samples.

        Args:
            waveforms: (batch, 1, time) mono audio tensor
            sample_rate: Unused, uses self.sample_rate

        Returns:
            Frequency-masked waveforms (same shape)
        """
        batch_size = waveforms.shape[0]
        device = waveforms.device
        apply_mask = torch.rand(batch_size, device=device) <= self.p

        if not apply_mask.any():
            return waveforms

        output = waveforms.clone()
        indices = apply_mask.nonzero(as_tuple=True)[0]
        subset = waveforms[indices]

        # Handle (batch, 1, time) -> (batch, time) for STFT
        was_3d = subset.ndim == 3
        if was_3d:
            subset = subset.squeeze(1)

        orig_len = subset.shape[-1]

        # STFT
        window = torch.hann_window(self.n_fft, device=device)
        spec = torch.stft(
            subset,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
        )

        # Randomly select a cutoff bin per sample, then zero bins above it with
        # a broadcast mask (vectorized over samples and frames).
        cutoff_indices = torch.randint(
            self._cutoff_bins.numel(), (len(indices),), device=device
        )
        cutoff_bin = self._cutoff_bins[cutoff_indices]  # (n,)
        num_freqs = spec.shape[1]
        assert cutoff_bin.max() <= num_freqs
        freq_ids = torch.arange(num_freqs, device=device).view(1, num_freqs, 1)
        keep = (freq_ids < cutoff_bin.view(-1, 1, 1)).to(spec.real.dtype)
        spec = spec * keep

        # ISTFT
        reconstructed = torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            length=orig_len,
        )

        if was_3d:
            reconstructed = reconstructed.unsqueeze(1)

        output[indices] = reconstructed

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


if __name__ == "__main__":
    import tempfile

    sr = 16000
    tmp = Path(tempfile.mkdtemp())
    for name, n in [("noise", 8000), ("rir", 400)]:
        torchaudio.save(str(tmp / f"{name}.wav"), torch.randn(1, n), sr)
        (tmp / f"{name}.csv").write_text(f"wav\n{tmp / f'{name}.wav'}\n")

    x = torch.randn(2, 1, sr)
    rev = NormalizedReverb(str(tmp / "rir.csv"), sr, p=1.0)
    noi = NoiseFromCSV(str(tmp / "noise.csv"), 10, 10, p=1.0, sample_rate=sr)
    spd = SpeedPerturb(sr, [0.9, 1.0, 1.1], False, 10)
    fmk = FreqMask(sr, p=1.0)
    out = noi(x)
    snr = 20 * torch.log10(x.square().mean().sqrt() / (out - x).square().mean().sqrt())
    print("reverb:", rev(x).shape, "noise:", out.shape, f"snr={snr:.1f}dB")
    print("speed:", spd(x).shape, "freqmask:", fmk(x).shape)
