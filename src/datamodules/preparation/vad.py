"""Optional voice activity detection (VAD) utilities used by dataset preparation scripts.

The VAD is applied at the metadata/CSV level:
- We keep the original audio file path.
- We represent trimming/splitting through `vad_start`/`vad_end` (seconds) and `vad_chunk_id`.

This module is designed for Hydra dependency injection:
configs set `dataset.vad._target_` and the prep scripts instantiate it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch


SpeechSegment = Tuple[float, float]


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _postprocess_speech_segments(
    segments: Sequence[SpeechSegment],
    *,
    duration_s: float,
    merge_gap_s: float,
    drop_short_speech_s: float,
) -> List[SpeechSegment]:
    """Fix common VAD glitches: merge tiny gaps, drop tiny speech bursts."""
    if not segments:
        return []

    # Normalize and clamp
    cleaned: List[SpeechSegment] = []
    for s, e in segments:
        s = _clamp(float(s), 0.0, duration_s)
        e = _clamp(float(e), 0.0, duration_s)
        if e > s:
            cleaned.append((s, e))
    cleaned.sort(key=lambda x: x[0])
    if not cleaned:
        return []

    # Merge overlaps / short gaps
    merged: List[SpeechSegment] = [cleaned[0]]
    for s, e in cleaned[1:]:
        ps, pe = merged[-1]
        if s <= pe + merge_gap_s:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))

    # Drop unreasonable short speech
    pruned = [(s, e) for (s, e) in merged if (e - s) >= drop_short_speech_s]
    return pruned


def _gap_threshold_s(duration_s: float, ratio: float, min_s: float, max_s: float) -> float:
    return _clamp(duration_s * ratio, min_s, max_s)


def _group_by_long_silence(speech: Sequence[SpeechSegment], gap_threshold_s: float) -> List[List[SpeechSegment]]:
    if not speech:
        return []
    groups: List[List[SpeechSegment]] = []
    cur: List[SpeechSegment] = [speech[0]]
    for (s0, e0), (s1, e1) in zip(speech, speech[1:]):
        if (s1 - e0) > gap_threshold_s:
            groups.append(cur)
            cur = [(s1, e1)]
        else:
            cur.append((s1, e1))
    groups.append(cur)
    return groups


class SileroCsvVad:
    """File-level Silero VAD that rewrites metadata rows.

    Intended usage:
    - instantiate via Hydra from configs
    - call `apply(rows, audio_root, split_name)` before segmentation
    """

    def __init__(
        self,
        enabled: bool = False,
        apply_to_splits: Optional[Sequence[str]] = ("train", "val"),
        sample_rate: int = 16000,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
        # Post-processing knobs
        merge_gap_s: float = 0.10,
        drop_short_speech_s: float = 0.20,
        # Split on long internal silences (dynamic threshold)
        long_silence_ratio: float = 0.20,
        long_silence_min_s: float = 0.60,
        long_silence_max_s: float = 2.00,
        # Segment-level filtering (used by segment_utterance)
        segment_max_silence_ratio: float = 0.60,
    ) -> None:
        self.enabled = bool(enabled)
        self.apply_to_splits = tuple(str(s) for s in (apply_to_splits or ()))
        self.sample_rate = int(sample_rate)
        self.min_speech_duration_ms = int(min_speech_duration_ms)
        self.min_silence_duration_ms = int(min_silence_duration_ms)
        self.speech_pad_ms = int(speech_pad_ms)

        self.merge_gap_s = float(merge_gap_s)
        self.drop_short_speech_s = float(drop_short_speech_s)

        self.long_silence_ratio = float(long_silence_ratio)
        self.long_silence_min_s = float(long_silence_min_s)
        self.long_silence_max_s = float(long_silence_max_s)

        self.segment_max_silence_ratio = float(segment_max_silence_ratio)

        self._model = None
        self._utils = None
        self._cache: Dict[str, List[SpeechSegment]] = {}

    def should_apply(self, split_name: str) -> bool:
        if not self.enabled:
            return False
        if not self.apply_to_splits:
            return True
        s = (split_name or "").lower()
        for token in self.apply_to_splits:
            t = str(token).lower()
            if t == s:
                return True
            if t in {"train", "val", "valid", "dev"} and t in s:
                return True
        return False

    def _load(self) -> Tuple[torch.nn.Module, Any]:
        if self._model is not None and self._utils is not None:
            return self._model, self._utils

        try:
            self._model, self._utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
            )
        except TypeError:
            self._model, self._utils = torch.hub.load(
                "snakers4/silero-vad",
                "silero_vad",
                force_reload=False,
            )

        return self._model, self._utils

    def speech_timestamps(self, audio_path: Path, duration_s: float) -> List[SpeechSegment]:
        key = str(audio_path)
        if key in self._cache:
            return self._cache[key]

        model, utils = self._load()
        try:
            (get_speech_timestamps, _, read_audio, _, _) = utils
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Unexpected Silero-VAD utils layout") from e

        wav = read_audio(str(audio_path), sampling_rate=self.sample_rate)
        with torch.no_grad():
            speech = get_speech_timestamps(
                wav,
                model,
                sampling_rate=self.sample_rate,
                min_speech_duration_ms=self.min_speech_duration_ms,
                min_silence_duration_ms=self.min_silence_duration_ms,
                speech_pad_ms=self.speech_pad_ms,
            )

        raw: List[SpeechSegment] = [
            (float(seg["start"]) / float(self.sample_rate), float(seg["end"]) / float(self.sample_rate))
            for seg in speech
            if float(seg["end"]) > float(seg["start"])
        ]

        fixed = _postprocess_speech_segments(
            raw,
            duration_s=float(duration_s),
            merge_gap_s=self.merge_gap_s,
            drop_short_speech_s=self.drop_short_speech_s,
        )

        self._cache[key] = fixed
        return fixed

    def apply(
        self,
        rows: Iterable[Dict[str, Any]],
        *,
        audio_root: Path,
        split_name: str,
        rel_filepath_key: str = "rel_filepath",
        recording_duration_key: str = "recording_duration",
    ) -> List[Dict[str, Any]]:
        if not self.should_apply(split_name):
            return list(rows)

        out: List[Dict[str, Any]] = []
        for row in rows:
            rel = str(row[rel_filepath_key])
            audio_path = (audio_root / rel).resolve()
            duration_s = float(row[recording_duration_key])

            speech = self.speech_timestamps(audio_path, duration_s)
            if not speech:
                continue

            long_gap = _gap_threshold_s(duration_s, self.long_silence_ratio, self.long_silence_min_s, self.long_silence_max_s)
            groups = _group_by_long_silence(speech, long_gap)

            for idx, group in enumerate(groups):
                vad_start = group[0][0]
                vad_end = group[-1][1]
                if vad_end <= vad_start:
                    continue

                new_row = dict(row)
                new_row["vad_start"] = round(vad_start, 3)
                new_row["vad_end"] = round(vad_end, 3)
                new_row["vad_chunk_id"] = f"{rel}::vad{idx:02d}"
                new_row[recording_duration_key] = round(vad_end - vad_start, 3)

                # Store absolute (file) timestamps for segment-level filtering.
                new_row["vad_speech_timestamps"] = json.dumps(
                    [[round(s, 3), round(e, 3)] for (s, e) in group],
                    separators=(",", ":"),
                )
                out.append(new_row)

        return out
