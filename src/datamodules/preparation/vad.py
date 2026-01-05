"""Voice activity detection (VAD) utilities with multiprocessing support.

The VAD is applied at the metadata/CSV level:
- Original audio file path is preserved
- Trimming/splitting represented through `vad_start`/`vad_end` (seconds) and `vad_chunk_id`
"""

from __future__ import annotations

import json
import multiprocessing as mp
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from tqdm.auto import tqdm

from src import utils

log = utils.get_pylogger(__name__)

SpeechSegment = Tuple[float, float]

# Worker globals (initialized once per worker)
_worker_model = None
_worker_get_speech_timestamps = None
_worker_read_audio = None
_worker_sample_rate = None


def _init_worker(sample_rate: int):
    """Initialize Silero VAD model once per worker process.
    
    Args:
        sample_rate: Audio sample rate (stored globally for workers)
        
    Raises:
        RuntimeError: If model loading or initialization fails
    """
    global _worker_model, _worker_get_speech_timestamps, _worker_read_audio, _worker_sample_rate
    
    # Limit PyTorch threads to 1 per worker to avoid CPU oversubscription
    torch.set_num_threads(1)
    
    # Load Silero VAD model
    model, vad_utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        verbose=False,
    )
            
    # Move to CPU and set to eval mode
    model.to(torch.device("cpu"))
    model.eval()
    
    # Robust unpacking of Silero utils
    get_speech_timestamps, _, read_audio, *_ = vad_utils
    
    if get_speech_timestamps is None or read_audio is None:
        raise RuntimeError("Failed to extract Silero VAD utility functions")
    
    # Set globals for worker process
    _worker_model = model
    _worker_get_speech_timestamps = get_speech_timestamps
    _worker_read_audio = read_audio
    _worker_sample_rate = sample_rate

def _postprocess_segments(
    segments: Sequence[SpeechSegment],
    duration_s: float,
    merge_gap_s: float,
    drop_short_s: float,
) -> List[SpeechSegment]:
    """Merge nearby segments and drop short speech bursts."""
    if not segments:
        return []
    
    # Clamp to valid range and filter invalid
    cleaned = [(max(0.0, min(s, duration_s)), max(0.0, min(e, duration_s))) for s, e in segments if e > s]
    if not cleaned:
        return []
    
    cleaned.sort()
    
    # Merge segments within merge_gap_s
    merged = [cleaned[0]]
    for s, e in cleaned[1:]:
        ps, pe = merged[-1]
        if s <= pe + merge_gap_s:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    
    # Drop short segments
    return [(s, e) for s, e in merged if (e - s) >= drop_short_s]


def _group_by_silence(segments: Sequence[SpeechSegment], gap_threshold_s: float) -> List[List[SpeechSegment]]:
    """Split segments into groups separated by long silences."""
    if not segments:
        return []
    
    groups = []
    current = [segments[0]]
    prev_end = segments[0][1]
    
    for s, e in segments[1:]:
        if s - prev_end > gap_threshold_s:
            groups.append(current)
            current = [(s, e)]
        else:
            current.append((s, e))
        prev_end = e
    
    groups.append(current)
    return groups


def _process_file(
    row: Dict[str, Any],
    audio_root: Path,
    rel_key: str,
    dur_key: str,
    merge_gap_s: float,
    drop_short_s: float,
    silence_ratio: float,
    silence_min_s: float,
    silence_max_s: float,
    min_speech_ms: int,
    min_silence_ms: int,
    speech_pad_ms: int,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Process a single audio file with VAD."""
    global _worker_model, _worker_get_speech_timestamps, _worker_read_audio, _worker_sample_rate
    
    # Sanity check: ensure worker was initialized correctly
    if _worker_model is None or _worker_get_speech_timestamps is None or _worker_read_audio is None or _worker_sample_rate is None:
        return [], {"rel_filepath": "unknown", "speaker_id": "unknown", "duration_s": 0.0, "error": "Worker not initialized correctly"}
    
    # Validate required keys exist
    try:
        rel = row[rel_key]
        duration_s = float(row[dur_key])
        speaker_id = str(row.get("speaker_id", "unknown"))
    except (KeyError, ValueError, TypeError) as e:
        return [], {"rel_filepath": str(row.get(rel_key, "unknown")), "speaker_id": "unknown", "duration_s": 0.0, "error": f"Invalid row data: {e}"}
    
    audio_path = audio_root / rel
    
    # Check file exists
    if not audio_path.exists():
        return [], {"rel_filepath": rel, "speaker_id": speaker_id, "duration_s": duration_s, "error": "File not found"}
    
    try:
        wav = _worker_read_audio(str(audio_path), sampling_rate=_worker_sample_rate)
        
        with torch.no_grad():
            speech = _worker_get_speech_timestamps(
                wav,
                _worker_model,
                sampling_rate=_worker_sample_rate,
                min_speech_duration_ms=min_speech_ms,
                min_silence_duration_ms=min_silence_ms,
                speech_pad_ms=speech_pad_ms,
            )
        
        # Convert to seconds (pre-compute inverse for efficiency)
        sr_inv = 1.0 / _worker_sample_rate
        raw = [(seg["start"] * sr_inv, seg["end"] * sr_inv) for seg in speech if seg["end"] > seg["start"]]
        
        # Post-process and split on long silences
        fixed = _postprocess_segments(raw, duration_s, merge_gap_s, drop_short_s)
        if not fixed:
            return [], {"rel_filepath": rel, "speaker_id": speaker_id, "duration_s": duration_s}
        
        gap_threshold = max(silence_min_s, min(duration_s * silence_ratio, silence_max_s))
        groups = _group_by_silence(fixed, gap_threshold)
        
        # Create output rows
        out_rows = []
        for idx, group in enumerate(groups):
            start, end = group[0][0], group[-1][1]
            if end <= start:
                continue
            
            new_row = row.copy()
            new_row.update({
                "vad_start": round(start, 3),
                "vad_end": round(end, 3),
                "vad_chunk_id": f"{rel}::vad{idx:02d}",
                dur_key: round(end - start, 3),
                "vad_speech_timestamps": json.dumps([[round(s, 3), round(e, 3)] for s, e in group], separators=(",", ":")),
            })
            out_rows.append(new_row)
        
        return out_rows, None
        
    except Exception as e:
        return [], {"rel_filepath": rel, "speaker_id": speaker_id, "duration_s": duration_s, "error": str(e)}


class SileroCsvVad:
    """File-level Silero VAD with multiprocessing support.
    
    This class applies voice activity detection to audio files in parallel using
    multiple CPU workers. Each worker loads the Silero VAD model once at startup
    for efficiency.
    """

    @staticmethod
    def _compute_speaker_stats(
        input_rows: List[Dict[str, Any]],
        out_rows: List[Dict[str, Any]],
        skipped: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute speaker-level exclusion statistics. This is relevant in case some 
        speaker were completely excluded due to all their files being skipped by VAD.
        """
        # Count files per speaker in original input
        speakers_before: Dict[str, int] = {}
        for r in input_rows:
            spk = str(r.get("speaker_id", "unknown"))
            speakers_before[spk] = speakers_before.get(spk, 0) + 1
        
        # Count files per speaker in output
        speakers_after: Dict[str, int] = {}
        for r in out_rows:
            spk = str(r.get("speaker_id", "unknown"))
            speakers_after[spk] = speakers_after.get(spk, 0) + 1
        
        # Find fully excluded speakers (all files skipped)
        excluded_speakers = [spk for spk in speakers_before if spk not in speakers_after]
        num_speakers_before = len(speakers_before)
        num_speakers_after = len(speakers_after)
        num_excluded_speakers = len(excluded_speakers)
        excluded_speakers_pct = num_excluded_speakers / num_speakers_before * 100 if num_speakers_before else 0
        
        # Compute per-skipped-file speaker exclusion info
        skipped_speaker_stats: Dict[str, Dict[str, Any]] = {}
        for s in skipped:
            spk = s.get("speaker_id", "unknown")
            if spk not in skipped_speaker_stats:
                total_files_for_spk = speakers_before.get(spk, 1)
                skipped_files_for_spk = sum(1 for x in skipped if x.get("speaker_id") == spk)
                skipped_speaker_stats[spk] = {
                    "total_files": total_files_for_spk,
                    "skipped_files": skipped_files_for_spk,
                    "skip_pct": skipped_files_for_spk / total_files_for_spk * 100 if total_files_for_spk else 0,
                    "fully_excluded": spk in excluded_speakers,
                }
        
        return {
            "speakers_before": speakers_before,
            "speakers_after": speakers_after,
            "excluded_speakers": excluded_speakers,
            "num_speakers_before": num_speakers_before,
            "num_speakers_after": num_speakers_after,
            "num_excluded_speakers": num_excluded_speakers,
            "excluded_speakers_pct": excluded_speakers_pct,
            "skipped_speaker_stats": skipped_speaker_stats,
        }

    @staticmethod
    def _write_skip_list_and_log( 
        input_rows: List[Dict[str, Any]],
        output_rows: List[Dict[str, Any]],
        skip_list_path: Optional[Path],
        split_name: str,
        speaker_stats: Dict[str, Any],
        skipped: List[Dict[str, Any]],
        recording_duration_key: str,
    ) -> None: 
        """Write skip list file and log VAD summary.
        """
        excluded_speakers = speaker_stats["excluded_speakers"]
        num_speakers_before = speaker_stats["num_speakers_before"]
        num_speakers_after = speaker_stats["num_speakers_after"]
        num_excluded_speakers = speaker_stats["num_excluded_speakers"]
        excluded_speakers_pct = speaker_stats["excluded_speakers_pct"]
        skipped_speaker_stats = speaker_stats["skipped_speaker_stats"]

        total_files = len(input_rows)
        total_seconds_before = sum(float(r.get(recording_duration_key, 0.0)) for r in input_rows)
        total_seconds_after = sum(float(r.get(recording_duration_key, 0.0)) for r in output_rows) if output_rows else 0.0
        skip_pct = len(skipped) / total_files * 100 if total_files else 0
        reduction_pct = (1 - total_seconds_after / total_seconds_before) * 100 if total_seconds_before else 0
        num_output_segments = len(output_rows)

        # Write skip list
        if skip_list_path:
            skip_list_path = Path(skip_list_path)  # Normalize to Path
            try:
                skip_list_path.parent.mkdir(parents=True, exist_ok=True)
                lines = [
                    f"# split={split_name}",
                    f"# total_files={total_files}",
                    f"# skipped_files={len(skipped)}",
                    f"# skipped_percent={skip_pct:.2f}",
                    f"# minutes_before={total_seconds_before/60:.2f}",
                    f"# minutes_after={total_seconds_after/60:.2f}",
                    f"# reduction_percent={reduction_pct:.1f}",
                    f"# speakers_before={num_speakers_before}",
                    f"# speakers_after={num_speakers_after}",
                    f"# excluded_speakers={num_excluded_speakers}",
                    f"# excluded_speakers_percent={excluded_speakers_pct:.2f}",
                    f"# excluded_speaker_ids={','.join(sorted(excluded_speakers)) if excluded_speakers else 'none'}",
                    "#",
                    "# Per-file details (speaker_skip_pct = % of speaker's files skipped):",
                    "rel_filepath,speaker_id,duration_s,speaker_skip_pct,speaker_fully_excluded,error",
                ]
                for s in skipped:
                    spk = s.get("speaker_id", "unknown")
                    stats = skipped_speaker_stats.get(spk, {})
                    lines.append(
                        f"{s['rel_filepath']},{spk},{s.get('duration_s', 0)},{stats.get('skip_pct', 0):.1f},{stats.get('fully_excluded', False)},{s.get('error', '')}"
                    )
                
                skip_list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                log.info(f"Wrote skip list: {skip_list_path}")
            except Exception as e:
                log.exception(f"Failed to write skip list: {e}")

        # Log summary including speaker statistics
        log.info(f"VAD complete: {num_output_segments} segments from {total_files - len(skipped)} files, "
                 f"skipped {len(skipped)} ({skip_pct:.1f}%), "
                 f"reduced {total_seconds_before/60:.1f} \u2192 {total_seconds_after/60:.1f} min ({reduction_pct:.1f}%)")
        log.info(f"Speaker stats: {num_speakers_before} \u2192 {num_speakers_after} speakers "
                 f"({num_excluded_speakers} fully excluded, {excluded_speakers_pct:.1f}%)")


    def __init__(
        self,
        enabled: bool = False,
        apply_to_splits: Optional[Sequence[str]] = ("train", "val"),
        sample_rate: int = 16000,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
        merge_gap_s: float = 0.10,
        drop_short_speech_s: float = 0.20,
        long_silence_ratio: float = 0.20,
        long_silence_min_s: float = 0.60,
        long_silence_max_s: float = 2.00,
        segment_max_silence_ratio: float = 0.60,
        num_workers: Optional[int] = None,
        chunksize: int = 4,
    ) -> None:
        """Initialize VAD processor.
        
        Args:
            enabled: Whether VAD is enabled
            apply_to_splits: Which splits to apply VAD to (None = all)
            sample_rate: Audio sample rate in Hz
            min_speech_duration_ms: Minimum speech segment duration (Silero param)
            min_silence_duration_ms: Minimum silence between speech (Silero param)
            speech_pad_ms: Padding around speech segments (Silero param)
            merge_gap_s: Merge segments with gaps shorter than this
            drop_short_speech_s: Drop speech segments shorter than this
            long_silence_ratio: Ratio of file duration for long silence threshold
            long_silence_min_s: Minimum long silence threshold
            long_silence_max_s: Maximum long silence threshold
            segment_max_silence_ratio: Max silence ratio in segments (unused currently)
            num_workers: Number of worker processes (None = CPU count - 1)
            chunksize: Task chunk size for workers (auto-adjusted per dataset)
        """
        self.enabled = enabled
        self.apply_to_splits = tuple(apply_to_splits or ())
        self.sample_rate = sample_rate
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.merge_gap_s = merge_gap_s
        self.drop_short_speech_s = drop_short_speech_s
        self.long_silence_ratio = long_silence_ratio
        self.long_silence_min_s = long_silence_min_s
        self.long_silence_max_s = long_silence_max_s
        self.segment_max_silence_ratio = segment_max_silence_ratio
        self.num_workers = max(1, mp.cpu_count() - 1) if num_workers is None else max(1, num_workers)
        self.chunksize = max(1, chunksize)

    def should_apply(self, split_name: str) -> bool:
        """Check if VAD should be applied to this split."""
        if not self.enabled:
            return False
        if not self.apply_to_splits:
            return True
        
        s = split_name.lower() if split_name else ""
        return any(t == s or (t in {"train", "val", "valid", "dev"} and t in s) 
                   for t in (str(x).lower() for x in self.apply_to_splits))

    def apply(
        self,
        rows: Iterable[Dict[str, Any]],
        *,
        audio_root: Path,
        split_name: str,
        rel_filepath_key: str = "rel_filepath",
        recording_duration_key: str = "recording_duration",
        skip_list_path: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        """Apply VAD to all rows using multiprocessing.
        
        Args:
            rows: Input metadata rows (dicts with audio info)
            audio_root: Root directory containing audio files
            split_name: Name of the split being processed
            rel_filepath_key: Key for relative file path in rows
            recording_duration_key: Key for duration in rows
            skip_list_path: Optional path to write skipped files list
            
        Returns:
            List of output rows with VAD information added
        """
        if not self.should_apply(split_name):
            return list(rows)

        rows = list(rows)
        if not rows:
            log.warning(f"No files to process for split={split_name}")
            return []
        
        total_files = len(rows)
        total_seconds_before = sum(float(r[recording_duration_key]) for r in rows)
        
        # Adapt chunksize: aim for ~4 chunks per worker.
        # This balances IPC overhead with progress update frequency
        chunksize = max(1, min(self.chunksize, max(1, total_files // (self.num_workers * 4))))
        
        log.info(f"Applying VAD to {total_files} files ({total_seconds_before/60:.1f} min) "
                 f"using {self.num_workers} workers (chunksize={chunksize})")

        # Setup multiprocessing
        process_func = partial(
            _process_file,
            audio_root=audio_root,
            rel_key=rel_filepath_key,
            dur_key=recording_duration_key,
            merge_gap_s=self.merge_gap_s,
            drop_short_s=self.drop_short_speech_s,
            silence_ratio=self.long_silence_ratio,
            silence_min_s=self.long_silence_min_s,
            silence_max_s=self.long_silence_max_s,
            min_speech_ms=self.min_speech_duration_ms,
            min_silence_ms=self.min_silence_duration_ms,
            speech_pad_ms=self.speech_pad_ms,
        )
        
        # Worker initializer only needs sample_rate (VAD params are per-file)
        init_func = partial(_init_worker, self.sample_rate)

        # Process files in parallel
        out, skipped = [], []
        with mp.Pool(processes=self.num_workers, initializer=init_func) as pool:
            results = pool.imap_unordered(process_func, rows, chunksize=chunksize)
            for out_rows, skipped_info in tqdm(results, total=total_files, desc=f"VAD {split_name}", unit="file"):
                if out_rows:
                    out.extend(out_rows)
                if skipped_info:
                    skipped.append(skipped_info)

        # Compute speaker-level exclusion statistics
        speaker_stats = self._compute_speaker_stats(input_rows=rows, output_rows=out, skipped=skipped)

        # Write skip list and log summary
        self._write_skip_list_and_log(
            input_rows=rows,
            output_rows=out,
            skip_list_path=skip_list_path,
            split_name=split_name,
            speaker_stats=speaker_stats,
            skipped=skipped,
            recording_duration_key=recording_duration_key,
        )

        return out