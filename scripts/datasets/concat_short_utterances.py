#!/usr/bin/env python3
"""
Concatenate short utterances from the same speaker to create longer audio files.

This script runs BEFORE the main CNCeleb preprocessing pipeline. It:
1. Scans the dataset for audio files shorter than a target duration
2. Groups short files by speaker
3. Concatenates them (greedy bin-packing) until they reach target duration
4. Saves concatenated files to a separate directory (does NOT alter original data)
5. Creates a CSV mapping file to track which files have been merged

The mapping file is then read by cnceleb_prep.py to:
- Skip original short files that have been merged
- Include the new concatenated files in preprocessing

CSV format (pipe-separated):
    output_path|source_paths|duration|speaker_id|original_split
    
Where source_paths uses semicolon as delimiter for multiple paths.

Usage:
    python scripts/datasets/concat_short_utterances.py \
        --root_dir data/cnceleb \
        --cnceleb1 CN-Celeb_flac \
        --cnceleb2 CN-Celeb2_flac \
        --target_duration 5.0 \
        --output_dir data/cnceleb/concatenated \
        --mapping_file data/cnceleb/metadata/concat_mapping.map
"""

import argparse
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import soundfile as sf
import numpy as np
from tqdm.auto import tqdm


@dataclass
class AudioFileInfo:
    """Information about an audio file."""
    abs_path: Path
    rel_path: str  # Relative to sub-dataset root (e.g., CN-Celeb_flac)
    speaker_id: str
    duration: float
    sample_rate: int
    sub_dataset: str  # e.g., "CN-Celeb_flac" or "CN-Celeb2_flac"


@dataclass
class ConcatGroup:
    """A group of files to be concatenated."""
    speaker_id: str
    source_files: List[AudioFileInfo] = field(default_factory=list)
    total_duration: float = 0.0
    
    def add_file(self, file_info: AudioFileInfo):
        self.source_files.append(file_info)
        self.total_duration += file_info.duration


def scan_audio_file(args: Tuple[str, str, str]) -> Optional[AudioFileInfo]:
    """Scan a single audio file and return its info."""
    abs_path_str, sub_dataset_root_str, sub_dataset_name = args
    abs_path = Path(abs_path_str)
    sub_dataset_root = Path(sub_dataset_root_str)
    
    try:
        info = sf.info(str(abs_path))
        rel_path = abs_path.relative_to(sub_dataset_root)
        
        # Extract speaker ID from path (e.g., data/id00010/... -> id00010)
        parts = rel_path.parts
        speaker_id = None
        
        if len(parts) >= 2:
            first_part = parts[0]
            if first_part in ['data', 'dev']:
                speaker_id = parts[1]
            elif first_part.startswith('id'):
                speaker_id = first_part
        
        if speaker_id is None or not speaker_id.startswith('id'):
            return None
            
        return AudioFileInfo(
            abs_path=abs_path,
            rel_path=str(rel_path),
            speaker_id=speaker_id,
            duration=info.duration,
            sample_rate=info.samplerate,
            sub_dataset=sub_dataset_name,
        )
    except Exception as e:
        print(f"Error processing {abs_path}: {e}")
        return None


def scan_dataset(
    root_dir: Path,
    sub_datasets: List[str],
    n_jobs: int = 8,
) -> List[AudioFileInfo]:
    """Scan all audio files in the dataset."""
    all_files = []
    
    for sub_dataset in sub_datasets:
        sub_root = root_dir / sub_dataset
        if not sub_root.exists():
            print(f"Warning: Sub-dataset {sub_root} does not exist, skipping...")
            continue
            
        # Scan data directory
        data_dir = sub_root / 'data'
        if data_dir.exists():
            audio_files = list(data_dir.rglob('*.flac'))
            print(f"Found {len(audio_files)} audio files in {data_dir}")
            all_files.extend([
                (str(f), str(sub_root), sub_dataset) 
                for f in audio_files
            ])
    
    print(f"Scanning {len(all_files)} audio files...")
    
    results = []
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {executor.submit(scan_audio_file, task): task for task in all_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Scanning files"):
            result = future.result()
            if result is not None:
                results.append(result)
    
    return results


def group_short_files_by_speaker(
    files: List[AudioFileInfo],
    target_duration: float,
) -> Tuple[Dict[str, List[AudioFileInfo]], List[AudioFileInfo]]:
    """
    Group files by speaker, separating short files from long files.
    
    Returns:
        short_by_speaker: Dict mapping speaker_id to list of short files
        long_files: List of files that are already >= target_duration
    """
    short_by_speaker: Dict[str, List[AudioFileInfo]] = defaultdict(list)
    long_files: List[AudioFileInfo] = []
    
    for file_info in files:
        if file_info.duration < target_duration:
            short_by_speaker[file_info.speaker_id].append(file_info)
        else:
            long_files.append(file_info)
    
    return dict(short_by_speaker), long_files


def create_concat_groups(
    short_by_speaker: Dict[str, List[AudioFileInfo]],
    target_duration: float,
) -> List[ConcatGroup]:
    """
    Create groups of files to concatenate using greedy bin-packing.
    
    Files within each speaker are sorted by duration (longest first) and
    greedily packed into groups until each group reaches target_duration.
    
    Args:
        short_by_speaker: Dict mapping speaker_id to list of short files
        target_duration: Minimum duration threshold for concatenated files
        
    Returns:
        List of ConcatGroup objects, each containing files to concatenate
    """
    groups = []
    
    for speaker_id, files in short_by_speaker.items():
        # Sort by duration descending (pack largest first)
        sorted_files = sorted(files, key=lambda x: x.duration, reverse=True)
        
        current_group = ConcatGroup(speaker_id=speaker_id)

        for file_info in sorted_files:
            current_group.add_file(file_info)

            # Once we meet the minimum threshold, finalize the group.
            if current_group.total_duration >= target_duration and len(current_group.source_files) > 1:
                groups.append(current_group)
                current_group = ConcatGroup(speaker_id=speaker_id)

        # If we have leftover files that didn't reach the threshold, absorb them into
        # the last valid group for this speaker (so target_duration acts as a minimum).
        if current_group.source_files:
            if groups:
                groups[-1].source_files.extend(current_group.source_files)
                groups[-1].total_duration += current_group.total_duration
            # else: not enough material to reach target_duration for this speaker; skip
    
    return groups


def concatenate_audio_files(
    group: ConcatGroup,
    root_dir: Path,
    output_dir: Path,
    group_idx: int,
) -> Tuple[Optional[str], List[str], float, Optional[str]]:
    """
    Concatenate audio files in a group and save to output directory.
    
    Returns:
        Tuple of (output_rel_path, source_rel_paths, total_duration, original_split) 
        or (None, [], 0, None) on error
    """
    try:
        # Read and concatenate audio
        audio_segments = []
        sample_rate = None
        
        for file_info in group.source_files:
            audio, sr = sf.read(str(file_info.abs_path))
            if sample_rate is None:
                sample_rate = sr
            elif sr != sample_rate:
                # Resample if needed (simple case: just use first sample rate)
                print(f"Warning: Sample rate mismatch in {file_info.abs_path}")
                continue
            audio_segments.append(audio)
        
        if not audio_segments:
            return None, [], 0.0, None
        
        # Concatenate
        concatenated = np.concatenate(audio_segments)
        
        # Create output path: concatenated/<speaker_id>/<concat_idx>.flac
        # We use a simple naming: speaker_id/concat_XXXX.flac
        speaker_dir = output_dir / group.speaker_id
        speaker_dir.mkdir(parents=True, exist_ok=True)
        
        output_filename = f"concat_{group_idx:06d}.flac"
        output_path = speaker_dir / output_filename
        
        # Save concatenated audio
        sf.write(str(output_path), concatenated, sample_rate)
        
        # Return relative path from root_dir
        output_rel_path = str(output_path.relative_to(root_dir))
        source_rel_paths = [
            str(Path(f.sub_dataset) / f.rel_path) 
            for f in group.source_files
        ]
        
        # Get original split from first source file (all files in group are from same speaker/split)
        original_split = group.source_files[0].sub_dataset if group.source_files else None
        
        return output_rel_path, source_rel_paths, len(concatenated) / sample_rate, original_split
        
    except Exception as e:
        print(f"Error concatenating group {group_idx}: {e}")
        return None, [], 0.0, None


def main():
    parser = argparse.ArgumentParser(
        description="Concatenate short utterances from the same speaker"
    )
    parser.add_argument(
        "--root_dir", type=str, required=True,
        help="Root directory containing CNCeleb datasets"
    )
    parser.add_argument(
        "--cnceleb1", type=str, required=True,
        help="Name of CNCeleb1 subdirectory (e.g., CN-Celeb_flac)"
    )
    parser.add_argument(
        "--cnceleb2", type=str, default=None,
        help="Name of CNCeleb2 subdirectory (optional)"
    )
    parser.add_argument(
        "--target_duration", type=float, default=5.0,
        help="Target duration for concatenated files (seconds)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for concatenated files"
    )
    parser.add_argument(
        "--mapping_file", type=str, required=True,
        help="Path to save the CSV mapping file"
    )
    parser.add_argument(
        "--n_jobs", type=int, default=None,
        help="Number of parallel jobs (default: min(cpu_count, 8))"
    )
    
    args = parser.parse_args()
    
    root_dir = Path(args.root_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    mapping_file = Path(args.mapping_file).expanduser().resolve()
    
    n_jobs = args.n_jobs if args.n_jobs else min(cpu_count(), 8)
    
    # Build sub-dataset list
    sub_datasets = [args.cnceleb1]
    if args.cnceleb2:
        sub_datasets.append(args.cnceleb2)
    
    print(f"=" * 60)
    print(f"Concatenating short utterances for CNCeleb")
    print(f"=" * 60)
    print(f"Root directory: {root_dir}")
    print(f"Sub-datasets: {sub_datasets}")
    print(f"Target duration (minimum): {args.target_duration}s")
    print(f"Output directory: {output_dir}")
    print(f"Mapping file: {mapping_file}")
    print()
    
    # Step 1: Scan all audio files
    print("Step 1: Scanning audio files...")
    all_files = scan_dataset(root_dir, sub_datasets, n_jobs)
    print(f"Found {len(all_files)} valid audio files")
    
    # Step 2: Group short files by speaker
    print("\nStep 2: Grouping short files by speaker...")
    short_by_speaker, long_files = group_short_files_by_speaker(
        all_files, args.target_duration
    )
    
    total_short = sum(len(files) for files in short_by_speaker.values())
    print(f"Found {total_short} short files (< {args.target_duration}s) across {len(short_by_speaker)} speakers")
    print(f"Found {len(long_files)} long files (>= {args.target_duration}s)")
    
    # Step 3: Create concatenation groups
    print("\nStep 3: Creating concatenation groups...")
    concat_groups = create_concat_groups(
        short_by_speaker, args.target_duration
    )
    print(f"Created {len(concat_groups)} concatenation groups")
    
    # Calculate how many files will be merged
    files_to_merge = set()
    for group in concat_groups:
        for f in group.source_files:
            files_to_merge.add(str(Path(f.sub_dataset) / f.rel_path))
    print(f"Total files to be merged: {len(files_to_merge)}")
    
    # Step 4: Perform concatenation
    print("\nStep 4: Concatenating audio files...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect mapping rows for CSV
    mapping_rows = []
    
    successful_concats = 0
    for idx, group in enumerate(tqdm(concat_groups, desc="Concatenating")):
        output_rel_path, source_rel_paths, duration, original_split = concatenate_audio_files(
            group, root_dir, output_dir, idx
        )
        
        if output_rel_path:
            mapping_rows.append({
                "output_path": output_rel_path,
                "source_paths": ";".join(source_rel_paths),  # Semicolon-delimited
                "duration": duration,
                "speaker_id": group.speaker_id,
                "original_split": original_split,  # Preserves CN-Celeb1/CN-Celeb2 origin
            })
            successful_concats += 1
    
    # Step 5: Save mapping file as CSV
    print("\nStep 5: Saving mapping file...")
    mapping_file.parent.mkdir(parents=True, exist_ok=True)
    
    mapping_df = pd.DataFrame(mapping_rows)
    mapping_df.to_csv(mapping_file, sep="|", index=False)
    
    # Count total merged source files
    total_merged = sum(len(row["source_paths"].split(";")) for row in mapping_rows)
    
    print(f"\n{'=' * 60}")
    print(f"Concatenation complete!")
    print(f"{'=' * 60}")
    print(f"Successfully created {successful_concats} concatenated files")
    print(f"Merged {total_merged} source files")
    print(f"Mapping saved to: {mapping_file}")
    print()
    print("Next steps:")
    print(f"1. Set concat_mapping_file in CNCeleb config to: {mapping_file}")
    print("2. The preprocessing will skip merged source files and include concatenated files")


if __name__ == "__main__":
    main()
