#!/usr/bin/env python3
"""
Concatenate short utterances from the same speaker AND genre to create longer audio files.

This script runs BEFORE the main CNCeleb preprocessing pipeline. It:
1. Scans the dataset for audio files shorter than a target duration
2. Groups short files by (speaker_id, genre) - only concatenates files from same genre
3. Concatenates them (greedy bin-packing) until they reach target duration
4. Saves concatenated files to a separate directory (does NOT alter original data)
5. Creates a CSV mapping file to track which files have been merged

The mapping file is then read by cnceleb_prep.py to:
- Skip original short files that have been merged
- Include the new concatenated files in preprocessing

CNCeleb directory structure: data/{speaker_id}/{genre}-{recording}.flac
Genre is extracted from filename prefix (e.g., "singing-01-002.flac" -> genre="singing")

CSV format (pipe-separated):
    output_path|source_paths|duration|speaker_id|genre|dataset_source

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
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm.auto import tqdm


class ConcatenationError(Exception):
    """Exception raised for errors during concatenation."""
    pass


class ValidationError(Exception):
    """Exception raised for validation failures."""
    pass


@dataclass
class AudioFileInfo:
    """Information about an audio file."""
    abs_path: Path
    rel_path: str  # Relative to sub-dataset root (e.g., CN-Celeb_flac)
    speaker_id: str
    genre: str  # e.g., "singing", "speech", "interview"
    duration: float
    sample_rate: int
    sub_dataset: str  # e.g., "CN-Celeb_flac" or "CN-Celeb2_flac"


@dataclass
class ConcatGroup:
    """A group of files to be concatenated (same speaker AND genre)."""
    speaker_id: str
    genre: str
    source_files: List[AudioFileInfo] = field(default_factory=list)
    total_duration: float = 0.0

    def add_file(self, file_info: AudioFileInfo):
        """Add a file to this concatenation group."""
        self.source_files.append(file_info)
        self.total_duration += file_info.duration


def scan_audio_file(args: Tuple[str, str, str]) -> Optional[AudioFileInfo]:
    """
    Scan a single audio file and return its info.
    
    Args:
        args: Tuple of (abs_path, sub_dataset_root, sub_dataset_name)
        
    Returns:
        AudioFileInfo object or None if the file should be skipped
    """
    abs_path_str, sub_dataset_root_str, sub_dataset_name = args
    abs_path = Path(abs_path_str)
    sub_dataset_root = Path(sub_dataset_root_str)
    
    if not abs_path.exists():
        raise FileNotFoundError(f"File does not exist: {abs_path}")    
    info = sf.info(str(abs_path))
        
    # Compute relative path
    try:
        rel_path = abs_path.relative_to(sub_dataset_root)
    except ValueError as e:
        raise ValidationError(
            f"File {abs_path} is not under sub-dataset root {sub_dataset_root}"
        ) from e
    
    # Extract speaker ID and genre from path
    # Expected format: data/{speaker_id}/{genre}-{recording}.flac
    # Genre is the prefix of filename before first hyphen (e.g., "singing-01-002.flac" -> "singing")
    parts = rel_path.parts
    speaker_id = None
    genre = None

    if len(parts) >= 2 and parts[0] in ['data', 'dev']:
        # data/id00010/singing-01-002.flac -> speaker=id00010
        speaker_id = parts[1]
    elif len(parts) >= 1 and parts[0].startswith('id'):
        # id00010/singing-01-002.flac -> speaker=id00010
        speaker_id = parts[0]

    if not speaker_id or not speaker_id.startswith('id'):
        raise ValidationError(f"Could not extract speaker ID from path: {rel_path}")

    # Extract genre from filename (e.g., "singing-01-002.flac" -> "singing")
    filename = rel_path.name  # Get filename without directory
    filename_stem = Path(filename).stem  # Remove .flac extension
    if '-' in filename_stem:
        genre = filename_stem.split('-')[0]
    else:
        raise ValidationError(f"Could not extract genre from filename: {filename}")

    return AudioFileInfo(
        abs_path=abs_path,
        rel_path=str(rel_path),
        speaker_id=speaker_id,
        genre=genre,
        duration=info.duration,
        sample_rate=info.samplerate,
        sub_dataset=sub_dataset_name,
    )


def scan_dataset(
    root_dir: Path,
    sub_datasets: List[str],
    n_jobs: int = 8,
) -> List[AudioFileInfo]:
    """
    Scan all audio files in the dataset.
    
    Args:
        root_dir: Root directory containing sub-datasets
        sub_datasets: List of sub-dataset directory names
        n_jobs: Number of parallel workers
        
    Returns:
        List of AudioFileInfo objects
    """
    if not root_dir.exists():
        raise ValidationError(f"Root directory does not exist: {root_dir}")
    
    all_files = []
    
    for sub_dataset in sub_datasets:
        sub_root = root_dir / sub_dataset
        if not sub_root.exists():
            raise ValidationError(f"Sub-dataset directory does not exist: {sub_root}")
        
        # Check for data directory
        data_dir = sub_root / 'data'
        assert sub_root.exists(), f"Sub-dataset directory does not exist: {sub_root}"            
        audio_files = list(data_dir.rglob('*.flac'))

        if not audio_files:
            raise ValidationError(f"WARNING: No .flac files found in {data_dir}")
        else:
            print(f"Found {len(audio_files)} audio files in {data_dir}")
    
        all_files.extend(
            (str(f), str(sub_root), sub_dataset) 
            for f in audio_files
        )    
    print(f"\nScanning {len(all_files)} audio files...")
    
    results = []
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {executor.submit(scan_audio_file, task): task for task in all_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Scanning files"):
            try:
                result = future.result()
            except Exception as e:
                raise RuntimeError(f"ERROR: Failed to scan file {futures[future]}: {e}") from e
            if result is not None:
                results.append(result)
    return results


def group_short_files_by_speaker_and_genre(
    files: List[AudioFileInfo],
    target_duration: float,
) -> Tuple[Dict[Tuple[str, str], List[AudioFileInfo]], List[AudioFileInfo]]:
    """
    Group files by (speaker_id, genre), separating short files from long files.

    Args:
        files: List of all audio files
        target_duration: Duration threshold

    Returns:
        Tuple of (short_by_speaker_genre, long_files)
        where short_by_speaker_genre is keyed by (speaker_id, genre) tuple
    """
    if not files:
        raise ValidationError("Cannot group empty file list")

    if target_duration <= 0:
        raise ValidationError(f"Invalid target_duration: {target_duration}")

    short_by_speaker_genre: Dict[Tuple[str, str], List[AudioFileInfo]] = defaultdict(list)
    long_files: List[AudioFileInfo] = []

    for file_info in files:
        if file_info.duration < target_duration:
            key = (file_info.speaker_id, file_info.genre)
            short_by_speaker_genre[key].append(file_info)
        else:
            long_files.append(file_info)

    return dict(short_by_speaker_genre), long_files


def create_concat_groups(
    short_by_speaker_genre: Dict[Tuple[str, str], List[AudioFileInfo]],
    target_duration: float,
    min_threshold: Optional[float] = None,
) -> List[ConcatGroup]:
    """
    Create groups of files to concatenate using greedy bin-packing.

    Files within each (speaker_id, genre) group are sorted by duration (longest first)
    and greedily packed into groups until each group reaches target_duration.
    Leftover files that don't reach the target are either:
    - Merged into the last group if one exists
    - Created as a new group if they meet min_threshold (when specified)
    - Reported as insufficient material otherwise

    Args:
        short_by_speaker_genre: Dict mapping (speaker_id, genre) to list of short files
        target_duration: Target duration for concatenated files (preferred minimum)
        min_threshold: Optional minimum threshold to create groups even if target not met.
                      Groups with multiple files >= min_threshold will be included.

    Returns:
        List of ConcatGroup objects, each containing files to concatenate
    """
    if not short_by_speaker_genre:
        raise ValidationError("Cannot create groups from empty speaker-genre dictionary")

    # Use min_threshold if provided, otherwise fall back to target_duration
    effective_min = min_threshold if min_threshold is not None else target_duration

    all_groups = []
    groups_without_enough_material = []
    groups_below_target_but_included = []

    for (speaker_id, genre), files in short_by_speaker_genre.items():
        if not files:
            continue

        # Sort by duration descending (pack largest first for better bin-packing)
        sorted_files = sorted(files, key=lambda x: x.duration, reverse=True)

        speaker_genre_groups = []
        current_group = ConcatGroup(speaker_id=speaker_id, genre=genre)

        for file_info in sorted_files:
            current_group.add_file(file_info)

            # Once we meet the target threshold with multiple files, finalize the group
            if (current_group.total_duration >= target_duration and
                len(current_group.source_files) > 1):
                speaker_genre_groups.append(current_group)
                current_group = ConcatGroup(speaker_id=speaker_id, genre=genre)

        # Handle leftover files
        if current_group.source_files:
            if speaker_genre_groups:
                # Merge into last group
                speaker_genre_groups[-1].source_files.extend(current_group.source_files)
                speaker_genre_groups[-1].total_duration += current_group.total_duration
            elif (len(current_group.source_files) > 1 and
                  current_group.total_duration >= effective_min):
                # No previous groups, but meets min_threshold with multiple files
                speaker_genre_groups.append(current_group)
                if min_threshold is not None and current_group.total_duration < target_duration:
                    groups_below_target_but_included.append(
                        f"{speaker_id}/{genre}: {len(current_group.source_files)} files, "
                        f"total {current_group.total_duration:.2f}s"
                    )
            else:
                # Cannot create valid group (single file or below min_threshold)
                total_dur = sum(f.duration for f in files)
                groups_without_enough_material.append(
                    f"{speaker_id}/{genre}: {len(files)} files, "
                    f"total {total_dur:.2f}s"
                )

        all_groups.extend(speaker_genre_groups)

    # Report groups included below target (informational)
    if groups_below_target_but_included:
        print(f"\nINFO: {len(groups_below_target_but_included)} groups included below target "
              f"({target_duration}s) but above min_threshold ({effective_min}s):")
        for info in groups_below_target_but_included[:5]:
            print(f"  - {info}")
        if len(groups_below_target_but_included) > 5:
            print(f"  ... and {len(groups_below_target_but_included) - 5} more")
        print()

    # Report (speaker, genre) combinations without valid groups
    if groups_without_enough_material:
        print(f"\nWARNING: {len(groups_without_enough_material)} (speaker, genre) groups have "
              f"insufficient material (need {effective_min}s min with 2+ files):")
        for info in groups_without_enough_material[:5]:
            print(f"  - {info}")
        if len(groups_without_enough_material) > 5:
            print(f"  ... and {len(groups_without_enough_material) - 5} more")
        print()

    return all_groups


def concatenate_audio_files(
    group: ConcatGroup,
    root_dir: Path,
    output_dir: Path,
    group_idx: int,
) -> Tuple[str, List[str], float, str, str]:
    """
    Concatenate audio files in a group and save to output directory.

    Args:
        group: ConcatGroup containing files to concatenate
        root_dir: Root directory (for computing relative paths)
        output_dir: Output directory for concatenated files
        group_idx: Index for naming the output file

    Returns:
        Tuple of (output_rel_path, source_rel_paths, total_duration, genre, dataset_source)
    """
    if not group.source_files:
        raise ValidationError(f"Group {group_idx} has no source files")

    if len(group.source_files) < 2:
        raise ValidationError(
            f"Group {group_idx} has only {len(group.source_files)} file(s), "
            f"need at least 2 for concatenation"
        )

    # Verify all files are from the same speaker
    speaker_ids = set(f.speaker_id for f in group.source_files)
    if len(speaker_ids) > 1:
        raise ValidationError(
            f"Group {group_idx} contains files from multiple speakers: {speaker_ids}"
        )

    # Verify all files are from the same genre
    genres = set(f.genre for f in group.source_files)
    if len(genres) > 1:
        raise ValidationError(
            f"Group {group_idx} contains files from multiple genres: {genres}"
        )

    # Read and concatenate audio
    audio_segments = []
    included_files = []
    sample_rate = 16000  # Default sample rate for CNCeleb

    for file_info in group.source_files:
        audio, sr = sf.read(str(file_info.abs_path), dtype="float32")
        if sr != sample_rate:
            raise ValidationError(
                f"File {file_info.abs_path} has sample rate {sr}Hz, "
                f"expected {sample_rate}Hz for CNCeleb. Please override this manually if needed."
            )

        audio_segments.append(audio)
        included_files.append(file_info)

    if not audio_segments:
        raise ConcatenationError(
            f"Group {group_idx}: No valid audio segments after reading files"
        )

    # Concatenate all segments
    concatenated = np.concatenate(audio_segments)
    assert len(concatenated) > 0, "Concatenated audio should not be empty"

    # Create output path: output_dir/{speaker_id}/{genre}/concat_XXXXXX.flac
    speaker_genre_dir = output_dir / group.speaker_id / group.genre
    speaker_genre_dir.mkdir(parents=True, exist_ok=True)

    output_filename = f"concat_{group_idx:06d}.flac"
    output_path = speaker_genre_dir / output_filename

    # Save concatenated audio
    sf.write(str(output_path), concatenated, sample_rate)

    # Verify the written file
    written_info = sf.info(str(output_path))
    expected_duration = len(concatenated) / sample_rate
    duration_diff = abs(written_info.duration - expected_duration)
    assert duration_diff < 0.01, f"Significant duration mismatch for {output_path}: "

    # Compute relative paths for CSV
    output_rel_path = str(output_path.relative_to(root_dir))
    source_rel_paths = [
        str(Path(f.sub_dataset) / f.rel_path)
        for f in included_files
    ]

    # Get dataset source from first source file (e.g., CN-Celeb_flac or CN-Celeb2_flac)
    dataset_source = included_files[0].sub_dataset
    actual_duration = len(concatenated) / sample_rate
    return output_rel_path, source_rel_paths, actual_duration, group.genre, dataset_source


def main():
    """Main entry point for the concatenation script."""
    parser = argparse.ArgumentParser(
        description="Concatenate short utterances from the same speaker AND genre",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
        help="Target minimum duration for concatenated files (seconds)"
    )
    parser.add_argument(
        "--min_threshold", type=float, default=None,
        help="Minimum threshold to create concat even if target not met (seconds). "
             "Groups with multiple files >= min_threshold will be included. "
             "Default: None (only create groups that meet target_duration)"
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
        help="Number of parallel jobs (default: min(cpu_count(), 8))"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    root_dir = Path(args.root_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    mapping_file = Path(args.mapping_file).expanduser().resolve()
    
    # Validate target duration
    if args.target_duration <= 0:
        raise ValidationError(f"Invalid target_duration: {args.target_duration}")

    # Validate min_threshold if provided
    min_threshold = args.min_threshold
    if min_threshold is not None:
        if min_threshold <= 0:
            raise ValidationError(f"Invalid min_threshold: {min_threshold}")
        if min_threshold > args.target_duration:
            raise ValidationError(
                f"min_threshold ({min_threshold}) cannot be greater than "
                f"target_duration ({args.target_duration})"
            )

    # Determine number of parallel jobs
    n_jobs = args.n_jobs if args.n_jobs else min(cpu_count(), 8)
    if n_jobs < 1:
        raise ValidationError(f"Invalid n_jobs: {n_jobs}")
    
    # Build sub-dataset list
    sub_datasets = [args.cnceleb1, args.cnceleb2] if args.cnceleb2 else [args.cnceleb1]
    
    # Print configuration
    print("=" * 60)
    print("Concatenating short utterances for CNCeleb")
    print("=" * 60)
    print(f"Root directory: {root_dir}")
    print(f"Sub-datasets: {sub_datasets}")
    print(f"Target duration: {args.target_duration}s")
    if min_threshold is not None:
        print(f"Min threshold: {min_threshold}s (include groups below target if >= this)")
    print(f"Output directory: {output_dir}")
    print(f"Mapping file: {mapping_file}")
    print(f"Parallel jobs: {n_jobs}")
    print()
    
    # Step 1: Scan all audio files
    print("Step 1: Scanning audio files...")
    all_files = scan_dataset(root_dir, sub_datasets, n_jobs)
    print(f"Found {len(all_files)} valid audio files")

    # Step 2: Group short files by (speaker, genre)
    print("\nStep 2: Grouping short files by (speaker, genre)...")
    short_by_speaker_genre, long_files = group_short_files_by_speaker_and_genre(
        all_files, args.target_duration
    )

    total_short = sum(len(files) for files in short_by_speaker_genre.values())
    unique_speakers = len(set(k[0] for k in short_by_speaker_genre.keys()))
    unique_genres = len(set(k[1] for k in short_by_speaker_genre.keys()))
    print(f"Found {total_short} short files (< {args.target_duration}s) "
          f"across {len(short_by_speaker_genre)} (speaker, genre) groups")
    print(f"  Unique speakers: {unique_speakers}, Unique genres: {unique_genres}")
    print(f"Found {len(long_files)} long files (>= {args.target_duration}s)")

    assert short_by_speaker_genre, "No short files found for concatenation"

    # Step 3: Create concatenation groups
    print("\nStep 3: Creating concatenation groups...")
    concat_groups = create_concat_groups(
        short_by_speaker_genre, args.target_duration, min_threshold
    )
    assert concat_groups, "No concatenation groups created"
    print(f"Created {len(concat_groups)} concatenation groups")

    # Calculate statistics
    files_to_merge = set()
    for group in concat_groups:
        for f in group.source_files:
            files_to_merge.add(str(Path(f.sub_dataset) / f.rel_path))
    print(f"Total source files to be merged: {len(files_to_merge)}")

    # Step 4: Perform concatenation
    print("\nStep 4: Concatenating audio files...")
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping_rows = []

    for idx, group in enumerate(tqdm(concat_groups, desc="Concatenating")):
        output_rel_path, source_rel_paths, duration, genre, dataset_source = concatenate_audio_files(
            group, root_dir, output_dir, idx
        )

        mapping_rows.append({
            "output_path": output_rel_path,
            "source_paths": ";".join(source_rel_paths),
            "duration": duration,
            "speaker_id": group.speaker_id,
            "genre": genre,
            "dataset_source": dataset_source,
        })
    
    # Step 5: Save mapping file
    print("\nStep 5: Saving mapping file...")
    mapping_file.parent.mkdir(parents=True, exist_ok=True)
    
    if not mapping_rows:
        raise ConcatenationError("No successful concatenations produced")
    
    mapping_df = pd.DataFrame(mapping_rows)
    
    # Validate mapping before saving
    if len(mapping_df) != len(concat_groups):
        raise ValidationError(
            f"Mapping row count ({len(mapping_df)}) does not match "
            f"group count ({len(concat_groups)})"
        )
    
    mapping_df.to_csv(mapping_file, sep="|", index=False)

    # Verify the saved file
    verify_df = pd.read_csv(mapping_file, sep="|")
    assert len(verify_df) == len(mapping_df), "Mismatch in saved mapping file rows"
    
    # Calculate final statistics
    total_merged = sum(len(row["source_paths"].split(";")) for row in mapping_rows)
    
    print(f"\n{'=' * 60}")
    print("Concatenation complete!")
    print("=" * 60)
    print(f"Successfully created {len(mapping_rows)} concatenated files")
    print(f"Merged {total_merged} source files")
    print(f"Mapping saved to: {mapping_file}")
    print()
    print("Next steps:")
    print(f"1. Set concat_mapping_file in CNCeleb config to: {mapping_file}")
    print("2. The preprocessing will skip merged source files and include concatenated files")


if __name__ == "__main__":
    main()
