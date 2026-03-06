#!/usr/bin/env python3
"""Convert FLAC audio files to WAV format.

Based on WeSpeaker's flac2wav.py but adapted for this project's structure.
Uses sox by default for more robust FLAC decoding (handles malformed
seek tables and other FLAC issues). Falls back to soundfile if sox is
not available.

Usage:
    python scripts/datasets/flac2wav.py --dataset_dir data/cnceleb/CN-Celeb_flac
    python scripts/datasets/flac2wav.py --dataset_dir data/cnceleb --nj 16
"""

import argparse
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

# Target format
TARGET_SAMPLE_RATE = 16000
TARGET_SUBTYPE = "PCM_16"  # 16-bit

# Check if sox is available at module load
SOX_AVAILABLE = shutil.which("sox") is not None


def find_flac_files(dataset_dir: Path) -> list[tuple[Path, Path]]:
    """Find all FLAC files and compute their WAV output paths.

    Converts directory names from *_flac to *_wav to maintain parallel structure.
    E.g., CN-Celeb_flac/data/id00001/file.flac -> CN-Celeb_wav/data/id00001/file.wav

    Args:
        dataset_dir: Root directory to search (can be the flac subdir or parent)

    Returns:
        List of (flac_path, wav_path) tuples
    """
    dataset_dir = Path(dataset_dir).resolve()

    # Find all FLAC files
    flac_files = list(dataset_dir.rglob("*.flac"))
    if not flac_files:
        print(f"No FLAC files found in {dataset_dir}")
        return []

    print(f"Found {len(flac_files)} FLAC files in {dataset_dir}")

    conversions = []
    created_dirs = set()

    for flac_path in flac_files:
        # Convert path: replace _flac with _wav in directory names
        wav_path_str = str(flac_path).replace("_flac", "_wav").replace(".flac", ".wav")
        wav_path = Path(wav_path_str)

        # Create output directory if needed
        wav_dir = wav_path.parent
        if wav_dir not in created_dirs:
            wav_dir.mkdir(parents=True, exist_ok=True)
            created_dirs.add(wav_dir)

        conversions.append((flac_path, wav_path))

    return conversions


def _convert_with_sox(flac_path: Path, wav_path: Path) -> tuple[bool, str]:
    """Convert FLAC to WAV using sox."""
    # sox -t flac input.flac -t wav -r 16k -b 16 output.wav channels 1
    cmd = [
        "sox",
        "-t", "flac", str(flac_path),
        "-t", "wav",
        "-r", "16k",
        "-b", "16",
        str(wav_path),
        "channels", "1",
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        return (False, result.stderr.strip())
    return (True, "")


def _convert_with_soundfile(flac_path: Path, wav_path: Path) -> tuple[bool, str]:
    """Convert FLAC to WAV using soundfile (libsndfile).

    Resamples to 16kHz mono 16-bit PCM to match sox output.
    """
    # Read FLAC file
    audio, sample_rate = sf.read(flac_path, dtype="float32")

    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sample_rate != TARGET_SAMPLE_RATE:
        # Simple linear interpolation resampling
        duration = len(audio) / sample_rate
        target_length = int(duration * TARGET_SAMPLE_RATE)
        indices = np.linspace(0, len(audio) - 1, target_length)
        audio = np.interp(indices, np.arange(len(audio)), audio)

    # Write WAV file (16-bit PCM)
    sf.write(wav_path, audio, TARGET_SAMPLE_RATE, subtype=TARGET_SUBTYPE)
    return (True, "")


def convert_flac_to_wav(args: tuple[Path, Path]) -> tuple[Path, bool, str]:
    """Convert a single FLAC file to WAV.

    Uses sox if available (more tolerant of malformed FLAC files),
    otherwise falls back to soundfile.

    Args:
        args: Tuple of (flac_path, wav_path)

    Returns:
        Tuple of (flac_path, success, error_message)
    """
    flac_path, wav_path = args

    # Skip if already converted
    if wav_path.exists():
        return (flac_path, True, "skipped (exists)")

    try:
        if SOX_AVAILABLE:
            success, error = _convert_with_sox(flac_path, wav_path)
        else:
            success, error = _convert_with_soundfile(flac_path, wav_path)

        return (flac_path, success, error)
    except subprocess.TimeoutExpired:
        return (flac_path, False, "timeout")
    except Exception as e:
        return (flac_path, False, str(e))


def main():
    parser = argparse.ArgumentParser(
        description="Convert FLAC files to WAV (uses sox if available, otherwise soundfile)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Directory containing FLAC files (e.g., data/cnceleb or data/cnceleb/CN-Celeb_flac)",
    )
    parser.add_argument(
        "--nj",
        type=int,
        default=8,
        help="Number of parallel jobs",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    # Find all FLAC files
    print(f"Scanning {dataset_dir} for FLAC files...")
    conversions = find_flac_files(dataset_dir)

    if not conversions:
        print("No FLAC files to convert.")
        return

    # Filter out already converted files for progress reporting
    to_convert = [(f, w) for f, w in conversions if not w.exists()]
    already_done = len(conversions) - len(to_convert)

    if already_done > 0:
        print(f"Already converted: {already_done} files")

    if not to_convert:
        print("All files already converted.")
        return

    backend = "sox" if SOX_AVAILABLE else "soundfile"
    print(f"Converting {len(to_convert)} FLAC files to WAV using {backend} (nj={args.nj})...")

    # Convert in parallel
    failed = []
    with ProcessPoolExecutor(max_workers=args.nj) as executor:
        futures = {executor.submit(convert_flac_to_wav, task): task for task in to_convert}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Converting"):
            flac_path, success, error = future.result()
            if not success and error != "skipped (exists)":
                failed.append((flac_path, error))

    # Report results
    successful = len(to_convert) - len(failed)
    print(f"\nConversion complete: {successful}/{len(to_convert)} successful")

    if failed:
        print(f"\nFailed conversions ({len(failed)}):")
        # Write failed files to a log
        failed_log = dataset_dir / "flac2wav_failed.txt"
        with open(failed_log, "w") as f:
            for flac_path, error in failed:
                print(f"  {flac_path}: {error}")
                f.write(f"{flac_path}|{error}\n")
        print(f"\nFailed files logged to: {failed_log}")


if __name__ == "__main__":
    main()
