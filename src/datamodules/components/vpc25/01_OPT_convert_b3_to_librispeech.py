#!/usr/bin/env python3
"""
Convert B3 metadata to original LibriSpeech metadata format.
You need to have prepared B3 before running this script.
"""

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, Optional, List

import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
FILENAME_PATTERN = r'(\d+)-(\d+)-(\d+)\.wav'
SPLIT_MAPPING = {
    'dev': 'dev-clean',
    'test': 'test-clean',
    'train': 'train-clean-360'
}


def convert_filepath(filepath: str, split_name: str) -> str:
    """
    Convert B3 filepath to LibriSpeech filepath format.
    
    Args:
        filepath: The original B3 filepath
        split_name: The dataset split name (dev, test, train)
    
    Returns:
        The converted LibriSpeech format filepath
    
    Example:
        B3/data/libri_dev_trials_m_B3/wav/422-122949-0001.wav
        ->
        librispeech/dev-clean/LibriSpeech/dev-clean/422/122949/422-122949-0001.flac
    """
    # Extract speaker ID and utterance info from the filename
    filename = Path(filepath).name
    match = re.match(FILENAME_PATTERN, filename)
    
    if not match:
        logger.warning(f"Could not parse filename: {filename}, using original path")
        return filepath
    
    speaker_id, chapter_id, utterance_id = match.groups()
    
    # Determine the split (dev-clean, test-clean, train-clean-360)
    for key, libri_split in SPLIT_MAPPING.items():
        if key in split_name.lower():
            break
    else:
        libri_split = SPLIT_MAPPING['train']  # Default to train
    
    # Construct the new path
    new_path = f"librispeech/{libri_split}/LibriSpeech/{libri_split}/{speaker_id}/{chapter_id}/{speaker_id}-{chapter_id}-{utterance_id}.flac"
    
    return new_path


def process_metadata_file(input_file: Path, output_file: Path) -> None:
    """
    Process a single metadata file by converting B3 paths to LibriSpeech format.
    
    Args:
        input_file: Path to the input B3 metadata file
        output_file: Path where to save the converted metadata
    """
    logger.info(f"Processing {input_file} -> {output_file}")
    
    try:
        # Read the CSV file
        df = pd.read_csv(input_file, sep='|')
    except Exception as e:
        logger.error(f"Error reading file {input_file}: {e}")
        return
    
    # Infer split name from the input file path
    file_path_str = str(input_file).lower()
    for key, split_value in SPLIT_MAPPING.items():
        if key in file_path_str:
            split_name = split_value
            break
    else:
        split_name = SPLIT_MAPPING['train']  # Default to train
    
    logger.info(f"Inferred split name: {split_name}")

    # Process all columns that might contain file paths
    path_columns = [col for col in df.columns if 'path' in col.lower() or 'file' in col.lower()]
    
    for col in path_columns:
        # Convert the paths using the split value
        df[col] = df.apply(
            lambda row: convert_filepath(row[col], split_name) if isinstance(row[col], str) else row[col], 
            axis=1
        )

    # Add or modify the model column
    if 'model' in df.columns:
        df['model'] = 'librispeech'

    # Update source/split column
    source_col = 'split' if 'split' in df.columns else 'source'
    if source_col in df.columns:
        df[source_col] = df[source_col].apply(lambda x: x.replace("B3", "librispeech") if isinstance(x, str) else x)

    try:
        # Save the modified dataframe
        df.to_csv(output_file, sep='|', index=False)
        logger.info(f"Saved {output_file}")
    except Exception as e:
        logger.error(f"Error saving file {output_file}: {e}")


def main() -> None:
    """Main function to process all metadata files."""
    parser = argparse.ArgumentParser(description="Convert B3 metadata to LibriSpeech format")
    parser.add_argument("--input_dir", required=True, help="Input directory with B3 metadata")
    parser.add_argument("--output_dir", required=True, help="Output directory for LibriSpeech metadata")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        logger.error(f"Input directory {input_dir} does not exist")
        return
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each CSV file in the input directory
    csv_files = list(input_dir.glob("*.csv"))
    if not csv_files:
        logger.warning(f"No CSV files found in {input_dir}")
    
    for file in csv_files:
        # Skip temporary files
        if file.name.startswith('.'):
            continue
            
        output_file = output_dir / file.name
        process_metadata_file(file, output_file)
    
    logger.info(f"Conversion complete. Output files saved to {output_dir}")


if __name__ == "__main__":
    main()
