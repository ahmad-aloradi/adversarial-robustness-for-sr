#!/usr/bin/env python3
"""
Convert B3 metadata to original LibriSpeech metadata format. You need to have prepared B3 before running this scripts
"""

import os
import argparse
import pandas as pd
import re
from pathlib import Path


def convert_filepath(filepath, split_name):
    """
    Convert B3 filepath to LibriSpeech filepath format.
    
    Example:
    B3/data/libri_dev_trials_m_B3/wav/422-122949-0001.wav
    ->
    librispeech/dev-clean/422/122949/422-122949-0001.flac
    """
    # Extract speaker ID and utterance info from the filename
    filename = os.path.basename(filepath)
    match = re.match(r'(\d+)-(\d+)-(\d+)\.wav', filename)
    
    if not match:
        return filepath  # Return original if pattern doesn't match
    
    speaker_id, chapter_id, utterance_id = match.groups()
    
    # Determine the split (dev-clean, test-clean, train-clean-360)
    if 'dev' in split_name.lower():
        libri_split = 'dev-clean'
    elif 'test' in split_name.lower():
        libri_split = 'test-clean'
    else:
        libri_split = 'train-clean-360'
    
    # Construct the new path
    new_path = f"librispeech/{libri_split}/LibriSpeech/{libri_split}/{speaker_id}/{chapter_id}/{speaker_id}-{chapter_id}-{utterance_id}.flac"
    
    return new_path

def process_metadata_file(input_file, output_file):
    """Process a single metadata file."""
    print(f"Processing {input_file} -> {output_file}")
    
    # Read the CSV file
    df = pd.read_csv(input_file, sep='|')
        
    # Infer split name from the input file path
    if 'dev' in str(input_file).lower():
        split_name = 'dev-clean'
    elif 'test' in str(input_file).lower():
        split_name = 'test-clean'
    else:
        split_name = 'train-clean-360'
    
    print(f"Inferred split name: {split_name}")

    # Process all columns that might contain file paths
    path_columns = [col for col in df.columns if 'path' in col.lower() or 'file' in col.lower()]
    
    for col in path_columns:
        # Convert the paths using the split value from each row
        df[col] = df.apply(
            lambda row: convert_filepath(row[col],split_name) if isinstance(row[col], str) else row[col], 
            axis=1
        )

    # Add or modify the model column
    if 'model' in df.columns:
        df['model'] = 'original'

    SOURCE = 'split' if 'split' in df.columns else 'source'
    if SOURCE in df.columns:
        df[SOURCE] = df[SOURCE].apply(lambda x: x.replace("B3", "org"))

    # Save the modified dataframe
    df.to_csv(output_file, sep='|', index=False)
    print(f"Saved {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Convert B3 metadata to LibriSpeech format")
    parser.add_argument("--input_dir", required=True, help="Input directory with B3 metadata")
    parser.add_argument("--output_dir", required=True, help="Output directory for LibriSpeech metadata")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each CSV file in the input directory
    for file in input_dir.glob("*.csv"):
        # Skip temporary files
        if file.name.startswith('.'):
            continue
            
        output_file = output_dir / file.name
        process_metadata_file(file, output_file)
    
    print(f"Conversion complete. Output files saved to {output_dir}")

if __name__ == "__main__":
    main()
