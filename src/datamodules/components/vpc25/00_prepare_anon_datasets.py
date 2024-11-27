import os
import pandas as pd
from pathlib import Path
import glob
from typing import List, Dict, Optional

class KaldiDataProcessor:
    """Process Kaldi-style data directories and combine information into CSV files.
    Example:
    python src/datamodules/components/vpc25/prepare_anon_datasets.py data/vpc25_data/data/b5_asrbn_hifigan_bn_tdnnf_600h
    """
    
    def __init__(self, base_dirs: List[str]):
        """
        Initialize processor with list of base directories.
        
        Args:
            base_dirs: List of directory paths (relative or absolute)
        """
        self.base_dirs = [Path(d) for d in base_dirs]
        
    def read_kaldi_file(self, filepath: Path) -> Dict[str, str]:
        """
        Read space-separated Kaldi file into dictionary.
        
        Args:
            filepath: Path to Kaldi format file
            
        Returns:
            Dictionary mapping first column to rest of line
        """
        if not filepath.exists():
            return {}
            
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        return {
            parts[0]: parts[1] if len(parts) > 1 else parts[0]
            for line in lines 
            if line.strip() and (parts := line.strip().split(maxsplit=1))
        }
        
    def update_wav_paths(self, wav_scp_path: Path, base_dir: Path, subdir: Path) -> Dict[str, str]:
        """
        Update paths in wav.scp to match relative structure.
        
        Args:
            wav_scp_path: Path to wav.scp file
            base_dir: Base directory path
            subdir: Current subdirectory being processed
            
        Returns:
            Dictionary mapping utterance IDs to updated wav paths
        """
        if not wav_scp_path.exists():
            return {}
            
        wav_dict = {}
        with open(wav_scp_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line in lines:
            if not line.strip():
                continue
                
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
                
            utt_id, wav_path = parts
            wav_filename = Path(wav_path).name
            new_wav_path = f"{base_dir.name}/{subdir.name}/wav/{wav_filename}"
            wav_dict[utt_id] = new_wav_path
            
        return wav_dict
        
    def process_subdir(self, base_dir: Path, subdir: Path) -> Optional[pd.DataFrame]:
        """
        Process a single Kaldi subdirectory.
        
        Args:
            base_dir: Base directory path
            subdir: Path to subdirectory containing Kaldi files
            
        Returns:
            DataFrame containing combined data, or None if no data found
        """
        print(f"Processing {subdir}...")
        
        # Read all files preserving original IDs
        utt2spk = self.read_kaldi_file(subdir / 'utt2spk')
        
        if not utt2spk:  # If no utt2spk file, skip this directory
            return None
            
        file_data = {
            'spk2gender': self.read_kaldi_file(subdir / 'spk2gender'),
            'text': self.read_kaldi_file(subdir / 'text'),
            'utt2dur': self.read_kaldi_file(subdir / 'utt2dur'),
            'utt2spk': utt2spk,
            'wav_scp': self.update_wav_paths(subdir / 'wav.scp', base_dir, subdir)
        }
        
        # Return None if no wav_scp data
        if not file_data['wav_scp']:
            return None
            
        # Combine all information preserving original IDs
        combined_data = []
        for utt_id in file_data['wav_scp'].keys():
            spk_id = file_data['utt2spk'].get(utt_id, '')
            
            entry = {
                'utterance_id': utt_id,  # Original utterance ID from files
                'speaker_id': spk_id,    # Original speaker ID from utt2spk
                'gender': file_data['spk2gender'].get(spk_id, ''),
                'text': file_data['text'].get(utt_id, ''),
                'duration': float(file_data['utt2dur'].get(utt_id, '0')),
                'wav_path': file_data['wav_scp'].get(utt_id, '')
            }
            combined_data.append(entry)
            
        return pd.DataFrame(combined_data) if combined_data else None
        
    def process_all(self):
        """Process all base directories and save combined data to CSV files."""
        for base_dir in self.base_dirs:
            if not base_dir.exists():
                print(f"Warning: Directory {base_dir} does not exist")
                continue
                
            # Find all subdirectories
            subdirs = [d for d in base_dir.glob("*") if d.is_dir()]
            
            for subdir in subdirs:
                df = self.process_subdir(base_dir, subdir)
                
                if df is not None:
                    output_file = subdir / 'combined_data.csv'
                    df.to_csv(output_file, index=False, sep='|')
                    print(f"Created {output_file}")
                else:
                    print(f"No data found in {subdir}")

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process Kaldi data directories.')
    parser.add_argument('dirs', nargs='+', help='Base directories to process')
    args = parser.parse_args()
    
    processor = KaldiDataProcessor(args.dirs)
    processor.process_all()


if __name__ == "__main__":
    main()