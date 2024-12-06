import os
import pandas as pd
from pathlib import Path
import glob
import argparse
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
        
    def get_wav_paths(self, base_dir: Path, subdir: Path) -> Dict[str, str]:
        wav_dir = subdir / 'wav'
        if not wav_dir.exists() or not wav_dir.is_dir():
            return {}
            
        wav_dict = {}
        for wav_file in wav_dir.glob('*'):
            if wav_file.is_file():
                utt_id = wav_file.stem
                wav_path = f"{base_dir.name}/{subdir.name}/wav/{wav_file.name}"
                wav_dict[utt_id] = wav_path
                
        return wav_dict

    def get_wav_paths(self, base_dir: Path, subdir: Path) -> Dict[str, str]:
        wav_dir = subdir / 'wav'
        if not wav_dir.exists() or not wav_dir.is_dir():
            return {}
            
        wav_dict = {}
        prefix = base_dir.parts[-2]  # Gets 'B3'
        
        for wav_file in wav_dir.glob('*'):
            if wav_file.is_file():
                utt_id = wav_file.stem
                wav_path = f"{prefix}/data/{subdir.name}/wav/{wav_file.name}"
                wav_dict[utt_id] = wav_path
                
        return wav_dict

    def process_subdir(self, base_dir: Path, subdir: Path) -> Optional[pd.DataFrame]:
        """Process a single Kaldi subdirectory.

            Args:
                base_dir: Base directory path
                subdir: Path to subdirectory containing Kaldi files

            Returns:
                DataFrame containing combined data, or None if no data found
        """
        print(f"Processing {subdir}...")
        
        utt2spk = self.read_kaldi_file(subdir / 'utt2spk')
        if not utt2spk:
            return None
            
        file_data = {
            'spk2gender': self.read_kaldi_file(subdir / 'spk2gender'),
            'text': self.read_kaldi_file(subdir / 'text'),
            'utt2dur': self.read_kaldi_file(subdir / 'utt2dur'),
            'utt2spk': utt2spk,
            'wav_paths': self.get_wav_paths(base_dir, subdir)
        }
        
        if not file_data['wav_paths']:
            return None
            
        combined_data = []
        for utt_id in file_data['wav_paths'].keys():
            spk_id = file_data['utt2spk'].get(utt_id, '')
            # Extract B3 from path like data/vpc2025_official/B3/data
            prefix = base_dir.parts[-2]
            
            entry = {
                'utterance_id': f"{prefix}_{utt_id}",
                'speaker_id': spk_id,
                'wav_path': file_data['wav_paths'].get(utt_id, ''),
                'gender': file_data['spk2gender'].get(spk_id, ''),
                'duration': float(file_data['utt2dur'].get(utt_id, '0')),
                'text': file_data['text'].get(utt_id, '')
            }
            combined_data.append(entry)
            
        return pd.DataFrame(combined_data) if combined_data else None
        
    def process_all(self):
        for base_dir in self.base_dirs:
            if not base_dir.exists():
                continue
                
            subdirs = [d for d in base_dir.glob("*") if d.is_dir()]
            print([str(d) for d in base_dir.glob("*") if d.is_dir()])
            
            for subdir in subdirs:
                df = self.process_subdir(base_dir, subdir)
                if df is not None:
                    output_file = subdir / 'combined_data.csv'
                    df.to_csv(output_file, index=False, sep='|')
                    print(f"Created {output_file}")
                else:
                    print(f"No data found in {subdir}")

def main():
    parser = argparse.ArgumentParser(description='Process Kaldi data directories.')
    parser.add_argument('dirs', nargs='+', help='Base directories to process')
    args = parser.parse_args()
    
    processor = KaldiDataProcessor(args.dirs)
    processor.process_all()

if __name__ == "__main__":
    main()