import os
import pandas as pd
from pathlib import Path
import fileinput
import re
from typing import List, Dict, Optional, Union

class KaldiDataProcessor:
    """Fixes the structure of the data copied from Audiolabs repos to match our repo's style.
    Example:
    python src/datamodules/components/vpc25/modify_scp.py data/vpc25_data/data/b3_sttts/train-clean-360_sttts/wav.scp

    """
    
    def __init__(self, wav_scp_paths: Union[str, List[str]]):
        """
        Initialize processor with list of wav.scp paths.
        
        Args:
            wav_scp_paths: Path or list of paths to wav.scp files
        """
        if isinstance(wav_scp_paths, str):
            wav_scp_paths = [wav_scp_paths]
            
        self.wav_scp_paths = [Path(p) for p in wav_scp_paths]
        
    def update_wav_scp(self, wav_scp_path: Path) -> None:
        """Update wav.scp file to use relative paths."""
        if not wav_scp_path.exists():
            return
            
        pattern = r'/home/leschaaa@alabsad\.fau\.de/Voice-Privacy-Challenge-2024/(.*\.wav)'
        
        with fileinput.input(files=[str(wav_scp_path)], inplace=True) as f:
            for line in f:
                if line.strip():
                    updated_line = re.sub(pattern, r'\1', line)
                    print(updated_line, end='')
        
    def read_kaldi_file(self, filepath: Path) -> Dict[str, str]:
        """Read space-separated Kaldi file into dictionary."""
        if not filepath.exists():
            return {}
            
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        return {
            parts[0]: parts[1] if len(parts) > 1 else parts[0]
            for line in lines 
            if line.strip() and (parts := line.strip().split(maxsplit=1))
        }
        
    def read_wav_scp(self, wav_scp_path: Path) -> Dict[str, str]:
        """Read and update wav.scp file, returning dictionary of paths."""
        if not wav_scp_path.exists():
            return {}
        
        # First update the wav.scp file to use relative paths
        self.update_wav_scp(wav_scp_path)
            
        wav_dict = {}
        with open(wav_scp_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    utt_id, wav_path = line.strip().split(maxsplit=1)
                    wav_dict[utt_id] = wav_path
            
        return wav_dict
        
    def process_directory(self, wav_scp_path: Path) -> Optional[pd.DataFrame]:
        """Process a directory containing wav.scp and related files."""
        dir_path = wav_scp_path.parent
        print(f"Processing directory: {dir_path}")
        
        # Read all files preserving original IDs
        utt2spk = self.read_kaldi_file(dir_path / 'utt2spk')
        
        if not utt2spk:
            print(f"No utt2spk found in {dir_path}")
            return None
            
        file_data = {
            'spk2gender': self.read_kaldi_file(dir_path / 'spk2gender'),
            'text': self.read_kaldi_file(dir_path / 'text'),
            'utt2dur': self.read_kaldi_file(dir_path / 'utt2dur'),
            'utt2spk': utt2spk,
            'wav_scp': self.read_wav_scp(wav_scp_path)
        }
        
        if not file_data['wav_scp']:
            print(f"No wav.scp data found in {dir_path}")
            return None
            
        # Combine all information
        combined_data = []
        for utt_id in file_data['wav_scp'].keys():
            spk_id = file_data['utt2spk'].get(utt_id, '')
            
            entry = {
                'utterance_id': utt_id,
                'speaker_id': spk_id,
                'gender': file_data['spk2gender'].get(spk_id, ''),
                'text': file_data['text'].get(utt_id, ''),
                'duration': float(file_data['utt2dur'].get(utt_id, '0')),
                'wav_path': file_data['wav_scp'].get(utt_id, '')
            }
            combined_data.append(entry)
            
        return pd.DataFrame(combined_data) if combined_data else None
        
    def process_all(self):
        """Process all specified wav.scp files."""
        for wav_scp_path in self.wav_scp_paths:
            if not wav_scp_path.exists():
                print(f"Warning: {wav_scp_path} does not exist")
                continue
                
            df = self.process_directory(wav_scp_path)
            
            if df is not None:
                output_file = wav_scp_path.parent / 'combined_data.csv'
                df.to_csv(output_file, index=False, sep='|')
                print(f"Created {output_file}")

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process Kaldi data directories.')
    parser.add_argument('wav_scp_paths', nargs='+', help='Paths to wav.scp files to process')
    args = parser.parse_args()
    
    processor = KaldiDataProcessor(args.wav_scp_paths)
    processor.process_all()

if __name__ == "__main__":
    main()