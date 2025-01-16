import pandas as pd
from pathlib import Path
import argparse
from typing import List, Dict, Optional
import re
import os
from src.datamodules.components.common import get_dataset_class, LibriSpeechDefaults

DATASET_DEFAULTS = LibriSpeechDefaults()
DATASET_CLS, DF_COLS = get_dataset_class(DATASET_DEFAULTS.dataset_name)
SAVE_CSV_ARGS = {'index': False, 'sep': '|', 'filename': 'combined_data.csv'}
PREIX_ID = 'librispeech'


class KaldiDataProcessor:
    """Process Kaldi-style data directories and combine information into CSV files
    with standardized format matching librispeech-style metadata.
    """
    
    def __init__(self, base_dirs: List[str], speakers_file: str,
                 dataset_name: str = "vpc25", language: str = DATASET_DEFAULTS.language, 
                 sample_rate: float = DATASET_DEFAULTS.sample_rate,
                 speaker_id_prefix: str = PREIX_ID):
        """
        Initialize processor with list of base directories and dataset metadata.
        
        Args:
            base_dirs: List of directory paths (relative or absolute)
            speakers_file: Path to SPEAKERS.txt file containing speaker metadata
            dataset_name: Name of the dataset
            language: Language code (e.g., 'en')
            sample_rate: Audio sample rate in Hz
        """
        self.base_dirs = [Path(d) for d in base_dirs]
        self.dataset_name = dataset_name
        self.language = language
        self.sample_rate = sample_rate
        self.speaker_df, self.speaker_metadata = self.read_speakers_file(speakers_file)
        
    def read_speakers_file(self, speakers_file: str) -> Dict[str, Dict[str, str]]:
        """Same as before"""
        speaker_df = pd.read_csv(speakers_file, sep='|')
        speaker_df[DATASET_CLS.SPEAKER_ID] = speaker_df[DATASET_CLS.SPEAKER_ID].apply(lambda x: f"{PREIX_ID}_{x.split('_')[-1]}")
        speaker_data = speaker_df.set_index(DATASET_CLS.SPEAKER_ID).to_dict(orient='index')
        return speaker_df, speaker_data
        
    def read_kaldi_file(self, filepath: Path) -> Dict[str, str]:
        """Same as before"""
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
        """Same as before"""
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
        """Same as before"""
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
            spk_id = f"{PREIX_ID}" + '_' + file_data['utt2spk'].get(utt_id, '')
            speaker_info = self.speaker_metadata.get(spk_id, {})
            
            entry = {
                DATASET_CLS.DATASET: self.dataset_name,
                DATASET_CLS.LANGUAGE: self.language,
                DATASET_CLS.NATIONALITY: DATASET_DEFAULTS.country,
                DATASET_CLS.SR: self.sample_rate,
                DATASET_CLS.SPEAKER_ID: spk_id,
                DATASET_CLS.GENDER: speaker_info[DATASET_CLS.GENDER],
                DATASET_CLS.SPLIT: subdir.name,
                DATASET_CLS.REC_DURATION: float(file_data['utt2dur'].get(utt_id)),
                DATASET_CLS.REL_FILEPATH: file_data['wav_paths'].get(utt_id),
                DATASET_CLS.TEXT: file_data['text'].get(utt_id, '').upper(),
                DATASET_CLS.SPEAKER_NAME: speaker_info[DATASET_CLS.SPEAKER_NAME]
            }
            
            combined_data.append(entry)
            
        return pd.DataFrame(combined_data) if combined_data else None

    def cleanup_combined_data_files(self):
        """Delete all combined_data.csv files in the directory structure."""
        for base_dir in self.base_dirs:
            for csv_file in base_dir.rglob('combined_data.csv'):
                try:
                    csv_file.unlink()
                    print(f"Deleted: {csv_file}")
                except Exception as e:
                    print(f"Error deleting {csv_file}: {e}")

    def validate_directory_structure(self, subdirs: List[Path]) -> bool:
        """
        Validate if directory structure follows expected pattern.
        
        Args:
            subdirs: List of subdirectory paths
            
        Returns:
            bool: True if valid, False otherwise
        """
        expected_pattern = re.compile(r'(libri_(dev|test)_(enrolls|trials_[mf])|train-clean-360)_[A-Z0-9\-]+|metadata')
        
        for subdir in subdirs:
            if not expected_pattern.match(subdir.name):
                print(f"Warning: Directory structure does not match expected pattern: {subdir.name}")
                print("Expected patterns: libri_dev_enrolls_MODEL, libri_dev_trials_[mf]_MODEL, " 
                      "libri_test_enrolls_MODEL, libri_test_trials_[mf]_MODEL, or train-clean-360_MODEL")
                return False
        return True

    # def collect_enrollment_data(self, base_dir: Path, split: str) -> Optional[pd.DataFrame]:
    #     """Collect enrollment data from enrolls subdirectory.
        
    #     Args:
    #         base_dir: Base directory path
    #         split: Either 'dev' or 'test'
            
    #     Returns:
    #         DataFrame with enrollment data or None
    #     """
    #     enroll_pattern = f"libri_{split}_enrolls_*/enrolls"
    #     enroll_files = list(base_dir.glob(enroll_pattern))
    #     assert len(enroll_files) == 1, f"Multiple enrollment directories found in {base_dir}"
    #     enroll_file = enroll_files[0]
        
    #     if not enroll_files:
    #         print(f"WARNING: No enrollment data found in {base_dir}")
    #         return None
            
    #     enrollment_data = []
    #     model = str(enroll_file).split(os.sep)[-2].split('_')[-1]

    #     try:
    #         with open(enroll_file, 'r') as f:
    #             enrolls = f.read().splitlines()

    #         for enroll_line in enrolls:
    #             speaker_id = f"{PREIX_ID}_{enroll_line.split('-')[0]}"
    #             source = os.path.dirname(str(enroll_files[0])).split(os.sep)[-1]
    #             enrollment_data.append({'model': model, 'source': source, 
    #                                     'speaker_id': speaker_id, 'enrollments': enroll_line})    

    #     except Exception as e:
    #         print(f"Error reading {enroll_file}: {e}")

    #     return pd.DataFrame(enrollment_data) if enrollment_data else None
           
    # def collect_trials_data(self, base_dir: Path, split: str) -> Optional[pd.DataFrame]:
    #     """Collect trials data from trials file."""
    #     trial_pattern = f"libri_{split}_trials_[mf]_*"
    #     trial_dirs = [d for d in base_dir.glob(trial_pattern) 
    #                  if d.is_dir() and d.name != 'metadata']
        
    #     if not trial_dirs:
    #         return None
            
    #     trials_data = []
    #     for trial_dir in trial_dirs:
    #         model = trial_dir.name.split('_')[-1]
    #         gender = trial_dir.name.split('_')[-2]  # Extract 'm' or 'f'
    #         trials_file = trial_dir / 'trials'
            
    #         if not trials_file.exists() or not trials_file.is_file():
    #             print(f"Warning: Trials file not found at {trials_file}")
    #             continue
                
    #         try:
    #             with open(trials_file, 'r') as f:
    #                 for line in f:
    #                     enrollment_id, test_utt, label = line.strip().split()
    #                     trials_data.append({
    #                         'enrollment_id': enrollment_id,
    #                         'test_utt': test_utt,
    #                         'label': 1 if label == 'target' else 0,
    #                         'model': model,
    #                         'gender': 'male' if gender == 'm' else 'female'
    #                     })
    #         except Exception as e:
    #             print(f"Error processing trials file {trials_file}: {e}")
                    
    #     return pd.DataFrame(trials_data) if trials_data else None

    def collect_enrollment_data(self, base_dir: Path, split: str) -> Optional[pd.DataFrame]:
        """Collect enrollment data from enrolls subdirectory.
        
        Args:
            base_dir: Base directory path
            split: Either 'dev' or 'test'
            
        Returns:
            DataFrame with enrollment data or None
        """
        enroll_pattern = f"libri_{split}_enrolls_*/enrolls"
        enroll_files = list(base_dir.glob(enroll_pattern))
        assert len(enroll_files) == 1, f"Multiple enrollment directories found in {base_dir}"
        enroll_file = enroll_files[0]
        
        if not enroll_files:
            print(f"WARNING: No enrollment data found in {base_dir}")
            return None
            
        enrollment_data = []
        model = str(enroll_file).split(os.sep)[-2].split('_')[-1]
        enroll_dir = os.path.dirname(str(enroll_file))

        try:
            with open(enroll_file, 'r') as f:
                enrolls = f.read().splitlines()

            for enroll_line in enrolls:
                speaker_id = f"{PREIX_ID}_{enroll_line.split('-')[0]}"
                source = os.path.dirname(str(enroll_files[0])).split(os.sep)[-1]
                # Construct full relative path
                enrollment_path = f"{model}/data/{source}/wav/{enroll_line}.wav"
                enrollment_data.append({
                    'model': model, 
                    'source': source, 
                    'speaker_id': speaker_id, 
                    'enrollment_path': enrollment_path
                })    

        except Exception as e:
            print(f"Error reading {enroll_file}: {e}")

        return pd.DataFrame(enrollment_data) if enrollment_data else None
        
    def collect_trials_data(self, base_dir: Path, split: str) -> Optional[pd.DataFrame]:
        """Collect trials data from trials file."""
        trial_pattern = f"libri_{split}_trials_[mf]_*"
        trial_dirs = [d for d in base_dir.glob(trial_pattern) 
                    if d.is_dir() and d.name != 'metadata']
        
        if not trial_dirs:
            return None
            
        trials_data = []
        for trial_dir in trial_dirs:
            model = trial_dir.name.split('_')[-1]
            gender = trial_dir.name.split('_')[-2]  # Extract 'm' or 'f'
            trials_file = trial_dir / 'trials'
            source_dir = trial_dir.name
            
            if not trials_file.exists() or not trials_file.is_file():
                print(f"Warning: Trials file not found at {trials_file}")
                continue
                
            try:
                with open(trials_file, 'r') as f:
                    for line in f:
                        enrollment_id, test_utt, label = line.strip().split()
                        # Construct full relative path
                        test_path = f"{model}/data/{source_dir}/wav/{test_utt}.wav"
                        trials_data.append({
                            'enrollment_id': enrollment_id,
                            'test_path': test_path,
                            'label': 1 if label == 'target' else 0,
                            'model': model,
                            'gender': 'male' if gender == 'm' else 'female'
                        })
            except Exception as e:
                print(f"Error processing trials file {trials_file}: {e}")
                    
        return pd.DataFrame(trials_data) if trials_data else None

    def process_all(self):
        """Process all directories and save combined CSV files in train/dev/test splits."""
        for base_dir in self.base_dirs:
            if not base_dir.exists():
                continue

            subdirs = [d for d in base_dir.glob("*") if d.is_dir()]
            if not self.validate_directory_structure(subdirs):
                continue

            # Create metadata directory
            metadata_dir = base_dir / 'metadata'
            metadata_dir.mkdir(exist_ok=True)

            # Process enrollment and trials data
            dev_enrolls = self.collect_enrollment_data(base_dir, 'dev')
            test_enrolls = self.collect_enrollment_data(base_dir, 'test')
            dev_trials = self.collect_trials_data(base_dir, 'dev')
            test_trials = self.collect_trials_data(base_dir, 'test')
            
            if dev_enrolls is not None:
                dev_enrolls.to_csv(metadata_dir / 'dev_enrolls.csv', index=False, sep='|')
                print(f"Created {metadata_dir}/dev_enrolls.csv")

            if test_enrolls is not None:
                test_enrolls.to_csv(metadata_dir / 'test_enrolls.csv', index=False, sep='|')
                print(f"Created {metadata_dir}/test_enrolls.csv")

            if dev_trials is not None:
                dev_trials.to_csv(metadata_dir / 'dev_trials.csv', index=False, sep='|')
                print(f"Created {metadata_dir}/dev_trials.csv")
                
            if test_trials is not None:
                test_trials.to_csv(metadata_dir / 'test_trials.csv', index=False, sep='|')
                print(f"Created {metadata_dir}/test_trials.csv")


            # Initialize DataFrames for each split
            train_data = []
            dev_data = []
            test_data = []

            for subdir in subdirs:
                df = self.process_subdir(base_dir, subdir)
                if df is not None:
                    # Determine which split this belongs to based on directory name
                    if 'train-clean-360' in subdir.name:
                        train_data.append(df)
                    elif 'libri_dev' in subdir.name:
                        dev_data.append(df)
                    elif 'libri_test' in subdir.name:
                        test_data.append(df)

            # Combine and save splits
            if train_data:
                train_df = pd.concat(train_data, ignore_index=True)
                train_df.to_csv(metadata_dir / 'train.csv', index=False, sep='|')
                print(f"Created {metadata_dir}/train.csv")

            if dev_data:
                dev_df = pd.concat(dev_data, ignore_index=True)
                # Shuffle the combined dev data
                dev_df = dev_df.sample(frac=1, random_state=42).reset_index(drop=True)
                dev_df.to_csv(metadata_dir / 'dev.csv', index=False, sep='|')
                print(f"Created {metadata_dir}/dev.csv")

            if test_data:
                test_df = pd.concat(test_data, ignore_index=True)
                # Shuffle the combined test data
                test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
                test_df.to_csv(metadata_dir / 'test.csv', index=False, sep='|')
                print(f"Created {metadata_dir}/test.csv")


def main():
    parser = argparse.ArgumentParser(description='Process Kaldi data directories')
    parser.add_argument('base_dirs', nargs='+', help='Base directories containing Kaldi data')
    parser.add_argument('--speakers-file', 
                        default='data/librispeech/metadata/SPEAKERS.csv',
                        help='Path to SPEAKERS.csv file')
    parser.add_argument('--dataset-name', default='vpc25', help='Name of the dataset')
    parser.add_argument('--language', default='en', help='Language code')
    parser.add_argument('--sample-rate', type=float, default=16000.0, help='Audio sample rate in Hz')
    
    args = parser.parse_args()
    
    processor = KaldiDataProcessor(
        args.base_dirs,
        speakers_file=args.speakers_file,
        dataset_name=args.dataset_name,
        language=args.language,
        sample_rate=args.sample_rate
    )
    processor.process_all()

if __name__ == '__main__':
    main()