from pathlib import Path
from typing import List, Dict, Optional, Union
from tqdm import tqdm
import soundfile as sf
import pandas as pd
import wget
from dataclasses import dataclass
import hashlib


@dataclass
class VoxCelebUtterance:
    utterance_id: str      # Unique identifier for utterance
    speaker_id: str        # ID of the speaker
    path: str             # Path relative to dataset root
    dataset: str          # 'vox1' or 'vox2'
    duration: float       # Duration in seconds
    gender: Optional[str] = None
    language: Optional[str] = None
    nationality: Optional[str] = None
    age: Optional[str] = None


class VoxCelebProcessor:
    """Process VoxCeleb 1 & 2 datasets and generate metadata"""
    
    # URLs for metadata files
    METADATA_URLS = {
        'vox1': {
            'veri_test': 'https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test.txt',
            'meta': 'https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/vox1_meta.csv'
        },
        'vox2': {
            'meta': 'https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/vox2_meta.csv'
        }
    }
    
    def __init__(self, root_dir: Union[str, Path]):
        self.root_dir = Path(root_dir)
        self.metadata_dir = self.root_dir / 'metadata'
        self.metadata_dir.mkdir(exist_ok=True)
        
        # Download metadata files if they don't exist
        self._ensure_metadata_files()
        
        # Load metadata and verification files
        self.speaker_metadata = self._load_speaker_metadata()
        self.test_files = self._load_test_files()

    def _download_file(self, url: str, target_path: Path) -> None:
        """Download a file from URL to target path"""
        try:
            print(f"Downloading {url} to {target_path}")
            wget.download(url, str(target_path))
            print()  # New line after wget progress bar
        except Exception as e:
            raise RuntimeError(f"Failed to download {url}: {e}")

    def _ensure_metadata_files(self) -> None:
        """Ensure all required metadata files exist"""
        for dataset, files in self.METADATA_URLS.items():
            for file_type, url in files.items():
                filename = f"{dataset}_{file_type}.txt" if file_type == 'veri_test' else f"{dataset}_meta.csv"
                target_path = self.metadata_dir / filename
                
                if not target_path.exists():
                    self._download_file(url, target_path)

    def _load_speaker_metadata(self) -> Dict:
        """Load and merge speaker metadata from both VoxCeleb1 and VoxCeleb2"""
        metadata = {}
        
        for dataset in ['vox1', 'vox2']:
            meta_path = self.metadata_dir / f"{dataset}_meta.csv"
            if meta_path.exists():
                df = pd.read_csv(meta_path)
                # Process each dataset's metadata and merge
                for _, row in df.iterrows():
                    speaker_id = row['VoxCeleb_ID'] if 'VoxCeleb_ID' in df.columns else row['VGGFace1_ID']
                    metadata[speaker_id] = {
                        'gender': row.get('Gender'),
                        'nationality': row.get('Nationality'),
                        'language': row.get('Language', row.get('Nationality')),  # Use nationality as proxy if language missing
                        'age': row.get('Age')
                    }
        
        return metadata

    def _load_test_files(self) -> set:
        """Load verification test files to exclude"""
        test_files = set()
        veri_test_path = self.metadata_dir / "vox1_veri_test.txt"
        
        assert veri_test_path.exists(), f"Verification test file not found: {veri_test_path}"
        with open(veri_test_path, 'r') as f:
            for line in f:
                _, path1, path2 = line.strip().split()
                test_files.add(path1)
                test_files.add(path2)
        
        return test_files

    def _determine_dataset(self, path: Path) -> str:
        """Determine if path belongs to VoxCeleb1 or VoxCeleb2"""
        # This might need adjustment based on your exact directory structure
        parts = path.parts
        if 'voxceleb1' in parts or 'vox1' in parts:
            return 'vox1'
        elif 'voxceleb2' in parts or 'vox2' in parts:
            return 'vox2'
        else:
            # Try to determine from speaker ID format or other means
            speaker_id = parts[0]  # Assuming first part is speaker ID
            return 'vox1' if speaker_id.startswith('id1') else 'vox2'

    def _generate_utterance_id(self, wav_path: Path, speaker_id: str, dataset: str) -> str:
        """Generate unique utterance ID"""
        # Create a unique ID based on path, speaker, and dataset
        path_str = str(wav_path.relative_to(self.root_dir))
        unique_str = f"{dataset}_{speaker_id}_{path_str}"
        # Create a short hash for uniqueness
        hash_str = hashlib.md5(unique_str.encode()).hexdigest()[:8]
        return f"{dataset}_{speaker_id}_{hash_str}"

    def generate_metadata(self, min_duration: float) -> List[VoxCelebUtterance]:
        """Generate metadata for all valid utterances"""
        utterances = []
        
        print("Scanning for audio files...")
        for wav_path in tqdm(list(self.root_dir.rglob("*.wav"))):
            try:
                # Get relative path
                rel_path = wav_path.relative_to(self.root_dir)
                rel_path_str = str(rel_path)
                
                # Skip if in test files
                if rel_path_str in self.test_files:
                    continue
                
                # Get duration and skip if too short
                info = sf.info(wav_path)
                if info.duration < min_duration:
                    continue
                
                # Determine dataset and speaker ID
                dataset = self._determine_dataset(rel_path)
                speaker_id = rel_path.parts[0]
                
                # Get speaker metadata
                speaker_info = self.speaker_metadata.get(speaker_id, {})
                
                # Create utterance object
                utterance = VoxCelebUtterance(
                    utterance_id=self._generate_utterance_id(wav_path, speaker_id, dataset),
                    speaker_id=speaker_id,
                    path=rel_path_str,
                    dataset=dataset,
                    duration=info.duration,
                    gender=speaker_info.get('gender'),
                    language=speaker_info.get('language'),
                    nationality=speaker_info.get('nationality'),
                    age=speaker_info.get('age')
                )
                
                utterances.append(utterance)
                
            except Exception as e:
                print(f"Error processing {wav_path}: {e}")
        
        return utterances


def write_metadata(self, utterances: List[VoxCelebUtterance], output_file: str):
    """
    Write metadata to CSV file with all VoxCelebUtterance fields as columns
    
    Args:
        utterances: List of VoxCelebUtterance objects
        output_file: Output CSV file path
    """
    # Convert list of utterances to pandas DataFrame
    df = pd.DataFrame([vars(utt) for utt in utterances])
    
    # Ensure consistent column ordering based on VoxCelebUtterance fields
    columns = [
        'utterance_id',
        'speaker_id',
        'path',
        'dataset',
        'duration',
        'gender',
        'language',
        'nationality',
        'age'
    ]
    
    # Reorder columns and write to CSV
    df = df[columns]
    
    # Handle output path
    output_path = Path(output_file)
    if not output_path.suffix == '.csv':
        output_path = output_path.with_suffix('.csv')
    
    # Write CSV file
    df.to_csv(output_path, index=False, na_rep='NA')
    
    print(f"\nMetadata written to {output_path}")
    print(f"CSV columns: {', '.join(columns)}")
    print("\nFirst few rows:")
    print(df.head().to_string())
    
    # Print some basic statistics
    print("\nDataset statistics:")
    print(f"Total utterances: {len(df)}")
    print("\nUtterances per dataset:")
    print(df['dataset'].value_counts())
    print("\nGender distribution:")
    print(df['gender'].value_counts(dropna=False))
    if 'language' in df.columns:
        print("\nTop 5 languages:")
        print(df['language'].value_counts().head())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate VoxCeleb metadata")
    parser.add_argument("--root_dir", type=str, required=True, 
                      help="Root directory containing both VoxCeleb1 and VoxCeleb2")
    parser.add_argument("--output_file", type=str, default="voxceleb_train.txt",
                      help="Output metadata file")
    parser.add_argument("--min_duration", type=float, default=0.5,
                      help="Minimum duration in seconds")
    
    args = parser.parse_args()
    
    processor = VoxCelebProcessor(args.root_dir)
    utterances = processor.generate_metadata(args.min_duration, min_duration=args.min_duration)
    processor.write_metadata(utterances, args.output_file)
    
    # Print statistics
    total_vox1 = sum(1 for u in utterances if u.dataset == 'vox1')
    total_vox2 = sum(1 for u in utterances if u.dataset == 'vox2')
    print(f"\nTotal utterances: {len(utterances)}")
    print(f"VoxCeleb1 utterances: {total_vox1}")
    print(f"VoxCeleb2 utterances: {total_vox2}")
    print(f"Unique speakers: {len({u.speaker_id for u in utterances})}")
