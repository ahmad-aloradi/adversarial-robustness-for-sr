from pathlib import Path
from typing import List, Optional, Dict, Union
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import sys
sys.path.append(f"/home/aloradi/adversarial-robustness-for-sr")
from src.datamodules.components.vpc25.vpc_dataset import AnonymizedLibriSpeechDataset, VPC25PaddingCollate
import re


class AnonymizedLibriSpeechDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for anonymized LibriSpeech-like data."""

    def __init__(
        self,
        root_dir: Union[str, Path],
        subset_dirs: List[str] = ['b2_system', 'b5_b6_systems'],
        batch_size: int = 32,
        num_workers: int = 4,
        transform=None,
    ):
        """
        Initialize the DataModule.
        
        Args:
            root_dir: Root directory containing all data
            subset_dirs: List of subset directories to include
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            transform: Optional transform to be applied to audio
        """
        super().__init__()
        self.root_dir = Path(root_dir)
        self.subset_dirs = subset_dirs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        
    def setup(self, stage: Optional[str] = None):
        """Set up the datasets for each stage (fit, test, predict)."""
        if stage == 'fit' or stage is None:
            # Create training and validation datasets
            self.train_dataset = AnonymizedLibriSpeechDataset(
                root_dir=self.root_dir,
                subset_dirs=self.subset_dirs,
                transform=self.transform,
                split='train-clean-360'
            )
            self.eval_dataset = AnonymizedLibriSpeechDataset(
                root_dir=self.root_dir,
                subset_dirs=self.subset_dirs,
                transform=self.transform,
                split='dev_enrolls'
            )
        if stage == 'test' or stage is None:
            # Create test dataset
            self.test_dataset = AnonymizedLibriSpeechDataset(
                root_dir=self.root_dir,
                subset_dirs=self.subset_dirs,
                transform=self.transform,
                split='test_enrolls'
            )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=VPC25PaddingCollate()
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.eval_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=VPC25PaddingCollate()
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=VPC25PaddingCollate()
        )


if __name__ == "__main__":

    # Initialize DataModule
    root_dir = "data/vpc25_data/data"

    # Get all subdirectories in root_dir
    root_path = Path(root_dir)
    all_dirs = [d.name for d in root_path.iterdir() if d.is_dir()]
    
    # Filter directories matching pattern b3_, b4_, b5_
    pattern = re.compile(r'b[2-6]_.*')
    
    datamodule = AnonymizedLibriSpeechDataModule(
        root_dir=root_dir,
        subset_dirs=[d for d in all_dirs if pattern.match(d)],
        batch_size=32,
        num_workers=0,
        transform=None,
    )
    
    # Set up the datamodule
    datamodule.setup()
    
    # Get a batch from the training dataloader
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()
    batch = next(iter(train_loader))
    
    # Print batch information
    print("Batch keys:", batch.keys())
    print("Audio shape:", batch['audio'].shape)
    print("Number of samples:", len(batch['utterance_id']))