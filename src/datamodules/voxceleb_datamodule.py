import os
from typing import Dict, List, Optional
import torch
from dataclasses import dataclass

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.datamodules.components.voxceleb.voxceleb_dataset import (
    VoxCelebDataset, 
    VoxCelebVerificationDataset, 
    TrainCollate, 
    VerificationCollate)


@dataclass
class VoxCelebConfig:
    data_dir: str = "/path/to/voxceleb"  # Change this to your path
    veri_test_path: str = "/path/to/veri_test.txt"  # Change this to your path
    batch_size: int = 32
    num_workers: int = 4
    sample_rate: int = 16000
    max_duration: float = 8.0  # maximum duration in seconds
    train_list: str = "train_list.csv"


class VoxCelebDataModule(LightningDataModule):
    def __init__(self, cfg: VoxCelebConfig):
        super().__init__()
        self.cfg = cfg
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def prepare_data(self):
        pass
        
    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_data = VoxCelebDataset(
                self.cfg.data_dir,
                self.cfg.train_metadata,
                self.cfg.sample_rate,
                self.cfg.max_duration
            )
            self.val_data = VoxCelebDataset(
                self.cfg.data_dir,
                self.cfg.val_metadata,
                self.cfg.sample_rate,
                self.cfg.max_duration
            )
        
        if stage == 'test' or stage is None:
            self.test_data = VoxCelebVerificationDataset(
                self.cfg.data_dir,
                self.cfg.veri_test_path,
                self.cfg.sample_rate,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=TrainCollate()
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=TrainCollate()
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=VerificationCollate()
        )


#-----
# Test

def test_datamodule():
    print("Starting VoxCeleb DataModule test...")
    
    # Initialize config
    cfg = VoxCelebConfig(
        data_dir="adversarial-robustness-for-sr/data/voxceleb/wav",  # Update with your path
        veri_test_path="adversarial-robustness-for-sr/data/voxceleb/meta/veri_test.txt"  # Update with your path
    )
    
    try:
        # Initialize datamodule
        print("Initializing DataModule...")
        datamodule = VoxCelebDataModule(cfg)
        
        # Prepare data (this loads metadata)
        print("Preparing data...")
        datamodule.prepare_data()
        
        # Setup train/val/test splits
        print("Setting up splits...")
        datamodule.setup()
        
        # Test train dataloader
        print("Testing train dataloader...")
        train_loader = datamodule.train_dataloader()
        batch = next(iter(train_loader))
        print("\nTrain batch structure:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"{key}: type={type(value)}")
        
        # Test validation dataloader
        print("\nTesting validation dataloader...")
        val_loader = datamodule.val_dataloader()
        batch = next(iter(val_loader))
        print("\nValidation batch structure:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"{key}: type={type(value)}")
        
        # Test verification dataloader
        print("\nTesting verification dataloader...")
        test_loader = datamodule.test_dataloader()
        batch = next(iter(test_loader))
        print("\nTest batch structure:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"{key}: type={type(value)}")
        
        print("\nAll tests passed successfully!")
        
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        raise

if __name__ == "__main__":
    test_datamodule()