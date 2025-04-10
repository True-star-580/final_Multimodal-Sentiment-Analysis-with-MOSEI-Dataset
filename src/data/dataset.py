import os
import sys
import logging
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from config import (
    PROCESSED_DATA_DIR, DATASET_NAME, BATCH_SIZE,
    TEXT_EMBEDDING_DIM, AUDIO_FEATURE_SIZE, VISUAL_FEATURE_SIZE
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class MOSEIDataset(Dataset):
    def __init__(self, split="train", modalities=None):
        self.split = split
        self.modalities = modalities or ["language", "acoustic", "visual"]
        
        # Validate modalities
        for modality in self.modalities:
            if modality not in ["language", "acoustic", "visual"]:
                raise ValueError(f"Invalid modality: {modality}")
        
        # Load the data
        self.data_path = PROCESSED_DATA_DIR / DATASET_NAME / f"{split}_data.pkl"
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {self.data_path}. "
                f"Run preprocessing script first."
            )
        
        with open(self.data_path, "rb") as f:
            self.data = pickle.load(f)
        
        # Load metadata
        metadata_path = PROCESSED_DATA_DIR / DATASET_NAME / "metadata.pkl"
        if metadata_path.exists():
            with open(metadata_path, "rb") as f:
                self.metadata = pickle.load(f)
        else:
            logger.warning(f"Metadata file not found: {metadata_path}")
            # Fallback to default dimensions
            self.metadata = {
                "text_dim": TEXT_EMBEDDING_DIM,
                "audio_dim": AUDIO_FEATURE_SIZE,
                "visual_dim": VISUAL_FEATURE_SIZE
            }
        
        # Ensure labels are available
        if "labels" not in self.data:
            raise ValueError(f"Labels not found in data file: {self.data_path}")
        
        # Get number of samples
        self.num_samples = len(self.data["labels"])
        logger.info(f"Loaded {self.num_samples} samples for {split} split")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        sample = {}
        
        # Get features for each requested modality
        for modality in self.modalities:
            if modality in self.data:
                sample[modality] = torch.tensor(self.data[modality][idx], dtype=torch.float32)
            else:
                logger.warning(f"Modality {modality} not found in data")
                # Create empty tensor with proper dimensions
                dim = self.metadata.get(f"{modality}_dim", 0)
                sample[modality] = torch.zeros(dim, dtype=torch.float32)
        
        # Get label
        sample["label"] = torch.tensor(self.data["labels"][idx], dtype=torch.float32)
        
        return sample

class MOSEIUnimodalDataset(Dataset):
    def __init__(self, split="train", modality="text"):
        self.split = split
        self.modality = modality
        
        # Validate modality
        if modality not in ["language", "acoustic", "visual"]:
            raise ValueError(f"Invalid modality: {modality}")
        
        # Load the data
        self.data_path = PROCESSED_DATA_DIR / DATASET_NAME / f"{split}_data.pkl"
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {self.data_path}. "
                f"Run preprocessing script first."
            )
        
        with open(self.data_path, "rb") as f:
            self.data = pickle.load(f)
        
        # Ensure modality and labels are available
        if modality not in self.data:
            raise ValueError(f"Modality {modality} not found in data file: {self.data_path}")
        if "labels" not in self.data:
            raise ValueError(f"Labels not found in data file: {self.data_path}")
        
        # Get number of samples
        self.num_samples = len(self.data["labels"])
        logger.info(f"Loaded {self.num_samples} samples for {split} split, modality: {modality}")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        features = torch.tensor(self.data[self.modality][idx], dtype=torch.float32)
        label = torch.tensor(self.data["labels"][idx], dtype=torch.float32)
        
        return features, label

def get_dataloaders(modalities=None, batch_size=BATCH_SIZE, num_workers=2):
    dataloaders = {}
    
    for split in ["train", "val", "test"]:
        dataset = MOSEIDataset(split=split, modalities=modalities)
        
        shuffle = (split == "train")
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return dataloaders

def get_unimodal_dataloaders(modality, batch_size=BATCH_SIZE, num_workers=2):
    dataloaders = {}
    
    for split in ["train", "val", "test"]:
        dataset = MOSEIUnimodalDataset(split=split, modality=modality)
        
        shuffle = (split == "train")
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return dataloaders

if __name__ == "__main__":
    # Test the dataset
    dataset = MOSEIDataset(split="train")
    sample = dataset[0]
    
    # Print sample information
    print(f"Sample features:")
    for modality in ["language", "acoustic", "visual"]:
        if modality in sample:
            print(f"  {modality}: shape={sample[modality].shape}")
    
    print(f"Sample label: {sample['label']}")
    
    # Test dataloaders
    dataloaders = get_dataloaders(batch_size=32)
    batch = next(iter(dataloaders["train"]))
    
    print(f"\nBatch information:")
    for key, value in batch.items():
        print(f"  {key}: shape={value.shape}")