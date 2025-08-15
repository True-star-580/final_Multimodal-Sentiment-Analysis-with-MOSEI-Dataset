# src/data/dataset_v2.py
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path

class FinetuneDataset(Dataset):
    def __init__(self, split="train"):
        data_path = Path("./processed_data_v2/finetuning_data.csv")
        if not data_path.exists():
            raise FileNotFoundError(f"'{data_path}' not found. Please run preprocess_v2.py first.")
        
        df = pd.read_csv(data_path)
        self.df = df[df['split'] == split].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "text": row['text'],
            "label": torch.tensor(row['label'], dtype=torch.long)
        }

def collate_fn(batch, tokenizer):
    texts = [item['text'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    return inputs, labels