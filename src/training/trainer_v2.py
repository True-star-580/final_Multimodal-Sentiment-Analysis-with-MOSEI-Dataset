# src/training/trainer_v2.py
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

class TrainerV2:
    def __init__(self, model, optimizer, criterion, device, train_loader, val_loader):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for inputs, labels in tqdm(self.train_loader, desc="Training"):
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(**inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc="Validating"):
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                labels = labels.to(self.device)
                
                outputs = self.model(**inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        return avg_loss, accuracy, f1

    def train(self, num_epochs):
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch()
            val_loss, val_acc, val_f1 = self.validate()
            print(f"Epoch {epoch}/{num_epochs} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")