#main.py
import os
import sys
import logging
from pathlib import Path
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim

# Import global configurations
from config import (
    RAW_DATA_DIR, DATASET_NAME, DATASET_URL, PROCESSED_DATA_DIR, LOGS_DIR, MODELS_DIR,
    TEXT_MAX_LENGTH, AUDIO_FEATURE_SIZE, VISUAL_FEATURE_SIZE,
    TEXT_EMBEDDING_DIM, SEED, DEVICE, EARLY_STOPPING_PATIENCE, GRADIENT_CLIP_VAL,
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS, DROPOUT_RATE,
    HIDDEN_DIM, NUM_ATTENTION_HEADS, NUM_TRANSFORMER_LAYERS
)

# Import necessary modules
from src.data.download import install_mmsdk, download_mosei
from src.data.preprocess import MOSEIPreprocessor
from src.utils.logging import setup_logging
from src.data.dataset import get_dataloaders
from src.models.fusion import TransformerFusionModel
from src.training.trainer import Trainer
from src.utils.visualization import plot_training_curves, plot_scatter_predictions
from src.training.metrics import log_metrics, get_predictions

# Add project root to path to enable module imports
sys.path.append(str(Path(__file__).resolve().parents[2]))

# Set up basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def print_menu():
    """
    Display the main menu options for the console application.
    """
    print("\n===== Multimodal Sentiment Analysis Console =====")
    print("1. Download and preprocess dataset")
    print("2. Train a new model")
    print("3. Evaluate a trained model")
    print("4. Visualize training results")
    print("5. Exit")
    print("=================================================")

def download_and_preprocess():
    """
    Handle dataset download and preprocessing workflow.
    """
    logger.info("Starting dataset download and preprocessing...\n")
    RAW_DATA_DIR.mkdir(exist_ok=True, parents=True)
    install_mmsdk()
    success = download_mosei(RAW_DATA_DIR, DATASET_NAME, DATASET_URL)
    if not success:
        logger.error("Failed to download dataset.")
        return
    processed_dir = PROCESSED_DATA_DIR / DATASET_NAME
    if processed_dir.exists() and all((processed_dir / f"{split}_data.pkl").exists() for split in ["train", "val", "test"]):
        logger.info(f"Processed data already exists at {processed_dir}. Skipping preprocessing.")
        return
    preprocessor = MOSEIPreprocessor(RAW_DATA_DIR, PROCESSED_DATA_DIR)
    if preprocessor.process_dataset():
        preprocessor.save_processed_data()
        logger.info("Preprocessing completed successfully.")
    else:
        logger.error("Failed to preprocess dataset.")

def train_model():
    """
    Handle model training workflow with configurable parameters.
    """
    checkpoint = input("Enter the path to the model checkpoint (.pt) or leave blank to train from scratch: ").strip()
    modalities = input("Enter modalities (default: language,acoustic,visual): ").strip() or "language,acoustic,visual"
    batch_size = int(input("Enter batch size (default: 32): ").strip() or BATCH_SIZE)
    num_epochs = int(input("Enter number of epochs (default: 50): ").strip() or NUM_EPOCHS)
    hidden_dim = int(input("Enter hidden dimension (default: 256): ").strip() or HIDDEN_DIM)
    num_layers = int(input("Enter number of layers (default: 4): ").strip() or NUM_TRANSFORMER_LAYERS)
    num_heads = int(input("Enter number of heads (default: 8): ").strip() or NUM_ATTENTION_HEADS)
    dropout = float(input("Enter dropout rate (default: 0.3): ").strip() or DROPOUT_RATE)
    lr = float(input("Enter learning rate (default: 1e-4): ").strip() or LEARNING_RATE)
    weight_decay = float(input("Enter weight decay (default: 1e-5): ").strip() or WEIGHT_DECAY)
    early_stopping = int(input("Enter early stopping patience (default: 10): ").strip() or EARLY_STOPPING_PATIENCE)
    gradient_clip = float(input("Enter gradient clipping value (default: 1.0): ").strip() or GRADIENT_CLIP_VAL)
    seed = int(input("Enter random seed (default: 42): ").strip() or SEED)
    log_dir = input("Enter log directory (default: logs/): ").strip() or LOGS_DIR
    save_dir = input("Enter model save directory (default: models/): ").strip() or MODELS_DIR
    device = input("Enter device (cuda/mps/cpu/auto): ").strip()

    logger = setup_logging(log_dir)

    if not device:
        device = DEVICE
    else:
        device = device.lower()
    if device == "auto":
        device = DEVICE
    
    modalities = modalities.split(",")

    logger.info("All hyperparameters set.")
    logger.info(f"Using device: {device}")
    logger.info(f"Learning rate: {lr}")
    
    torch.manual_seed(seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    logger.info("\nLoading data...")
    dataloaders = get_dataloaders(modalities=modalities, batch_size=batch_size)

    logger.info("\nCreating model...")
    input_dims = {
        "language": TEXT_EMBEDDING_DIM,
        "acoustic": AUDIO_FEATURE_SIZE,
        "visual": VISUAL_FEATURE_SIZE
    }
    
    model = TransformerFusionModel(
        text_dim=input_dims["language"],
        audio_dim=input_dims["acoustic"],
        visual_dim=input_dims["visual"],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout_rate=dropout
    ).to(device)

    logger.info(f"Model architecture:\n{model}")

    start_epoch = 1
    if checkpoint:
        # ... (checkpoint loading logic)
        pass

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    trainer = Trainer(
        model=model,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        test_loader=dataloaders["test"],
        lr=lr,
        weight_decay=weight_decay,
        grad_clip_value=gradient_clip,
        device=device,
        log_dir=log_dir,
        model_dir=save_dir,
        experiment_name="multimodal_fusion",
    )

    logger.info("\nStarting training...")
    start_time = time.time()
    
    best_model, train_losses, val_losses, train_metrics, val_metrics = trainer.train(
        num_epochs=num_epochs,
        start_epoch=start_epoch,
        patience=early_stopping
    )
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")

    plot_path = Path(log_dir) / "multimodal_training_curves.png"
    plot_training_curves(train_losses, val_losses, save_path=plot_path)
    
    logger.info("Evaluating on test set...")
    test_metrics = trainer.test()
    if test_metrics:
        log_metrics(test_metrics, "test")

    predictions, targets = get_predictions(model=best_model, dataloader=dataloaders["test"], device=device)
    scatter_path = Path(log_dir) / "multimodal_predictions.png"
    plot_scatter_predictions(predictions, targets, save_path=scatter_path, title="Multimodal Sentiment Test Predictions vs Actual")
    
    logger.info("Training and evaluation completed!")

def evaluate_model():
    # ... (no changes needed)
    pass

def visualize_results():
    # ... (no changes needed)
    pass

def main():
    while True:
        print_menu()
        choice = input("Enter your choice (1-5): ").strip()

        if choice == "1":
            download_and_preprocess()
        elif choice == "2":
            train_model()
        elif choice == "3":
            evaluate_model()
        elif choice == "4":
            visualize_results()
        elif choice == "5":
            logger.info("Exiting the console. Goodbye!")
            break
        else:
            logger.error("Invalid choice. Please select 1-5.")
            continue

if __name__ == "__main__":
    main()