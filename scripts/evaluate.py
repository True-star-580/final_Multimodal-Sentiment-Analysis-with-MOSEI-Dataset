import os
import sys
import argparse
from pathlib import Path

import torch

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import (
    DEVICE, TEXT_EMBEDDING_DIM, AUDIO_FEATURE_SIZE, VISUAL_FEATURE_SIZE,
    HIDDEN_DIM, NUM_ATTENTION_HEADS, NUM_TRANSFORMER_LAYERS, DROPOUT_RATE,
    BATCH_SIZE
)
from src.data.dataset import get_dataloaders
from src.models.fusion import TransformerFusionModel
from src.training.metrics import log_metrics, get_predictions
from src.utils.logging import setup_logging
from src.utils.visualization import plot_scatter_predictions

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained multimodal model on the test set")
    
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True, 
        help="Path to model checkpoint"
    )
    
    parser.add_argument(
        "--modalities",
        type=str,
        default="language,acoustic,visual", 
        help="Comma-separated list of modalities to use"
    )

    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=BATCH_SIZE, 
        help="Batch size for evaluation"
    )

    parser.add_argument(
        "--log_dir", 
        type=str, 
        default="logs/", 
        help="Directory to save evaluation logs and plots"
    )

    return parser.parse_args()

def main():
    args = parse_args()
    logger = setup_logging(args.log_dir)
    
    device = DEVICE
    modalities = args.modalities.split(",")
    logger.info(f"Evaluating model with modalities: {modalities}")

    # Load test dataloader
    dataloaders = get_dataloaders(
        modalities=modalities,
        batch_size=args.batch_size
    )
    test_loader = dataloaders["test"]
    
    # Define input dimensions
    input_dims = {
        "language": TEXT_EMBEDDING_DIM,
        "acoustic": AUDIO_FEATURE_SIZE,
        "visual": VISUAL_FEATURE_SIZE
    }

    # Initialize model
    model = TransformerFusionModel(
        text_dim=input_dims["language"],
        audio_dim=input_dims["acoustic"],
        visual_dim=input_dims["visual"],
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_TRANSFORMER_LAYERS,
        num_heads=NUM_ATTENTION_HEADS,
        dropout_rate=DROPOUT_RATE
    )
    model = model.to(device)

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return

    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Run evaluation
    logger.info("Running evaluation on test set...")
    predictions, targets = get_predictions(model=model, dataloader=test_loader, device=device)

    # Compute and log metrics
    from src.training.metrics import compute_metrics
    test_metrics = compute_metrics(predictions, targets)
    log_metrics(test_metrics, split="test")

    # Plot scatter predictions
    scatter_path = Path(args.log_dir) / "test_predictions.png"
    plot_scatter_predictions(predictions, targets, save_path=scatter_path, title="Test Predictions vs Actual")
    logger.info(f"Saved prediction scatter plot to {scatter_path}")
    
    logger.info("Evaluation complete!")

if __name__ == "__main__":
    main()