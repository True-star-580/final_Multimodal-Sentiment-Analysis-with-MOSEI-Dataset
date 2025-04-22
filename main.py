# main.py

import os
import subprocess
import sys
from pathlib import Path

def print_menu():
    print("\n===== Multimodal Sentiment Analysis Console =====")
    print("1. Train a new model")
    print("2. Evaluate a trained model")
    print("3. Visualize training results")
    print("4. Exit")
    print("=================================================")

def train_model():
    print("\n-- Optional Training Parameters (press Enter to use defaults) --")
    epochs = input("Number of epochs (default: 20): ").strip()
    batch_size = input("Batch size (default: 32): ").strip()
    learning_rate = input("Learning rate (default: 1e-4): ").strip()
    hidden_dim = input("Hidden dimension size (default: 128): ").strip()
    num_heads = input("Number of attention heads (default: 4): ").strip()
    num_layers = input("Number of transformer layers (default: 2): ").strip()
    dropout = input("Dropout rate (default: 0.1): ").strip()
    modalities = input("Modalities (default: language,acoustic,visual): ").strip()
    log_dir = input("Log directory (default: logs/): ").strip()

    train_script = Path("scripts/train_multimodal.py")
    if not train_script.exists():
        print("train_multimodal.py NOT FOUND.")
        return

    cmd = [
        "python", str(train_script)
    ]

    # Add arguments if provided
    if epochs:
        cmd += ["--epochs", epochs]
    if batch_size:
        cmd += ["--batch_size", batch_size]
    if learning_rate:
        cmd += ["--learning_rate", learning_rate]
    if hidden_dim:
        cmd += ["--hidden_dim", hidden_dim]
    if num_heads:
        cmd += ["--num_heads", num_heads]
    if num_layers:
        cmd += ["--num_layers", num_layers]
    if dropout:
        cmd += ["--dropout", dropout]
    if modalities:
        cmd += ["--modalities", modalities]
    if log_dir:
        cmd += ["--log_dir", log_dir]

    print("\nLaunching training script with specified configuration...")
    subprocess.run(cmd)

def evaluate_model():
    checkpoint = input("Enter the path to the model checkpoint (.pt): ").strip()
    modalities = input("Enter modalities (default: language,acoustic,visual): ").strip() or "language,acoustic,visual"
    log_dir = input("Enter log directory (default: logs/): ").strip() or "logs/"

    eval_script = Path("scripts/evaluate.py")
    if not eval_script.exists():
        print("evaluate.py NOT FOUND.")
        return

    cmd = [
        "python", str(eval_script),
        "--checkpoint", checkpoint,
        "--modalities", modalities,
        "--log_dir", log_dir
    ]

    subprocess.run(cmd)

def visualize_results():
    log_dir = input("Enter log directory to visualize (default: logs/): ").strip() or "logs/"
    plot_path = Path(log_dir) / "multimodal_training_curves.png"
    scatter_path = Path(log_dir) / "test_predictions.png"

    if plot_path.exists():
        print(f"\nOpening training curves: {plot_path}")
        os.system(f"open {plot_path}" if sys.platform == "darwin" else f"xdg-open {plot_path}")
    else:
        print("Training curves plot NOT FOUND.")

    if scatter_path.exists():
        print(f"Opening prediction scatter plot: {scatter_path}")
        os.system(f"open {scatter_path}" if sys.platform == "darwin" else f"xdg-open {scatter_path}")
    else:
        print("Prediction scatter plot NOT FOUND.")

def main():
    while True:
        print_menu()
        choice = input("Enter your choice (1-4): ").strip()

        if choice == "1":
            train_model()
        elif choice == "2":
            evaluate_model()
        elif choice == "3":
            visualize_results()
        elif choice == "4":
            print("Exiting the console. Goodbye!")
            break
        else:
            print("Invalid choice. Please select 1-4.")

if __name__ == "__main__":
    main()