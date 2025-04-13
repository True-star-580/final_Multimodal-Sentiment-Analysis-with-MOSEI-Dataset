import os
from pathlib import Path
import torch

# Project paths
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
SRC_DIR = ROOT_DIR / "src"
DATA_SRC_DIR = SRC_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_SRC_DIR = SRC_DIR / "models"
UTILS_SRC_DIR = SRC_DIR / "utils"
LOGS_DIR = ROOT_DIR / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, SRC_DIR, DATA_SRC_DIR, PROCESSED_DATA_DIR,
                  MODELS_SRC_DIR, UTILS_SRC_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Dataset parameters
DATASET_NAME = "CMU_MOSEI"
DATASET_URL = "http://immortal.multicomp.cs.cmu.edu/CMU-MOSEI/"

# Feature extraction parameters
TEXT_MAX_LENGTH = 128
AUDIO_FEATURE_SIZE = 40  # MFCC features
VISUAL_FEATURE_SIZE = 35  # Facial landmarks

# Model parameters
TEXT_EMBEDDING_DIM = 768  # BERT embedding dimension

# Training parameters
SEED = 42
BATCH_SIZE = 32

# Device configuration
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu" 