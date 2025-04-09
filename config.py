import os
from pathlib import Path

# Project paths
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
SRC_DIR = ROOT_DIR / "src"
DATA_SRC_DIR = SRC_DIR / "data"

# Dataset parameters
DATASET_NAME = "CMU_MOSEI"
DATASET_URL = "http://immortal.multicomp.cs.cmu.edu/CMU-MOSEI/"
