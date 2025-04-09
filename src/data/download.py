import os
import sys
import argparse
import logging
from pathlib import Path
import subprocess

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from config import RAW_DATA_DIR, DATASET_NAME

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def install_mmsdk():
    # Check if mmsdk is already installed
    # If not, install it using pip
    try:
        import mmsdk
        logger.info("CMU-MultimodalSDK is already installed.")
    except ImportError:
        logger.info("Installing CMU-MultimodalSDK...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "git+https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.git"
            ])
            logger.info("CMU-MultimodalSDK installed successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install CMU-MultimodalSDK: {e}")
            sys.exit(1)

def download_mosei():
    try:
        from mmsdk import mmdatasdk as md
        
        # Create dataset directory
        dataset_path = RAW_DATA_DIR / DATASET_NAME
        dataset_path.mkdir(exist_ok=True, parents=True)
        
        # Define the computational sequences to download
        mosei_highlevel = {
            'acoustic': 'http://immortal.multicomp.cs.cmu.edu/CMU-MOSEI/acoustic/CMU_MOSEI_COVAREP.csd',
            'visual': 'http://immortal.multicomp.cs.cmu.edu/CMU-MOSEI/visual/CMU_MOSEI_VisualOpenFace2.csd',
            'language': 'http://immortal.multicomp.cs.cmu.edu/CMU-MOSEI/language/CMU_MOSEI_TimestampedWords.csd',
            'labels': 'http://immortal.multicomp.cs.cmu.edu/CMU-MOSEI/labels/CMU_MOSEI_Labels.csd'
        }

        logger.info("Downloading MOSEI high-level features...")
        mosei_dataset = md.mmdataset(mosei_highlevel, str(dataset_path))
        
        logger.info("Adding labels...")
        mosei_dataset.add_computational_sequences(
            {'Sentiment Labels': mosei_highlevel['labels']},
            str(dataset_path)
        )
        
        logger.info("Aligning to sentiment labels...")
        mosei_dataset.align('Sentiment Labels')
        
        logger.info(f"Dataset successfully downloaded and aligned to {dataset_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading MOSEI dataset: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download CMU-MOSEI dataset")
    parser.add_argument("--force", action="store_true", help="Force re-download even if data exists")
    args = parser.parse_args()
    
    # Create data directory if it doesn't exist
    RAW_DATA_DIR.mkdir(exist_ok=True, parents=True)
    
    # Install the MultimodalSDK if needed
    install_mmsdk()
    
    # Download the dataset
    success = download_mosei()
    if success:
        logger.info("Dataset download completed successfully.")
    else:
        logger.error("Failed to download dataset.")
        sys.exit(1)

if __name__ == "__main__":
    main()