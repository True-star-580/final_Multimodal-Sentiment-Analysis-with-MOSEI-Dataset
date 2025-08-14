# inference.py
import os
import sys
import logging
from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, pipeline
from moviepy.editor import VideoFileClip

# Add project root to path for our model imports
sys.path.append(str(Path(__file__).resolve().parent))

from src.models.fusion import TransformerFusionModel
from config import (
    TEXT_EMBEDDING_DIM, HIDDEN_DIM, NUM_ATTENTION_HEADS,
    NUM_TRANSFORMER_LAYERS, DROPOUT_RATE
)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- Feature Extraction Functions ---

def extract_audio_from_video(video_path, audio_output_path):
    """Extracts audio from a video file and saves it as WAV."""
    try:
        logger.info(f"Extracting audio from {video_path}...")
        video_clip = VideoFileClip(str(video_path))
        audio_clip = video_clip.audio
        if audio_clip is None:
            logger.error(f"No audio track found in {video_path}")
            return False
        audio_clip.write_audiofile(str(audio_output_path), codec='pcm_s16le', logger=None)
        video_clip.close()
        logger.info(f"Audio saved to {audio_output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to extract audio from {video_path}: {e}")
        return False

def get_transcript_from_audio(audio_path):
    """Generates a transcript from an audio file using Whisper."""
    try:
        logger.info(f"Transcribing audio from {audio_path}...")
        # Use a smaller, faster model for quicker inference. Use 'openai/whisper-large-v3' for higher accuracy.
        pipe = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=DEVICE)
        result = pipe(str(audio_path))
        transcript = result['text'].strip()
        logger.info(f"Transcript: '{transcript[:100]}...'")
        return transcript
    except Exception as e:
        logger.error(f"Failed to transcribe audio {audio_path}: {e}")
        return ""

def get_text_embedding(text, tokenizer, bert_model, device):
    """Converts raw text to a BERT [CLS] embedding."""
    try:
        inputs = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=512,  # Standard BERT max length
            return_tensors="pt"
        )
        inputs = {key: val.to(device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = bert_model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].cpu()
        return embedding
    except Exception as e:
        logger.error(f"Failed to create text embedding: {e}")
        return torch.zeros(1, TEXT_EMBEDDING_DIM)

# --- Main Inference Logic ---

def run_inference():
    """
    Main function to run inference on a directory of video files.
    """
    global DEVICE # Use the globally determined device

    # 1. Get user input
    model_path_str = input("Enter the path to your trained model checkpoint (.pt): ").strip()
    video_dir_str = input("Enter the path to the directory containing your .mp4 files: ").strip()

    model_path = Path(model_path_str)
    video_dir = Path(video_dir_str)

    if not model_path.exists():
        logger.error(f"Model checkpoint not found: {model_path}")
        return
    if not video_dir.is_dir():
        logger.error(f"Video directory not found: {video_dir}")
        return

    # 2. Setup Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {DEVICE}")

    # 3. Load the trained model
    logger.info("Loading trained model...")
    model = TransformerFusionModel(
        text_dim=TEXT_EMBEDDING_DIM,
        audio_dim=0, # Not used in text-only
        visual_dim=0, # Not used in text-only
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_TRANSFORMER_LAYERS,
        num_heads=NUM_ATTENTION_HEADS,
        dropout_rate=DROPOUT_RATE
    )
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    logger.info("Model loaded successfully.")

    # 4. Load the feature extractor (BERT)
    logger.info("Loading BERT model for feature extraction...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased").to(DEVICE)
    bert_model.eval()

    # 5. Process each video file
    video_files = list(video_dir.glob("*.mp4"))
    if not video_files:
        logger.error(f"No .mp4 files found in {video_dir}")
        return
    
    predictions_data = []
    temp_dir = Path("./temp_audio")
    temp_dir.mkdir(exist_ok=True)

    for video_path in tqdm(video_files, desc="Processing Videos"):
        video_id = video_path.stem
        temp_audio_path = temp_dir / f"{video_id}.wav"
        
        # Step A: Extract Audio
        if not extract_audio_from_video(video_path, temp_audio_path):
            continue

        # Step B: Get Transcript
        transcript = get_transcript_from_audio(temp_audio_path)
        if not transcript:
            logger.warning(f"Skipping video {video_id} due to empty transcript.")
            continue
        
        # Step C: Get Text Embedding
        text_embedding = get_text_embedding(transcript, tokenizer, bert_model, DEVICE)
        
        # Step D: Prepare input for the model
        model_input = {"language": text_embedding.to(DEVICE)}

        # Step E: Get Prediction
        with torch.no_grad():
            prediction = model(model_input)
            sentiment_score = prediction.item()
        
        logger.info(f"Video: {video_path.name}, Predicted Sentiment: {sentiment_score:.4f}")
        predictions_data.append({"ID": video_path.name, "Label": sentiment_score})

        # Clean up temporary audio file
        os.remove(temp_audio_path)

    # 6. Save results to CSV
    if predictions_data:
        df = pd.DataFrame(predictions_data)
        output_csv_path = video_dir / "predictions.csv"
        df.to_csv(output_csv_path, index=False)
        logger.info(f"Inference complete! Predictions saved to {output_csv_path}")
    else:
        logger.warning("No predictions were made.")

    # Clean up temp directory
    os.rmdir(temp_dir)

if __name__ == "__main__":
    run_inference()
