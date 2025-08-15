# submission/test_script.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
import logging
import cv2
import librosa
from tqdm import tqdm
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import os
import sys
import warnings
from transformers import AutoTokenizer, AutoModel, pipeline
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
import seaborn as sns

# --- Basic Setup ---
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Model Definition (Must be self-contained in the script) ---
TEXT_DIM, AUDIO_DIM, VISUAL_DIM = 768, 768, 512
class SubNet(nn.Module):
    def __init__(self, in_size, hidden_size, num_classes, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(in_size)
        self.linear1 = nn.Linear(in_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.linear3 = nn.Linear(hidden_size // 2, num_classes)
    def forward(self, x):
        h2 = self.dropout(F.relu(self.linear2(F.relu(self.linear1(self.norm(x))))))
        return h2, self.linear3(h2)

class MultimodalFinal(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=5, dropout=0.4):
        super().__init__()
        self.text_subnet = SubNet(TEXT_DIM, hidden_dim, num_classes, dropout)
        self.audio_subnet = SubNet(AUDIO_DIM, hidden_dim, num_classes, dropout)
        self.visual_subnet = SubNet(VISUAL_DIM, hidden_dim, num_classes, dropout)
        fusion_input_dim = (hidden_dim // 2) * 3
        self.fusion_layer = nn.Sequential(
            nn.LayerNorm(fusion_input_dim), nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, text, audio, visual):
        text_h, text_logits = self.text_subnet(text)
        audio_h, audio_logits = self.audio_subnet(audio)
        visual_h, visual_logits = self.visual_subnet(visual)
        combined_h = torch.cat([text_h, audio_h, visual_h], dim=1)
        fused_logits = self.fusion_layer(combined_h)
        return {"fused": fused_logits, "text": text_logits, "audio": audio_logits, "visual": visual_logits}

# --- Feature Extractor (Self-contained) ---
class FeatureExtractor:
    def __init__(self, device):
        self.device = device
        self.translator = pipeline("translation", model="Helsinki-NLP/opus-mt-zh-en", device=0)
        self.text_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.text_model = AutoModel.from_pretrained("bert-base-uncased").to(device).eval()
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
        self.audio_processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
        self.audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device).eval()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.visual_model = nn.Sequential(*list(resnet.children())[:-1]).to(device).eval()
        self.visual_transform = transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_features_for_video(self, video_path):
        import torchaudio
        # 1. Extract and Transcribe Audio
        temp_audio_path = Path("./temp_audio.wav")
        with VideoFileClip(str(video_path)) as clip:
            if clip.audio:
                clip.audio.write_audiofile(str(temp_audio_path), codec='pcm_s16le', fps=16000, logger=None)
            else:
                return None
        
        asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=0)
        transcript = asr_pipeline(str(temp_audio_path))['text']
        temp_audio_path.unlink()

        # 2. Translate if necessary
        is_chinese = any('\u4e00' <= char <= '\u9fff' for char in transcript)
        english_transcript = self.translator(transcript)[0]['translation_text'] if is_chinese else transcript

        # 3. Extract Multimodal Features
        with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
            # Text
            inputs = self.text_tokenizer(english_transcript, return_tensors="pt", max_length=128, padding=True, truncation=True).to(self.device)
            text_feat = self.text_model(**inputs).last_hidden_state[:, 0, :].squeeze()
            # Audio
            waveform, sr = torchaudio.load(video_path)
            if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
            if sr != 16000: waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
            inputs = self.audio_processor(waveform.squeeze().to(self.device), sampling_rate=16000, return_tensors="pt").to(self.device)
            audio_feat = self.audio_model(**inputs).last_hidden_state.mean(dim=1).squeeze()
            # Visual
            cap = cv2.VideoCapture(str(video_path)); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); frames = []
            indices = np.linspace(0, total - 1, 16, dtype=int)
            for i in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i); ret, frame = cap.read()
                if ret: frames.append(self.visual_transform(frame))
            cap.release()
            visual_feat = self.visual_model(torch.stack(frames).to(self.device)).flatten(1).mean(dim=0)
        
        return {"text": text_feat, "audio": audio_feat, "visual": visual_feat, "transcript": english_transcript}

def main():
    """Main function to be executed by the competition organizers."""
    logging.info("--- Starting Final Inference and Explainability Script ---")
    
    # Define paths based on submission structure
    TEST_DATA_DIR = Path("./Test_Data")
    RESULTS_DIR = Path("./Test_Results")
    SUPPLEMENTS_DIR = Path("./Test_Supplements")
    MODEL_PATH = Path("./model/model_weights.pt")
    RESULTS_DIR.mkdir(exist_ok=True); SUPPLEMENTS_DIR.mkdir(exist_ok=True)
    
    LABEL_INT_TO_STR = {0: 'SNEG', 1: 'WNEG', 2: 'NEUT', 3: 'WPOS', 4: 'SPOS'}

    # 1. Load Model
    logging.info(f"Loading final model from {MODEL_PATH}")
    model = MultimodalFinal().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 2. Initialize Feature Extractor
    extractor = FeatureExtractor(DEVICE)
    
    # 3. Process Test Videos
    video_files = list(TEST_DATA_DIR.glob("*.mp4"))
    predictions = []
    
    for video_path in tqdm(video_files, desc="Processing Private Test Set"):
        try:
            features = extractor.get_features_for_video(video_path)
            if features is None: continue

            # Get model predictions
            with torch.no_grad():
                outputs = model(features['text'].unsqueeze(0).to(DEVICE), 
                                features['audio'].unsqueeze(0).to(DEVICE), 
                                features['visual'].unsqueeze(0).to(DEVICE))

            # --- A. Save Prediction ---
            fused_probs = F.softmax(outputs['fused'], dim=1).squeeze()
            final_pred_idx = torch.argmax(fused_probs).item()
            final_label = LABEL_INT_TO_STR[final_pred_idx]
            predictions.append({"ID": video_path.name, "Label": final_label})

            # --- B. Save Explanation Chart ---
            explanation_data = {
                'Text': F.softmax(outputs['text'], dim=1).squeeze()[final_pred_idx].item(),
                'Audio': F.softmax(outputs['audio'], dim=1).squeeze()[final_pred_idx].item(),
                'Visual': F.softmax(outputs['visual'], dim=1).squeeze()[final_pred_idx].item(),
                'Fused': fused_probs[final_pred_idx].item()
            }
            df_exp = pd.DataFrame(list(explanation_data.items()), columns=['Modality', 'Confidence in Final Prediction'])
            
            plt.figure(figsize=(8, 6))
            sns.barplot(x='Modality', y='Confidence in Final Prediction', data=df_exp, palette='viridis')
            plt.title(f"Explainability for '{video_path.name}'\nFinal Prediction: {final_label} | Transcript: '{features['transcript'][:50]}...'")
            plt.ylabel("Confidence Score"); plt.ylim(0, 1)
            for index, row in df_exp.iterrows():
                plt.text(index, row['Confidence in Final Prediction'] + 0.02, f"{row['Confidence in Final Prediction']:.2f}", color='black', ha="center")
            
            chart_path = SUPPLEMENTS_DIR / f"{video_path.stem}_explanation.png"
            plt.savefig(chart_path); plt.close()

        except Exception as e:
            logging.error(f"Failed to process {video_path.name}: {e}")
            # Add a default prediction in case of error
            predictions.append({"ID": video_path.name, "Label": "NEUT"})

    # 4. Save Final Predictions CSV
    df_preds = pd.DataFrame(predictions)
    output_csv_path = RESULTS_DIR / "label_prediction.csv"
    df_preds.to_csv(output_csv_path, index=False)
    
    logging.info(f"--- Inference Complete. Predictions saved to {output_csv_path} ---")

if __name__ == '__main__':
    main()
