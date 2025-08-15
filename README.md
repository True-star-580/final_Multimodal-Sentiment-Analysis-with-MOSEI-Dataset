# Multimodal Sentiment Analysis

This work is an extention of [JugalGajjar's][https://github.com/JugalGajjar/Multimodal-Sentiment-Analysis-with-MOSEI-Dataset] and [CMU Multimodal Opinion Sentiment and Emotion Intensity ](http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/). It is prepared for the following Kaggle competition; [MSA Challenge @ The 4th Pazhou AI Competition](https://www.kaggle.com/competitions/msa-challenge-at-the-4th-pazhou-ai-competition/data?select=Public_Test_Data)


---

## Project Overview

This project aims to build a sentiment analysis system that integrates:
- **Textual data** (transcriptions of utterances)
- **Acoustic features** (tone, pitch, MFCC)
- **Visual expressions** (facial gestures and movement)

We utilize transformer-based encoders per modality and implement early fusion using cross-attention, followed by a regression head to predict sentiment on a continuous scale from -3 to +3.

---

## Directory Structure

```
Multimodal-Sentiment-Analysis-with-MOSEI-Dataset/
├── config.py                  # Configuration parameters
├── data/                      # Data storage directory
│   ├── raw/                   # Raw downloaded data
│   └── processed/             # Preprocessed features
├── src/
│   ├── data/
│   │   ├── download.py        # Download MOSEI dataset
│   │   ├── preprocess.py      # Feature extraction
│   │   └── dataset.py         # PyTorch dataset classes
│   ├── models/
│   │   ├── text.py            # Text-only models
│   │   ├── audio.py           # Audio-only models
│   │   ├── visual.py          # Visual-only models
│   │   ├── fusion.py          # Multimodal fusion models
│   │   └── attention.py       # Cross-attention mechanisms
│   ├── training/
│   │   ├── trainer.py         # Training loop
│   │   └── metrics.py         # Evaluation metrics
│   └── utils/
│       ├── logging.py         # Logging utilities
│       └── visualization.py   # Result visualization
├── scripts/
│   ├── train_unimodal.py      # Train unimodal baselines
│   ├── train_multimodal.py    # Train multimodal model
│   └── evaluate.py            # Evaluation script
├── main.py                    # Main entry point
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Dataset: CMU-MOSEI

- 23,500 annotated utterances from 1,000+ speakers
- Aligned text, audio, and visual data
- Continuous sentiment score from -3 (very negative) to +3 (very positive)

To download and process the dataset:

```bash
# Clone and set up the SDK
git clone https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.git
cd CMU-MultimodalSDK
pip install -e .

# Back in project root:
python src/data/download.py
python src/data/preprocess.py
```

---

## Methodology

### Feature Extraction
- **Text**: BERT embeddings (`bert-base-uncased`)
- **Audio**: Preprocessed audio features
- **Visual**: Facial action units and expression vectors (OpenFace or precomputed)

### Architecture
- **Modality Encoders**: Independent transformer layers per modality
- **Fusion**: Fusion using a cross-attention transformer
- **Output**: Regression head to predict a sentiment score ∈ [-3, +3]

---

## Evaluation Metrics Used

- Mean Absolute Error (MAE)
- Pearson Correlation Coefficient
- Binary Accuracy (positive vs negative sentiment)
- F1 Score for binary classification

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/JugalGajjar/Multimodal-Sentiment-Analysis-with-MOSEI-Dataset.git
cd Multimodal-Sentiment-Analysis-with-MOSEI-Dataset
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Execute main.py
```bash
python main.py
```

## License

This project is released under the [MIT License](https://opensource.org/licenses/MIT).
