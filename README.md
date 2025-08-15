# Multimodal Sentiment Analysis

This work is an extention of Jugal Gajjar [GitHub](https://github.com/JugalGajjar) and [CMU Multimodal Opinion Sentiment and Emotion Intensity ](http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/). It is prepared for the following Kaggle competition; [MSA Challenge @ The 4th Pazhou AI Competition](https://www.kaggle.com/competitions/msa-challenge-at-the-4th-pazhou-ai-competition/data?select=Public_Test_Data)


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
│   │   ├── download_v2.py     # Kaggle processed and raw dataset
│   │   ├── preprocess.py      # Feature extraction
│   │   └── dataset.py         # PyTorch dataset classes
│   ├── models/
│   │   ├── text.py            # Text-only models
│   │   ├── audio.py           # Audio-only models
│   │   ├── visual.py          # Visual-only models
│   │   ├── fusion.py          # Multimodal fusion models
│   │   ├── fusion_v2.py       # FINETUNED Multilingual model
│   │   └── attention.py       # Cross-attention mechanisms
│   ├── training/
│   │   ├── trainer.py         # Training loop
│   │   ├── trainer_v2.py      # Training v2 Multilingual model
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

## How to Run on Google Colab

Download this notebook [final_multimal_msa.ipynb](https://github.com/True-star-580/final_Multimodal-Sentiment-Analysis-with-MOSEI-Dataset/blob/main/final_multimal_msa.ipynb). Load it on Google Colab. Set runtime to T4 gpu, a single one will suffice for inference. 

Since the competition dataset is private, Kaggle Username and Token will be required, provide them in the first cell. To get the Token on Kggle, go to your profile on Kaggle, click on 'Settings', under 'API' click on 'Create New Token'. Your username is on the same page (it is not editable).

at the `infererence.py` code block, you will be prompted to load the model and the video files path, provide the paths by pasting directly to the dialog box that pops up, eg the model could be placed at 
```
'/content/multimodal_fusion_best.pt'.
```
Press enter and proceed to provide all the required paths. The competition data will be at 
```
'/root/.cache/kagglehub/competitions/msa-challenge-at-the-4th-pazhou-ai-competition/Public_Test_Data/Public_Test_Data'
```
 but feel free to specify the exact path.


## License

This project is released under the [MIT License](https://opensource.org/licenses/MIT).
