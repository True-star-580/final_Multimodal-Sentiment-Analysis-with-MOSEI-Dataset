# Cross-lingual Multimodal Explainable Sentiment Recognition

This project is a submission for the "MSA Challenge @ The 4th Pazhou AI Competition". It features a robust, multimodal, and explainable model for sentiment analysis on English and Chinese video data.

---

## Project Overview

The core of this project is a multimodal sentiment analysis system that fulfills the competition's primary goals:

-   **Cross-lingual Capability**: Achieved by creating a unified English-only text space. All Chinese text from the training data (CH-SIMS v2.0) is translated to English using a high-quality translation model (`Helsinki-NLP/opus-mt-zh-en`). This allows a single, powerful English-centric text model to be used for all samples.
-   **Multimodality**: The final model integrates features from three modalities:
    -   **Text**: Transcripts are embedded using a pre-trained `bert-base-uncased` model.
    -   **Audio**: Raw audio waveforms are processed by a pre-trained `Wav2Vec2` model.
    -   **Visual**: Video frames are analyzed by a pre-trained Vision Transformer (ViT) model (`google/vit-base-patch16-224-in21k`).
-   **Model Explainability**: The model is designed with a modular architecture consisting of three independent "expert" sub-networks (one per modality) and a final fusion head. This allows us to generate a prediction from each modality and visualize its contribution to the final, fused prediction, providing clear insights into the model's decision-making process.

---

## Directory Structure (Submission)

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

## Methodology

### 1. Data Preparation (Translation & Unification)

A robust preprocessing pipeline was created to:
1.  Load the CMU-MOSEI (English) and CH-SIMS v2.0 (Chinese) datasets.
2.  Translate all Chinese transcripts to English.
3.  Unify the sentiment labels into the required 5-class format: `SNEG, WNEG, NEUT, WPOS, SPOS`.
4.  Create a master data manifest mapping all video files to their corresponding English text and labels.

### 2. Offline Feature Extraction

To ensure fast and efficient training, all multimodal features were pre-computed and saved to disk. This one-time process uses powerful pre-trained models to generate high-quality vector representations for each modality.

### 3. Staged Training for Optimal Performance

A multi-stage training strategy was employed to build the final model:
1.  **Train Unimodal Experts**: The Text, Audio, and Visual sub-networks were trained independently on the 5-class sentiment task. This creates three specialized models.
2.  **Train Fusion Head**: The weights of the three expert sub-networks were frozen. Then, only the final fusion layer was trained. This allows the model to learn the optimal way to combine the insights from its already-intelligent experts, leading to a more stable and higher-performing final model. Advanced techniques like automatic mixed precision, a learning rate scheduler, and early stopping were used to maximize performance.

### 4. Inference and Explainability

The final `test_script.py` performs the following for each video in the `Test_Data` directory:
1.  Loads the final trained model (`model_weights.pt`).
2.  Extracts text, audio, and visual features on-the-fly.
3.  Generates a final prediction and saves it to `Test_Results/label_prediction.csv`.
4.  Generates an **explainability chart** for each video, visualizing the confidence of the Text, Audio, Visual, and Fused predictions in the final outcome. These charts are saved in `Test_Supplements/`.

---

## How to Run the Submission

1.  **Prerequisites**: A Linux environment with Python 3.9+, CUDA, and an NVIDIA GPU (16GB VRAM recommended).
2.  **Installation**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Place Data**: Place the private test video files (`.mp4`) into the `Test_Data/` directory.
4.  **Execute**: Run the main script from within the `submission/` directory.
    ```bash
    python test_script.py
    ```
5.  **Outputs**:
    -   Predictions will be saved to `Test_Results/label_prediction.csv`.
    -   Explainability charts will be saved as PNG files in `Test_Supplements/`.

## Running the Inference Notebook On Google Colab

Follow these steps to run the model inference on the competition's public test data using Google Colab.

### 1. Environment Setup

1.  **Open in Colab**: Open the [`final_multimal_msa.ipynb`](https://github.com/True-star-580/final_Multimodal-Sentiment-Analysis-with-MOSEI-Dataset/blob/main/final_multimal_msa.ipynb) notebook in Google Colab.
2.  **Select GPU Runtime**: Navigate to **Runtime** > **Change runtime type** and select **T4 GPU** from the dropdown menu.

### 2. Kaggle API Credentials

The notebook needs to download the private competition dataset, which requires authentication.

1.  Run the first code cell. You will be prompted to enter your Kaggle username and API token.
2.  **To find your credentials**:
    *   Go to your Kaggle Account settings page: `https://www.kaggle.com/settings`.
    *   Your **Username** is listed on this page.
    *   Under the **API** section, click **Create New Token**. This will download a `kaggle.json` file containing your token.

### 3. Execute Inference

1.  Continue running the notebook cells in order.
2.  When you reach the inference cell, you will be prompted to provide the following paths:
    *   **Model Path**: The location of the trained model file.
      *Example:*
      ```
      /content/multimodal_fusion_best.pt
      ```
    *   **Test Data Path**: The directory containing the public test data. After downloading via the Kaggle API, the default path is usually:
      *Example:*
      ```
      /root/.cache/kagglehub/competitions/msa-challenge-at-the-4th-pazhou-ai-competition/Public_Test_Data/Public_Test_Data
      ```
3.  Paste each path into its respective dialog box and press **Enter**. The script will then proceed with inference.

