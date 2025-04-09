# Multimodal Sentiment Analysis with MOSEI Dataset

A transformer-based project for multimodal sentiment analysis using the [CMU-MOSEI dataset](http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/), combining text, audio, and visual modalities to better understand human sentiment.

---

## Project Overview

This project aims to build a sentiment analysis system that integrates:
- **Textual data** (transcriptions of utterances)
- **Acoustic features** (tone, pitch, MFCC)
- **Visual expressions** (facial gestures and movement)

We utilize transformer-based encoders per modality and implement early fusion using cross-attention, followed by a regression head to predict sentiment on a continuous scale from -3 to +3.

---

## Dataset: CMU-MOSEI

- 23,500 annotated utterances from 1,000+ speakers
- Aligned text, audio, and visual data
- Continuous sentiment score from -3 (very negative) to +3 (very positive)

To download and process the dataset:

```bash
# Clone and set up the SDK
git clone https://github.com/A2Zadeh/CMU-MultimodalSDK.git
cd CMU-MultimodalSDK
pip install -e .

# Back in project root:
python scripts/download_data.py
python scripts/preprocess.py
```

---

## Initial Methodology

### Feature Extraction
- **Text**: BERT embeddings
- **Audio**: MFCCs, chroma, spectral contrast
- **Visual**: Facial action units and expression vectors

### Architecture
- **Modality Encoders**: Independent transformer layers per modality
- **Fusion**: Early fusion using a cross-attention transformer
- **Output**: Regression head to predict a sentiment score ∈ [-3, +3]

---

## Evaluation Metrics To Be Used

- Mean Absolute Error (MAE)
- Binary Accuracy (positive vs negative sentiment)
- F1 Score for binary classification

---

## Expected Outcomes

- Demonstration of superior performance of multimodal over unimodal models
- Insight into which modalities contribute most to sentiment prediction
- A robust, reproducible, and modular multimodal pipeline

---

## 👥 Team Members

- Jugal Gajjar [GitHub](https://github.com/JugalGajjar), [LinkedIn](https://www.linkedin.com/in/jugal-gajjar/)
- Kaustik Ranaware [GitHub](https://github.com/KAUSTIKR), [LinkedIn](https://www.linkedin.com/in/kaustik/)

---

## License

This project is developed for academic and research purposes only. The model weights and code in this project are released under the [MIT License](https://opensource.org/licenses/MIT).
