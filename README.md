# Speech Emotion Recognition

This project classifies emotions from speech using RAVDESS dataset.  
It uses MFCC features and a Multi‑Layer Perceptron (MLP) classifier.

## Dataset
- RAVDESS (1440 audio files, 8 emotions)

## Results
- Baseline MLP achieves ~43% accuracy on test set.
- Confusion matrix and classification report are available.

## How to Run
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Download the RAVDESS dataset and place it under `data/` or adjust paths in scripts.
4. Train: `python src/train.py`
5. Predict: `python src/predict.py /path/to/audio.wav`
6. Launch Gradio demo: `python app/app.py`

## Project Structure
- `src/`: Python modules for data loading, features, model, training, prediction.
- `notebooks/`: Jupyter notebook for experimentation.
- `models/`: Saved model and label encoder.
- `reports/`: Figures (confusion matrix).
- `app/`: Gradio web demo.
