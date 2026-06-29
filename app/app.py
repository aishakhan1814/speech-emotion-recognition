import gradio as gr
import joblib
import numpy as np
import librosa
from src.features import extract_mfcc  # if you have that as a module

# If you can't import from src, you can inline the function
def predict_emotion(audio_file):
    model = joblib.load('models/ser_mlp.pkl')
    le = joblib.load('models/label_encoder.pkl')
    # Extract features
    audio, sr = librosa.load(audio_file, sr=22050, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    mfcc = np.array([mfcc])
    pred = model.predict(mfcc)[0]
    emotion = le.inverse_transform([pred])[0]
    return emotion

iface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Audio(source="upload", type="filepath"),
    outputs="text",
    title="Speech Emotion Recognition",
    description="Upload a .wav file and get the predicted emotion."
)
iface.launch(share=True)
