import joblib
import numpy as np
from features import extract_mfcc

def predict_emotion(audio_path, model_path='models/ser_mlp.pkl', encoder_path='models/label_encoder.pkl'):
    model = joblib.load(model_path)
    le = joblib.load(encoder_path)
    feat = extract_mfcc(audio_path)
    if feat is not None:
        feat = np.array([feat])  # shape (1, 40)
        pred_encoded = model.predict(feat)[0]
        emotion = le.inverse_transform([pred_encoded])[0]
        return emotion
    return None

if __name__ == "__main__":
    # Test with a sample file
    sample_path = '/content/audio_speech_actors_01-24/Actor_16/03-01-05-02-01-02-16.wav'  # adjust
    print("Predicted:", predict_emotion(sample_path))
