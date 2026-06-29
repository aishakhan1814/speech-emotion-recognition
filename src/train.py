import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
from tqdm import tqdm

from data_loader import load_ravdess_metadata
from features import extract_mfcc
from model import create_mlp_model

def main():
    # 1. Load metadata
    data_path = '/content/audio_speech_actors_01-24'  # adjust path as needed
    df = load_ravdess_metadata(data_path)
    print(f"Loaded {len(df)} samples.")

    # 2. Extract features
    features = []
    labels = []
    for idx in tqdm(range(len(df))):
        feat = extract_mfcc(df['path'].iloc[idx])
        if feat is not None:
            features.append(feat)
            labels.append(df['emotion'].iloc[idx])

    X = np.array(features)
    y = np.array(labels)
    print(f"Features shape: {X.shape}")

    # 3. Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 4. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
    )

    # 5. Train model
    model = create_mlp_model()
    model.fit(X_train, y_train)

    # 6. Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # 7. Save model and label encoder
    joblib.dump(model, 'models/ser_mlp.pkl')
    joblib.dump(le, 'models/label_encoder.pkl')
    print("✅ Model and encoder saved.")

if __name__ == "__main__":
    main()
