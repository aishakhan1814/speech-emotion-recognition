import librosa
import numpy as np

def extract_mfcc(file_path, n_mfcc=40, duration=3, sr=22050, offset=0.5):
    """
    Load audio and return the mean MFCC vector.
    """
    try:
        audio, _ = librosa.load(file_path, sr=sr, duration=duration, offset=offset)
        mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc).T, axis=0)
        return mfcc
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
