import os
import pandas as pd

def load_ravdess_metadata(data_path):
    """
    Scan the RAVDESS folder and return a DataFrame with 'path' and 'emotion' columns.
    """
    file_paths = []
    emotions = []
    emotion_map = {
        1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
        5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'
    }

    for actor_folder in os.listdir(data_path):
        actor_path = os.path.join(data_path, actor_folder)
        if os.path.isdir(actor_path):
            for file in os.listdir(actor_path):
                if file.endswith('.wav'):
                    file_paths.append(os.path.join(actor_path, file))
                    parts = file.split('-')
                    emotion_code = int(parts[2])
                    emotions.append(emotion_code)

    emotion_labels = [emotion_map.get(code, 'unknown') for code in emotions]
    df = pd.DataFrame({'path': file_paths, 'emotion': emotion_labels})
    return df

# Example usage (if run as script):
if __name__ == "__main__":
    df = load_ravdess_metadata('/content/audio_speech_actors_01-24')
    df.to_csv('data/metadata.csv', index=False)
    print(f"✅ Saved metadata with {len(df)} samples.")
