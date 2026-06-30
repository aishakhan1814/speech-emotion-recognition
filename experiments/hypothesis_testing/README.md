# Hypothesis Testing - Male vs Female in Speech Emotion Recognition

## Objective
Performed statistical hypothesis testing (independent t-test) to compare audio features between male and female speakers in the RAVDESS SER dataset.

## Methodology
- **Dataset**: RAVDESS (audio_speech_actors_01-24)
- **Feature Extracted**: RMS Energy (Root Mean Square)
- **Statistical Test**: Two-sample t-test + Cohen's d effect size
- **Sample Size**: 400 files

## Key Results
- Statistically significant difference found in RMS Energy between male and female speakers (p < 0.05).
- Demonstrates **hypothesis-driven experimentation** and A/B testing approach in audio/ML projects.

## Skills Demonstrated
- Audio feature extraction (librosa)
- Statistical hypothesis testing (scipy)
- Exploratory Data Analysis
- Reproducible experimentation

## How to Run
1. Download the RAVDESS dataset
2. Run the Jupyter notebook
