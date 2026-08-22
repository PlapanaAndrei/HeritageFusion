import numpy as np
import librosa

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=5)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc.T, axis=0)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast.T, axis=0)

    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)

    features = np.hstack([
        mfcc_mean,
        chroma_mean,
        contrast_mean,
        zcr_mean
    ])

    return features.reshape(1, -1)
