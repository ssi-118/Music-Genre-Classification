from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

UPLOAD_ALLOWED_EXTENSIONS = {"au", "mp3", "wav"}
TRAINING_ALLOWED_EXTENSIONS = {"au", "mp3", "wav"}


def is_allowed_file(filename: str) -> bool:
    if "." not in filename:
        return False
    return filename.rsplit(".", 1)[1].lower() in UPLOAD_ALLOWED_EXTENSIONS


def is_trainable_file(filename: str) -> bool:
    if "." not in filename:
        return False
    return filename.rsplit(".", 1)[1].lower() in TRAINING_ALLOWED_EXTENSIONS


def extract_features(audio_path: Path) -> np.ndarray:
    y, sr = librosa.load(str(audio_path), sr=None, mono=True, duration=30)

    if y.size == 0:
        raise ValueError("The uploaded audio file is empty or unreadable.")

    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    harmonic, _ = librosa.effects.hpss(y)
    tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)

    feature_vector = np.concatenate([
        np.mean(mfcc, axis=1),
        np.mean(chroma, axis=1),
        np.mean(mel, axis=1),
        np.mean(spectral_contrast, axis=1),
        np.mean(tonnetz, axis=1),
    ])

    return feature_vector.reshape(1, -1)
