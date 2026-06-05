from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from audio_utils import extract_features, is_trainable_file


GENRE_DEFAULTS = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]


def load_dataset(dataset_dir: Path):
    """Read audio files from a folder structure like genres/genre_name/*.au."""
    features = []
    labels = []

    genre_dirs = [path for path in dataset_dir.iterdir() if path.is_dir()]

    for genre_dir in sorted(genre_dirs):
        genre_name = genre_dir.name.lower().strip()

        for audio_file in sorted(genre_dir.rglob("*")):
            if not audio_file.is_file() or not is_trainable_file(audio_file.name):
                continue

            try:
                feature_vector = extract_features(audio_file).flatten()
                features.append(feature_vector)
                labels.append(genre_name)
                print(f"Processed: {audio_file}")
            except Exception as exc:
                print(f"Skipped {audio_file}: {exc}")

    return np.array(features), np.array(labels)


def train_model(dataset_dir: Path, output_path: Path) -> None:
    features, labels = load_dataset(dataset_dir)

    if len(features) == 0:
        raise ValueError(
            "No training samples were found. Create folders like dataset/rock, dataset/jazz, etc."
        )

    if len(set(labels)) < 2:
        raise ValueError("Training requires at least two genre folders with audio files.")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    model = RandomForestClassifier(
        n_estimators=250,
        random_state=42,
        class_weight="balanced_subsample",
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\nValidation accuracy:", round(float(accuracy), 4))
    print("\nClassification report:\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    print(f"\nSaved trained model to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a music genre classification model.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("genres"),
        help="Folder containing genre subfolders with audio files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model/genre_model.pkl"),
        help="Path to save the trained model.",
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {args.dataset}. Create genre subfolders before training."
        )

    train_model(args.dataset, args.output)
