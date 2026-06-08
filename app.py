from __future__ import annotations

import uuid
from pathlib import Path

import joblib
import numpy as np
try:
    import librosa
    _dummy = np.zeros(22050, dtype=np.float32)
    librosa.feature.mfcc(y=_dummy, sr=22050, n_mfcc=20)
    print("librosa warm-up complete")
except Exception as e:
    print(f"librosa warm-up failed: {e}")

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from audio_utils import extract_features, is_allowed_file

import os
os.environ["NUMBA_DISABLE_JIT"] = "1"
os.environ["NUMBA_CACHE_DIR"] = "/tmp"


BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = BASE_DIR / "uploads"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "genre_model.pkl"

MAX_AUDIO_SIZE_MB = 25

# Common GTZAN-style label mapping for models that return numeric genre classes.
GENRE_MAP = {
    0: "blues",
    1: "classical",
    2: "country",
    3: "disco",
    4: "hiphop",
    5: "jazz",
    6: "metal",
    7: "pop",
    8: "reggae",
    9: "rock",
}

app = Flask(__name__, template_folder=str(TEMPLATE_DIR), static_folder=str(STATIC_DIR))
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["MAX_CONTENT_LENGTH"] = MAX_AUDIO_SIZE_MB * 1024 * 1024

# The model is loaded once and reused for each request.
MODEL = None


def load_model():
    """Load the pre-trained model from disk if it exists."""
    global MODEL

    if MODEL is not None:
        return MODEL

    if not MODEL_PATH.exists():
        return None

    MODEL = joblib.load(MODEL_PATH)
    return MODEL


def normalize_prediction(raw_prediction) -> str:
    """Convert the model output into a readable genre label."""
    prediction = raw_prediction

    if isinstance(prediction, (list, tuple, np.ndarray)):
        prediction = prediction[0]

    if isinstance(prediction, np.generic):
        prediction = prediction.item()

    if isinstance(prediction, bytes):
        prediction = prediction.decode("utf-8", errors="ignore")

    if isinstance(prediction, str):
        return prediction

    if isinstance(prediction, (int, np.integer)) and int(prediction) in GENRE_MAP:
        return GENRE_MAP[int(prediction)]

    return str(prediction)


def fallback_predict(features: np.ndarray) -> str:
    """Return a deterministic demo prediction when no trained model is available."""
    scores = features.flatten()

    rhythm_score = float(np.mean(scores[20:32]))
    brightness_score = float(np.mean(scores[32:160]))
    contrast_score = float(np.mean(scores[160:167]))
    harmonic_score = float(np.mean(scores[167:173]))

    if contrast_score > 40 and harmonic_score > 0.02:
        return "rock"
    if brightness_score < 0.01 and contrast_score < 20:
        return "classical"
    if rhythm_score > 0.15 and harmonic_score > 0.01:
        return "pop"
    if rhythm_score > 0.12 and contrast_score > 25:
        return "hiphop"
    if brightness_score > 0.04 and contrast_score > 30:
        return "metal"
    if harmonic_score > 0.03:
        return "jazz"
    if rhythm_score < 0.07:
        return "blues"
    if brightness_score > 0.02:
        return "country"
    if contrast_score > 18:
        return "reggae"
    return "disco"


@app.route("/")
def index():
    """Render the main upload page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Handle file upload, extract features, and return the predicted genre."""
    if "file" not in request.files:
        return jsonify({"success": False, "message": "No file was uploaded."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"success": False, "message": "Please choose an audio file."}), 400

    if not is_allowed_file(file.filename):
        return (
            jsonify(
                {
                    "success": False,
                    "message": "Unsupported file format. Please upload an MP3 or WAV file.",
                }
            ),
            400,
        )

    model = load_model()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    original_name = secure_filename(file.filename)
    stored_name = f"{uuid.uuid4().hex}_{original_name}"
    saved_path = UPLOAD_DIR / stored_name

    file.save(saved_path)

    try:
        features = extract_features(saved_path)
        if model is None:
            genre = fallback_predict(features)
            message = f"Predicted genre: {genre} (demo fallback used because model/genre_model.pkl is missing)"
        else:
            raw_prediction = model.predict(features)
            genre = normalize_prediction(raw_prediction)
            message = f"Predicted genre: {genre}"

        return jsonify(
            {
                "success": True,
                "genre": genre,
                "message": message,
            }
        )
    except Exception as exc:
        app.logger.exception("Genre prediction failed")
        return (
            jsonify(
                {
                    "success": False,
                    "message": f"Could not process the file: {exc}",
                }
            ),
            500,
        )
    finally:
        if saved_path.exists():
            saved_path.unlink()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
