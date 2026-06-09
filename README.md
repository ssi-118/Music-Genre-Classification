# Music Genre Classification System

A web-based music genre classifier powered by a **Random Forest** model trained on audio features extracted with **librosa**. Upload an MP3 or WAV file and the system predicts its genre from 10 categories.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-3.0.3-lightgrey)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.1-orange)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Dataset Used

This project was trained using the **GTZAN Genre Collection Dataset**, a widely used benchmark dataset for music genre classification research.

- **Dataset:** GTZAN Genre Collection
- **Source:** https://www.kaggle.com/datasets/carlthome/gtzan-genre-collection
- **Total Audio Files:** 1,000
- **Genres:** 10
- **Audio Format:** WAV
- **Clip Duration:** 30 seconds each

## Supported Genres

blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock

---

## Features

- Upload MP3, WAV, or AU audio files (up to 25 MB)
- Classifies audio into 10 music genres
- Extracts 173 audio features using librosa (MFCC, Chroma, Mel Spectrogram, Spectral Contrast, Tonnetz)
- Falls back to a rule-based demo predictor if no trained model is present
- Clean web interface with instant results


---

## Project Structure

```
music-genre-classifier/
├── app.py                  # Flask backend + prediction logic
├── audio_utils.py          # Feature extraction utilities
├── train_model.py          # Model training script
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container config for deployment
├── runtime.txt             # Python version for deployment
├── README.md
├── model/
│   └── genre_model.pkl     # Trained Random Forest model
├── static/
│   ├── css/
│   └── js/
├── templates/
│   └── index.html          # Frontend UI
└── uploads/                # Temporary audio file storage
    └── .gitkeep
```

---

## Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/your-username/music-genre-classifier.git
cd music-genre-classifier
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model (optional)

If you have the GTZAN dataset, organize it as genre subfolders:

```
genres/
├── blues/
├── classical/
├── country/
...
```

Then run:

```bash
python train_model.py --dataset genres --output model/genre_model.pkl
```

If you skip this step, the app will use a rule-based fallback predictor.

### 5. Start the development server

```bash
python app.py
```

Visit **http://127.0.0.1:5000** in your browser.

---

## Running with Docker

```bash
docker build -t music-genre-classifier .
docker run -p 7860:7860 music-genre-classifier
```

Visit **http://localhost:7860**

---

## Deployment

### Hugging Face Spaces (recommended — free)

1. Create a new Space at [huggingface.co](https://huggingface.co/new-space)
   - SDK: **Docker**

2. Add the following `README.md` frontmatter at the top of your README:

```yaml
---
title: Music Genre Classifier
emoji: 🎵
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
---
```

3. Push your code:

```bash
git init
git remote add origin https://huggingface.co/spaces/your-username/music-genre-classifier
git add .
git commit -m "initial commit"
git push origin main
```

The app will be live at `https://your-username-music-genre-classifier.hf.space`

> **Note:** The port in `Dockerfile` must be `7860` for HF Spaces.

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| Flask | 3.0.3 | Web framework |
| gunicorn | 22.0.0 | Production WSGI server |
| librosa | 0.9.2 | Audio feature extraction |
| numba | 0.57.1 | Required by librosa |
| llvmlite | 0.40.1 | Required by numba |
| numpy | 1.24.4 | Numerical operations |
| scikit-learn | 1.5.1 | Random Forest classifier |
| joblib | 1.4.2 | Model serialization |
| soundfile | 0.12.1 | Audio file I/O |
| audioread | 3.0.1 | MP3 decoding backend |

---

## 🧠 Model Details

| Property | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Number of Trees | 250 |
| Feature Vector Size | 173 |
| Training Dataset | GTZAN Genre Collection |
| Total Samples | 1,000 Audio Clips |
| Genres | 10 |
| Train/Test Split | 80% / 20% |
| Class Weighting | balanced_subsample |

---

## Live Demo

🌐 https://soha118-music-genre-classification.hf.space

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
