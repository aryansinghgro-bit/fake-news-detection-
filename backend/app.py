"""TruthScan AI — Flask backend serving fake-news predictions.

Endpoints:
  GET  /api/health       — service health check
  GET  /api/model-info    — model metrics from training
  POST /api/predict       — analyze a news article

The model and vectorizer are loaded once at startup and reused for every
request.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import joblib
from flask import Flask, jsonify, request
from flask_cors import CORS

from preprocessing import combine_and_clean

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "fake_news_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
METRICS_PATH = BASE_DIR / "metrics.json"

MAX_CONTENT_LENGTH = 1024 * 1024  # 1 MB request cap
MIN_TITLE_LEN = 5
MIN_TEXT_LEN = 30

ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("truthscan")

# --------------------------------------------------------------------------- #
# Application setup
# --------------------------------------------------------------------------- #

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
CORS(app, resources={r"/api/*": {"origins": ALLOWED_ORIGINS}})

# Load model artifacts once at import time.
model = None
vectorizer = None
metrics = None


def load_artifacts() -> None:
    global model, vectorizer, metrics
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        logger.error("Model artifacts not found. Run `python train_model.py` first.")
        return
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    logger.info("Model and vectorizer loaded.")
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
        logger.info("Metrics loaded.")
    else:
        metrics = None


load_artifacts()


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #

def _validate_input(data) -> tuple[str | None, str | None, str | None]:
    """Return (title, text, error_message). On success error is None."""
    if not isinstance(data, dict):
        return None, None, "Request body must be a JSON object."

    title = data.get("title")
    text = data.get("text")

    if title is None:
        return None, None, "News headline is required."
    if text is None:
        return None, None, "News article is required."

    if not isinstance(title, str) or not isinstance(text, str):
        return None, None, "Headline and article must be text."

    title = title.strip()
    text = text.strip()

    if not title:
        return None, None, "News headline is required."
    if not text:
        return None, None, "News article is required."
    if len(title) < MIN_TITLE_LEN:
        return None, None, f"Headline must be at least {MIN_TITLE_LEN} characters."
    if len(text) < MIN_TEXT_LEN:
        return None, None, f"Article must be at least {MIN_TEXT_LEN} characters."

    return title, text, None


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@app.get("/api/health")
def health():
    ready = model is not None and vectorizer is not None
    return jsonify({
        "status": "healthy" if ready else "degraded",
        "service": "TruthScan AI",
        "model_loaded": ready,
    }), (200 if ready else 503)


@app.get("/api/model-info")
def model_info():
    if metrics is None:
        return jsonify({"error": "Model metrics are not available. Train the model first."}), 503
    return jsonify(metrics), 200


@app.post("/api/predict")
def predict():
    if model is None or vectorizer is None:
        return jsonify({"error": "Analysis service is not ready. Please try again shortly."}), 503

    body = request.get_json(silent=True)
    if body is None:
        return jsonify({"error": "Invalid JSON. Send a JSON object with 'title' and 'text'."}), 400

    title, text, err = _validate_input(body)
    if err:
        return jsonify({"error": err}), 400

    try:
        cleaned = combine_and_clean(title, text)
        if not cleaned:
            return jsonify({"error": "The provided text could not be processed. Please enter a valid article."}), 400

        features = vectorizer.transform([cleaned])
        proba = model.predict_proba(features)[0]
        # Class index 0 = REAL, 1 = FAKE (per training label mapping)
        real_prob = float(proba[0])
        fake_prob = float(proba[1])

        if real_prob >= fake_prob:
            prediction = "REAL"
            confidence = round(real_prob * 100, 2)
        else:
            prediction = "FAKE"
            confidence = round(fake_prob * 100, 2)

        return jsonify({
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": {
                "REAL": round(real_prob * 100, 2),
                "FAKE": round(fake_prob * 100, 2),
            },
        }), 200
    except Exception as exc:  # pragma: no cover
        logger.exception("Prediction failed")
        return jsonify({"error": "Something went wrong while analyzing the article. Please try again."}), 500


@app.errorhandler(404)
def not_found(_e):
    return jsonify({"error": "Endpoint not found."}), 404


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": "Request payload too large."}), 413


@app.errorhandler(500)
def server_error(_e):
    return jsonify({"error": "Something went wrong while analyzing the article. Please try again."}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=False)
