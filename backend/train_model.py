"""Train the TruthScan AI fake-news classification model.

Pipeline:
  1. Load the labeled dataset from backend/data/news.csv
  2. Validate required columns (title, text, label)
  3. Drop rows with missing values
  4. Combine headline + body via the shared preprocessing module
  5. Stratified train/test split (80/20)
  6. TF-IDF vectorization (unigrams + bigrams, 10k features)
  7. Logistic Regression classifier
  8. Evaluate: accuracy, precision, recall, F1, confusion matrix
  9. Persist model + vectorizer to backend/model/
 10. Persist metrics to backend/metrics.json

Run:
    python train_model.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from preprocessing import combine_and_clean

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "news.csv"
MODEL_DIR = BASE_DIR / "model"
METRICS_PATH = BASE_DIR / "metrics.json"

MODEL_PATH = MODEL_DIR / "fake_news_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"

RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_FEATURES = 10000


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[ERROR] Dataset not found at {path}")
        print("Expected CSV columns: title, text, label (values: REAL, FAKE)")
        sys.exit(1)

    df = pd.read_csv(path)
    required = {"title", "text", "label"}
    missing = required - set(df.columns)
    if missing:
        print(f"[ERROR] Dataset missing required columns: {missing}")
        sys.exit(1)

    # Normalize labels to uppercase strings
    df["label"] = df["label"].astype(str).str.strip().str.upper()
    valid = {"REAL", "FAKE"}
    bad = set(df["label"].unique()) - valid
    if bad:
        print(f"[ERROR] Unexpected label values: {bad}. Expected REAL / FAKE.")
        sys.exit(1)

    before = len(df)
    df = df.dropna(subset=["title", "text", "label"]).reset_index(drop=True)
    df = df[df["text"].str.strip().str.len() > 0].reset_index(drop=True)
    after = len(df)
    if before != after:
        print(f"[INFO] Dropped {before - after} rows with missing/empty values.")

    print(f"[INFO] Loaded {len(df)} samples "
          f"(REAL: {(df['label'] == 'REAL').sum()}, "
          f"FAKE: {(df['label'] == 'FAKE').sum()})")
    return df


def build_features(df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    texts = [combine_and_clean(t, b) for t, b in zip(df["title"], df["text"])]
    labels = (df["label"] == "FAKE").astype(int).to_numpy()  # 1 = FAKE, 0 = REAL
    return texts, labels


def train() -> None:
    print("=" * 60)
    print("TruthScan AI — Model Training")
    print("=" * 60)

    df = load_dataset(DATA_PATH)
    texts, y = build_features(df)

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[INFO] Train: {len(X_train_text)} | Test: {len(X_test_text)}")

    print("[INFO] Building TF-IDF vectorizer (max_features=10000, ngram=(1,2))...")
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    print(f"[INFO] Feature matrix shape: {X_train.shape}")

    print("[INFO] Training Logistic Regression...")
    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=RANDOM_STATE,
        solver="liblinear",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)
    print(f"Accuracy : {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall   : {rec * 100:.2f}%")
    print(f"F1 Score : {f1 * 100:.2f}%")
    print(f"\nConfusion Matrix (rows=true, cols=pred):")
    print(f"  REAL(0) -> {cm[0]}")
    print(f"  FAKE(1) -> {cm[1]}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"\n[INFO] Saved model -> {MODEL_PATH}")
    print(f"[INFO] Saved vectorizer -> {VECTORIZER_PATH}")

    metrics = {
        "accuracy": round(float(acc) * 100, 2),
        "precision": round(float(prec) * 100, 2),
        "recall": round(float(rec) * 100, 2),
        "f1": round(float(f1) * 100, 2),
        "confusion_matrix": cm.tolist(),
        "label_mapping": {"0": "REAL", "1": "FAKE"},
        "train_samples": len(X_train_text),
        "test_samples": len(X_test_text),
        "total_samples": len(df),
        "max_features": MAX_FEATURES,
        "ngram_range": [1, 2],
        "model_type": "LogisticRegression",
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Saved metrics -> {METRICS_PATH}")
    print("\n[DONE] Training complete.")


if __name__ == "__main__":
    train()
