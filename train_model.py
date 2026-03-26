"""
train_model.py — Train a RandomForestClassifier on collected gesture data.

Usage
-----
    python train_model.py

Reads   : dataset/gesture_data.csv
Outputs : model/gesture_model.pkl
          model/label_encoder.pkl
"""

import os
import sys
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# ── Paths ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
CSV_PATH = os.path.join(BASE_DIR, "dataset", "gesture_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "gesture_model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")


def load_dataset(csv_path: str):
    """
    Load the gesture CSV and return feature matrix X and label array y.

    The CSV is expected to have a header row.  The last column is the
    label; all preceding columns are numeric features.
    """
    if not os.path.isfile(csv_path):
        print(f"[ERROR] Dataset not found at {csv_path}")
        print("        Run data_collection.py first to collect samples.")
        sys.exit(1)

    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1, dtype=str)

    if data.ndim == 1:
        # Only one sample in the file — reshape to (1, cols)
        data = data.reshape(1, -1)

    if data.shape[0] == 0:
        print("[ERROR] Dataset is empty. Collect more samples first.")
        sys.exit(1)

    X = data[:, :-1].astype(np.float64)
    y = data[:, -1]
    return X, y


def main() -> None:
    print("=" * 60)
    print("  Hand Gesture Model Training")
    print("=" * 60)

    # ── 1. Load data ─────────────────────────────────────────────────
    X, y = load_dataset(CSV_PATH)
    print(f"\n[INFO] Loaded {X.shape[0]} samples with {X.shape[1]} features.")

    # Check minimum class count for stratified split
    unique, class_counts = np.unique(y, return_counts=True)
    print(f"[INFO] Classes: {dict(zip(unique, class_counts))}")

    if len(unique) < 2:
        print("[ERROR] Need at least 2 gesture classes to train. "
              "Collect more data.")
        sys.exit(1)

    # Ensure every class has at least 2 samples for stratification
    min_count = class_counts.min()
    if min_count < 2:
        print(f"[WARNING] Some classes have fewer than 2 samples. "
              f"Minimum is {min_count}. Collect more data for robust training.")

    # ── 2. Encode labels ─────────────────────────────────────────────
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # ── 3. Train / test split (80/20, stratified) ────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )
    print(f"[INFO] Training samples: {len(X_train)}  |  "
          f"Test samples: {len(X_test)}")

    # ── 4. Train RandomForestClassifier ──────────────────────────────
    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    print("\n[INFO] Training RandomForestClassifier (n_estimators=200) …")
    clf.fit(X_train, y_train)
    print("[INFO] Training complete.")

    # ── 5. Evaluate ──────────────────────────────────────────────────
    y_pred = clf.predict(X_test)
    report = classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        zero_division=0,
    )
    print("\n-- Classification Report ----------------------------------")
    print(report)

    # ── 6. Save model and label encoder ──────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    print(f"[INFO] Model saved to     {MODEL_PATH}")
    print(f"[INFO] Encoder saved to   {ENCODER_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
