"""
Ensemble Model Training Script
Project: Abusive Comment Detection and Prevention

Ensemble:
TF-IDF + (Logistic Regression + Linear SVM + Naive Bayes)
"""

import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ---------------- CONFIG ----------------
TRAIN_PATH = "data/splits/train.csv"
VAL_PATH = "data/splits/val.csv"
MODEL_DIR = "models"
MODEL_PATH = "models/ensemble_tfidf.joblib"
RANDOM_STATE = 42
# ---------------------------------------

def train_ensemble():
    print("[INFO] Loading datasets...")
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)

    X_train = train_df["comment"]
    y_train = train_df["label"]

    X_val = val_df["comment"]
    y_val = val_df["label"]

    print("[INFO] Building ensemble pipeline...")

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=25000,
        min_df=2
    )

    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    nb = MultinomialNB()

    svm = LinearSVC()

    ensemble_clf = VotingClassifier(
        estimators=[
            ("lr", lr),
            ("nb", nb),
            ("svm", svm)
        ],
        voting="hard"  # soft requires predict_proba; LinearSVC doesn't support it
    )

    pipeline = Pipeline([
        ("tfidf", tfidf),
        ("ensemble", ensemble_clf)
    ])

    print("[INFO] Training ensemble model...")
    pipeline.fit(X_train, y_train)

    print("[INFO] Evaluating ensemble model...")
    y_pred = pipeline.predict(X_val)

    print("\n===== ENSEMBLE CLASSIFICATION REPORT =====")
    print(classification_report(y_val, y_pred))

    print("===== ENSEMBLE CONFUSION MATRIX =====")
    print(confusion_matrix(y_val, y_pred))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    print(f"\n[INFO] Ensemble model saved at: {MODEL_PATH}")

if __name__ == "__main__":
    train_ensemble()
