"""
train.py - Train Spam Comment Classifier

Trains a Multinomial Naive Bayes model using TF-IDF features
on the full YouTube Spam Collection dataset (5 CSV files).

Usage:
    python train.py
"""

import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


DATA_DIR = "data"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "spam_classifier.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

# All YouTube spam dataset CSV files
CSV_FILES = [
    "Youtube01-Psy.csv",
    "Youtube02-KatyPerry.csv",
    "Youtube03-LMFAO.csv",
    "Youtube04-Eminem.csv",
    "Youtube05-Shakira.csv",
]


def load_data():
    """Load and combine all YouTube spam CSV files."""
    dataframes = []
    for csv_file in CSV_FILES:
        filepath = os.path.join(DATA_DIR, csv_file)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            source = csv_file.replace(".csv", "").split("-")[-1]
            df["SOURCE"] = source
            dataframes.append(df)
            print(f"  ✓ Loaded {csv_file}: {len(df)} comments")
        else:
            print(f"  ✗ File not found: {filepath}")

    if not dataframes:
        raise FileNotFoundError("No CSV files found in data/ directory!")

    combined = pd.concat(dataframes, ignore_index=True)
    return combined


def train_model():
    """Train the spam classifier on the full dataset."""
    print("=" * 60)
    print("  SPAM COMMENT CLASSIFIER - TRAINING")
    print("=" * 60)

    # --- Load Data ---
    print("\n📂 Loading datasets...")
    data = load_data()

    # Keep only CONTENT and CLASS columns
    data = data[["CONTENT", "CLASS"]].dropna()
    print(f"\n📊 Dataset Summary:")
    print(f"   Total comments : {len(data)}")
    print(f"   Spam (1)       : {data['CLASS'].sum()}")
    print(f"   Ham  (0)       : {len(data) - data['CLASS'].sum()}")
    print(f"   Spam ratio     : {data['CLASS'].mean():.1%}")

    # --- Features & Labels ---
    X = data["CONTENT"]
    y = data["CLASS"]

    # --- Quick evaluation split (for reporting only) ---
    X_train_eval, X_test_eval, y_train_eval, y_test_eval = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fit vectorizer + model on eval split to report accuracy
    eval_vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X_train_tfidf = eval_vectorizer.fit_transform(X_train_eval)
    X_test_tfidf = eval_vectorizer.transform(X_test_eval)

    eval_model = MultinomialNB()
    eval_model.fit(X_train_tfidf, y_train_eval)
    y_pred = eval_model.predict(X_test_tfidf)

    print(f"\n📈 Evaluation (80/20 split):")
    print(f"   Accuracy: {accuracy_score(y_test_eval, y_pred):.4f}")
    print(f"\n   Classification Report:")
    print(classification_report(y_test_eval, y_pred, target_names=["Ham", "Spam"]))

    # --- Train FINAL model on ALL data ---
    print("🔧 Training final model on FULL dataset...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X_tfidf = vectorizer.fit_transform(X)

    model = MultinomialNB()
    model.fit(X_tfidf, y)

    # --- Save model and vectorizer ---
    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"   ✓ Model saved to {MODEL_PATH}")

    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"   ✓ Vectorizer saved to {VECTORIZER_PATH}")

    print("\n✅ Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    train_model()
