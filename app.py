"""
app.py - Spam Comment Detector Flask Application

Loads a pre-trained TF-IDF + Naive Bayes model and provides:
  - Manual comment classification
  - YouTube comment scraping + batch classification
"""

import os
import pickle
from flask import Flask, render_template, request
from scraper import scrape_comments

app = Flask(__name__)

# --- Load Pre-trained Model & Vectorizer ---
MODEL_PATH = os.path.join("models", "spam_classifier.pkl")
VECTORIZER_PATH = os.path.join("models", "tfidf_vectorizer.pkl")


def load_model():
    """Load the trained model and TF-IDF vectorizer."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(
            "Model files not found! Run 'python train.py' first."
        )

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


# Load model at startup
model, vectorizer = load_model()


def predict_comment(text):
    """Predict if a comment is spam or ham."""
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)[0]
    probability = model.predict_proba(text_tfidf)[0]
    return {
        "label": "Spam" if prediction == 1 else "Not Spam",
        "is_spam": bool(prediction == 1),
        "confidence": float(max(probability)) * 100,
    }


@app.route("/")
def index():
    """Render the main page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Predict spam/ham for a single comment."""
    comment_text = request.form.get("comment", "").strip()

    if not comment_text:
        return render_template("index.html", error="Please enter a comment.")

    result = predict_comment(comment_text)

    return render_template(
        "results.html",
        mode="single",
        comment=comment_text,
        result=result,
    )


@app.route("/scrape", methods=["POST"])
def scrape():
    """Scrape YouTube comments and classify them."""
    youtube_url = request.form.get("youtube_url", "").strip()
    max_comments = int(request.form.get("max_comments", 50))

    if not youtube_url:
        return render_template("index.html", error="Please enter a YouTube URL.")

    # Clamp max comments
    max_comments = max(5, min(max_comments, 200))

    try:
        comments = scrape_comments(youtube_url, max_comments=max_comments)
    except Exception as e:
        return render_template("index.html", error=f"Scraping failed: {str(e)}")

    if not comments:
        return render_template("index.html", error="No comments found for this video.")

    # Classify each comment
    results = []
    spam_count = 0
    for comment in comments:
        prediction = predict_comment(comment["text"])
        results.append({
            "author": comment["author"],
            "text": comment["text"],
            "time": comment["time"],
            "label": prediction["label"],
            "is_spam": prediction["is_spam"],
            "confidence": prediction["confidence"],
        })
        if prediction["is_spam"]:
            spam_count += 1

    stats = {
        "total": len(results),
        "spam": spam_count,
        "ham": len(results) - spam_count,
        "spam_pct": round(spam_count / len(results) * 100, 1) if results else 0,
    }

    return render_template(
        "results.html",
        mode="scrape",
        youtube_url=youtube_url,
        results=results,
        stats=stats,
    )


@app.route("/bulk", methods=["POST"])
def bulk():
    """Analyze multiple pasted comments (one per line)."""
    raw_text = request.form.get("bulk_comments", "").strip()

    if not raw_text:
        return render_template("index.html", error="Please paste some comments.")

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

    if not lines:
        return render_template("index.html", error="No valid comments found.")

    results = []
    spam_count = 0
    for line in lines:
        prediction = predict_comment(line)
        results.append({
            "author": "—",
            "text": line,
            "time": "—",
            "label": prediction["label"],
            "is_spam": prediction["is_spam"],
            "confidence": prediction["confidence"],
        })
        if prediction["is_spam"]:
            spam_count += 1

    stats = {
        "total": len(results),
        "spam": spam_count,
        "ham": len(results) - spam_count,
        "spam_pct": round(spam_count / len(results) * 100, 1) if results else 0,
    }

    return render_template(
        "results.html",
        mode="scrape",
        youtube_url="Pasted Comments",
        results=results,
        stats=stats,
    )


if __name__ == "__main__":
    app.run(debug=True)

