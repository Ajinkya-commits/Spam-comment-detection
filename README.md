# Spam Comment Detector

A complete Machine Learning web application designed to classify comments as **Spam** or **Not Spam (Ham)**. It uses a custom-trained Natural Language Processing (NLP) model to analyze comment text, and provides an intuitive web interface for predictions. 

## 🚀 Features

- **Single Comment Analysis**: Manually type or paste a comment to get a real-time spam prediction with confidence scores.
- **YouTube Video Scraping**: Provide a YouTube video URL, and the application will automatically scrape up to 200 comments from the video and classify them in bulk.
- **Bulk Pasted Comments**: Paste multiple comments (one per line) to analyze them all at once.
- **Custom Trained Model**: Uses a Multinomial Naive Bayes classifier trained on a dataset of real YouTube comments.
- **Responsive Web Interface**: Built with Flask and beautifully styled with HTML/CSS.

## 🛠️ Technology Stack

- **Backend**: Python, Flask
- **Machine Learning**: scikit-learn (Multinomial Naive Bayes, TF-IDF Vectorization)
- **Data Manipulation**: pandas, numpy
- **Web Scraping**: Custom script using `requests` and Python `re` module (No API key required)
- **Frontend**: HTML5, CSS3, Jinja2 Templates

## 📂 Project Structure

```text
Spam-Comment-Detector/
│
├── app.py               # Main Flask application
├── scraper.py           # Custom YouTube comment scraper
├── train.py             # Script to train the ML model
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
│
├── data/                # UCI YouTube Spam Collection CSV files
│   ├── Youtube01-Psy.csv
│   ├── Youtube02-KatyPerry.csv
│   └── ...
│
├── models/              # Serialized ML models (Generated after training)
│   ├── spam_classifier.pkl
│   └── tfidf_vectorizer.pkl
│
├── static/              # CSS files and static assets
│   └── style.css
│
└── templates/           # HTML templates for the Flask app
    ├── index.html       # Input form page
    └── results.html     # Prediction results page
```

## 📊 The Model & Data

### The Model
The machine learning model working under the hood is a **Multinomial Naive Bayes Classifier**. This algorithm is highly effective for document classification and text analysis. By using a `TfidfVectorizer`, the raw text comments are transformed into numerical feature vectors that the Naive Bayes model can accurately process, allowing it to predict whether the text pattern resembles historical spam.

### The Data
The model is trained on the [YouTube Spam Collection Data Set](https://archive.ics.uci.edu/ml/datasets/YouTube+Spam+Collection) from the UCI Machine Learning Repository. This dataset consists of real comments from five highly popular YouTube videos (by artists like Eminem, Katy Perry, LMFAO, Psy, and Shakira). 

## ⚙️ Setup & Installation

Follow these steps to set up the project locally.

### 1. Clone the repository
```bash
git clone https://github.com/your-username/Spam-Comment-Detector.git
cd Spam-Comment-Detector
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
```
Activate the virtual environment:
- **Windows**: `venv\Scripts\activate`
- **Mac/Linux**: `source venv/bin/activate`

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Model
The serialized model files are necessary for the app to run. Generate them by running the training script:
```bash
python train.py
```
*This will process the datasets in the `data/` folder and generate `spam_classifier.pkl` and `tfidf_vectorizer.pkl` inside the `models/` directory.*

## 💻 Usage

To start the web application, simply run:

```bash
python app.py
```

The application will launch on your local server, typically at `http://127.0.0.1:5000/`.

- Open this URL in your web browser.
- Select your preferred mode: **Detect Single Comment**, **Scrape Video**, or **Paste Comments**.
- Input the text or the YouTube URL and click **Analyze**.
