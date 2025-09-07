from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import re, string

# English preprocessing

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.data.path.append("./nltk_data")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_en(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

# Thai preprocessing
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus.common import thai_stopwords
thai_stop = set(thai_stopwords())
thai_punctuations = "ๆฯ“…”‘’—–•"

def clean_thai(text):
    for p in thai_punctuations:
        text = text.replace(p, "")
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    tokens = word_tokenize(text, engine="newmm")
    tokens = [t for t in tokens if t not in thai_stop]
    return " ".join(tokens)

# Load models
model_en = joblib.load("./fake_news_model.pkl")
model_th = joblib.load("./thai_news_model.pkl")

# Flask app
app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict_en():
    data = request.get_json()
    title = data.get("title", "")
    text  = data.get("text", "")
    clean_text = clean_en(text)

    df = pd.DataFrame([{
        "title": clean_en(title),
        "text_tokens_str": clean_text
    }])

    pred = model_en.predict(df)[0]
    proba = model_en.predict_proba(df)[0].tolist()

    return jsonify({"prediction": int(pred), "confidence": proba})

@app.route("/predictthai", methods=["POST"])
def predict_th():
    data = request.get_json()
    title = data.get("title", "")
    text  = data.get("text", "")
    clean_text = clean_thai(text)

    df = pd.DataFrame([{
        "title": clean_thai(title),
        "text": clean_text
    }])

    pred = model_th.predict(df)[0]
    proba = model_th.predict_proba(df)[0].tolist()

    return jsonify({"prediction": int(pred), "confidence": proba})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
