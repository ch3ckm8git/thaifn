from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import re, string
import os


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
CORS(app, resources={r"/*": {"origins": ["http://127.0.0.1:5500", "http://localhost:5500", "*"]}})


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
@app.route("/health")
def health():
    return jsonify({"ok": True})
    
@app.after_request
def apply_cors_headers(response):
    origin = request.headers.get("Origin")
    if origin:
        response.headers["Access-Control-Allow-Origin"] = origin
    else:
        response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response
    
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
