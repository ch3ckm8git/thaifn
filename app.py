from flask import Flask, request, jsonify
import joblib
import pandas as pd
import re, string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from flask_cors import CORS
import nltk
nltk.download('stopwords')
nltk.download('wordnet')  # if you use lemmatizer

# Load model
model = joblib.load("./fake_news_model.pkl")

# Preprocessing setup
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_single_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

# Flask app
app = Flask(__name__)
CORS(app) 
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    title = data.get("title", "")
    text = data.get("text", "")
    
    clean_text = clean_single_text(text)
    
    df = pd.DataFrame([{
        "title": title,
        "text_tokens_str": clean_text
    }])

    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0].tolist()

    return jsonify({
        "prediction": int(pred),
        "confidence": proba
    })

if __name__ == "__main__":
    app.run(debug=True)
