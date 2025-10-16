"""
app.py
------
Flask web app to classify waste images using high-accuracy SVM model.
Features: Color histogram + HOG (texture)
"""

from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
import joblib
import json

app = Flask(__name__)
# ensure uploads folder
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -----------------------------
# Load Model & Class Names
# -----------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "waste_svm_classifier.pkl")
CLASS_NAMES_JSON = os.path.join(os.path.dirname(__file__), "class_names.json")

model = joblib.load(MODEL_PATH)

with open(CLASS_NAMES_JSON, "r") as f:
    class_names = json.load(f)

IMG_SIZE = (64, 64)

# -----------------------------
# Feature Extraction Function
# -----------------------------
def extract_features(image_path):
    img = imread(image_path)
    img = resize(img, IMG_SIZE, anti_aliasing=True)
    
    # Color histogram
    hist = np.histogram(img, bins=32, range=(0, 1))[0]
    
    # HOG (texture)
    gray = np.mean(img, axis=2) if img.ndim == 3 else img
    hog_features = hog(gray, pixels_per_cell=(8,8), cells_per_block=(2,2), feature_vector=True)
    
    features = np.concatenate([hist, hog_features])
    return features.reshape(1, -1)

# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # match input name in template: "image"
        file = request.files.get("image")
        if not file or file.filename == "":
            return render_template("index.html", filename=None, predicted_class=None, category=None, degradable=None, confidence=None)

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_DIR, filename)
        file.save(filepath)

        # Extract features & predict
        features = extract_features(filepath)
        prediction = model.predict(features)[0]  # integer class index
        predicted_label = class_names[int(prediction)]

        # Map predicted label to bin category & degradability (simple keyword-based fallback)
        def get_bin_info(label):
            l = label.lower()
            if 'paper' in l:
                return 'Paper', 'Degradable'
            if 'plastic' in l:
                return 'Plastic', 'Non-degradable'
            if 'glass' in l:
                return 'Glass', 'Non-degradable'
            if 'metal' in l or 'can' in l:
                return 'Metal', 'Non-degradable'
            if any(k in l for k in ('bio', 'organic', 'food', 'compost')):
                return 'Organic', 'Degradable'
            return 'Unknown', 'Unknown'

        category, degradable = get_bin_info(predicted_label)

        # Compute confidence as numeric percent (float). Prefer predict_proba, fall back to decision_function.
        confidence = None
        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(features)[0]
                conf_val = probs[int(prediction)]
                confidence = conf_val * 100.0
            elif hasattr(model, "decision_function"):
                scores = model.decision_function(features)
                if np.ndim(scores) == 1:
                    score = float(scores[0])
                    conf_val = 1.0 / (1.0 + np.exp(-score))
                else:
                    scores = scores[0]
                    exp_scores = np.exp(scores - np.max(scores))
                    probs = exp_scores / np.sum(exp_scores)
                    conf_val = probs[int(prediction)]
                confidence = conf_val * 100.0
        except Exception:
            confidence = None

        return render_template(
            "index.html",
            filename=filename,
            predicted_class=predicted_label,
            category=category,
            degradable=degradable,
            confidence=confidence
        )

    return render_template("index.html", filename=None, predicted_class=None, category=None, degradable=None, confidence=None)

# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
