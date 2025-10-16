"""
train_svm_model.py
-----------------
Trains a high-accuracy waste classifier using SVM (from course syllabus).

Method:
- Support Vector Machine (Non-linear, RBF kernel)
- Features: Color histogram + HOG (texture)
- Preprocessing: Scaling
- Evaluation: Accuracy, Confusion Matrix, Classification Report
"""

import os
import json
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import make_pipeline
import joblib

# -----------------------------
# Configuration
# -----------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'garbage_classification', 'train')
MODEL_PATH = os.path.join(BASE_DIR, 'waste_svm_classifier.pkl')
CLASS_NAMES_JSON = os.path.join(BASE_DIR, 'class_names.json')

IMG_SIZE = (64, 64)

# -----------------------------
# Load Dataset & Extract Features
# -----------------------------
print("📂 Loading dataset and extracting features...")

X, y, class_names = [], [], []

for idx, folder in enumerate(sorted(os.listdir(DATA_DIR))):
    folder_path = os.path.join(DATA_DIR, folder)
    if not os.path.isdir(folder_path):
        continue
    class_names.append(folder)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        try:
            img = imread(img_path)
            img = resize(img, IMG_SIZE, anti_aliasing=True)
            
            # Feature 1: Color histogram
            hist = np.histogram(img, bins=32, range=(0, 1))[0]
            
            # Feature 2: HOG (texture)
            gray = np.mean(img, axis=2) if img.ndim == 3 else img
            hog_features = hog(gray, pixels_per_cell=(8,8), cells_per_block=(2,2), feature_vector=True)
            
            features = np.concatenate([hist, hog_features])
            X.append(features)
            y.append(idx)
        except:
            continue

X = np.array(X)
y = np.array(y)

print(f"✅ Loaded {len(X)} samples across {len(class_names)} classes.")

# Save class names
with open(CLASS_NAMES_JSON, "w") as f:
    json.dump(class_names, f, indent=2)

# -----------------------------
# Split Dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Train SVM (Non-linear, RBF)
# -----------------------------
print("🧠 Training SVM model (RBF Kernel)...")

model = make_pipeline(
    StandardScaler(),
    SVC(kernel='rbf', C=10, gamma='scale', decision_function_shape='ovr')
)

model.fit(X_train, y_train)

# -----------------------------
# Evaluation
# -----------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {acc*100:.2f}%")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=class_names))

# -----------------------------
# Save Model
# -----------------------------
joblib.dump(model, MODEL_PATH)
print(f"💾 Model saved at: {MODEL_PATH}")
