# Intelligent Waste Classification using Support Vector Machine (SVM)

A high-accuracy waste classification web application built using traditional Machine Learning with Feature Engineering and Flask.

-----

## 💡 Overview

This project is an **AI-powered web application** designed to classify images of waste into distinct categories (such as Paper, Plastic, Metal, etc.). Unlike modern deep learning methods, this system achieves high accuracy using a **Support Vector Machine (SVM)** classifier combined with sophisticated **handcrafted features**.

### Classification Methodology:

  * **Model:** Support Vector Machine (SVC) with Radial Basis Function (RBF) kernel.
  * **Features:** A combined feature vector of **Color Histograms** and **HOG (Histogram of Oriented Gradients)** to capture both color distribution and texture/shape information.
  * **Web Framework:** **Flask** handles image upload, feature extraction, model inference, and displaying results.

The model is trained on a comprehensive dataset (expected to be named `garbage_classification` or similar) to ensure robust performance.

-----

## ✨ Features

  * **Image Upload:** Users can upload a waste image via a simple web form.
  * **ML-Powered Classification:** Predicts the waste category using the highly efficient **SVM** classifier.
  * **Feature Engineering:** Classification is based on the image's **Color Histogram** and **HOG (texture)** features.
  * **Environmental Insights:** Provides crucial information like the waste's bin **Category** and its **Degradability** status.
  * **Confidence Score:** Displays the model's prediction confidence (derived from `predict_proba` or `decision_function`).
  * **User-Friendly Interface:** Clean, responsive design for easy interaction.

-----

## 💻 Technologies Used

| Category | Technology | Purpose |
| :--- | :--- | :--- |
| **Machine Learning** | **scikit-learn** (SVM, SVC) | The primary ML library for training and inference. |
| **Web Framework** | **Flask** | Python framework for the web application backend. |
| **Image Processing** | **scikit-image** (skimage) | Used for image manipulation (resize, HOG calculation). |
| **Serialization** | **Joblib** | Used to save and load the trained SVM model (`waste_svm_classifier.pkl`). |
| **Dependencies** | NumPy, werkzeug | Data handling and secure file management. |

-----

## 📁 Project Structure

```
.
├── Project files/
│   ├── app.py                     # Flask web application (front-end & inference)
│   ├── train_model.py             # Script for feature extraction & SVM training
│   ├── waste_svm_classifier.pkl   # The trained SVM model file (Joblib format)
│   └── class_names.json           # Mapping of class indices to human-readable names
├── requirements.txt               # Project dependencies
└── README.md
```

*Note: The model expects the training data to be in a directory structure like `garbage_classification/train/[CLASS_NAME]/`.*

-----

## 🛠️ Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/VivekAllu16/waste_management_and_identification.git
    cd waste_management_and_identification
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On macOS/Linux
    venv\Scripts\activate     # On Windows
    ```
3.  **Install dependencies:**
    *(You will need to create a `requirements.txt` file listing all dependencies: `flask`, `scikit-learn`, `scikit-image`, `numpy`, `joblib`, `werkzeug`)*
    ```bash
    pip install -r requirements.txt
    ```
4.  **Model and Class Names:** Ensure `waste_svm_classifier.pkl` and `class_names.json` are present in the root directory alongside `app.py`.

-----

## 🧠 Model Training (Optional)

You only need to run this if you want to retrain the model on your own dataset.

1.  Place your image dataset in the expected structure (e.g., `garbage_classification/train/`).
2.  Run the training script:
    ```bash
    python train_model.py
    ```
    This script will:
      * Load images and extract features (Histogram + HOG).
      * Split the data (80% train, 20% test).
      * Train the RBF-kernel SVM model.
      * Print the **Accuracy, Confusion Matrix, and Classification Report**.
      * Save the final model as `waste_svm_classifier.pkl` and class names as `class_names.json`.

-----

## ▶️ Usage

1.  **Start the Flask application:**
    ```bash
    python Project files/app.py
    ```
2.  **Access the application:**
    Open your web browser and navigate to:
    ```
    http://127.0.0.1:5000/
    ```
3.  **Classify Waste:** Upload an image and view the predicted class, category, degradability, and confidence score.

-----

## 🌿 Environmental Information Mapping

The `app.py` script contains simple keyword logic to map the predicted waste class to a general category and degradability status:

| Predicted Waste Class (Example) | Bin Category | Degradability Status |
| :--- | :--- | :--- |
| Any class containing 'paper' | Paper | Degradable |
| Any class containing 'plastic' | Plastic | Non-degradable |
| Any class containing 'glass' | Glass | Non-degradable |
| Any class containing 'metal' or 'can' | Metal | Non-degradable |
| Any class containing 'bio', 'organic', 'food', 'compost' | Organic | Degradable |
| Others | Unknown | Unknown |

-----

## 🎯 Future Enhancements

  * **Dataset Integration:** Incorporate the full dataset for a more robust model.
  * **Dynamic Mapping:** Implement a more comprehensive and configurable mapping for environmental labels outside of the `app.py` script.
  * **Web Interface Polish:** Improve the `index.html` template for better visualization and user experience.
  * **Cloud Deployment:** Deploy the Flask application to a cloud platform (e.g., AWS, Heroku).
