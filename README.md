# Transforming Waste Management with Transfer Learning

An Intelligent Waste Classification Web Application Using TensorFlow and Flask

---

## Overview

This project is an AI-powered web application that classifies waste images into categories such as cardboard, glass, metal, paper, plastic, trash, and more. It uses transfer learning with MobileNetV2 for accurate waste identification and provides environmental information like recyclability and degradability.

The model is trained on the `garbage_classification` dataset, which contains 15,000+ labeled images across multiple waste categories.

---

## Features

- **Image Upload:** Upload waste images via a simple web form.
- **Automated Classification:** Predicts the waste category using a fine-tuned MobileNetV2 model.
- **Environmental Labels:** Shows recyclability and degradability for each prediction.
- **Confidence Score:** Displays prediction confidence as a percentage.
- **Image Preview:** See the uploaded image alongside results.
- **User-Friendly Interface:** Clean, responsive design for easy use.

---

## Technologies Used

- **TensorFlow 2.x:** Deep learning framework for model training and inference.
- **MobileNetV2:** Pre-trained CNN model for transfer learning.
- **Flask:** Web framework for the frontend and backend.
- **NumPy:** Numerical computations.
- **scikit-learn:** Evaluation metrics (optional).
- **Matplotlib:** Training visualization (optional).

---

## Project Structure

```
README.md
Project files/
    app.py
    twmcode.ipynb
    waste_classifier_model.h5
    garbage_classification/
        battery/
        biological/
        brown-glass/
        cardboard/
        clothes/
        green-glass/
        metal/
        paper/
        plastic/
        shoes/
        trash/
        ...
    static/
        bg.jpg
        uploads/
    templates/
        index.html
Video Demo/
    Video-Demo.mp4
    readme.md
```

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/waste-management-app.git
   cd waste-management-app
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate   # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure the trained model file `waste_classifier_model.h5` is in the `Project files/` directory.**

5. **Create the uploads folder if it doesn't exist:**
   ```bash
   mkdir static\uploads
   ```

---

## Usage

1. **Start the Flask application:**
   ```bash
   python Project files/app.py
   ```

2. **Open your browser and go to:**
   ```
   http://127.0.0.1:5000/
   ```

3. **Upload a waste image and view the prediction, recyclability, degradability, and confidence score.**

---

## How It Works

- The Flask app (`app.py`) manages image uploads and serves the web interface.
- Uploaded images are stored in `static/uploads`.
- The TensorFlow model (`waste_classifier_model.h5`) predicts the waste category and confidence.
- The app displays environmental labels and the uploaded image.

---

## Model Training

- The model is trained using transfer learning with MobileNetV2 in `twmcode.ipynb`.
- The training uses the `garbage_classification` dataset with over 15,000 images.
- Data augmentation and fine-tuning are applied for improved accuracy.
- The trained model is saved as `waste_classifier_model.h5`.

---

## Environment Labels

| Waste Type   | Recyclability      | Degradability      |
|--------------|--------------------|--------------------|
| cardboard    | ♻️ Recyclable      | 🌱 Degradable      |
| glass        | ♻️ Recyclable      | ❌ Non-Degradable  |
| metal        | ♻️ Recyclable      | ❌ Non-Degradable  |
| paper        | ♻️ Recyclable      | 🌱 Degradable      |
| plastic      | ♻️ Recyclable      | ❌ Non-Degradable  |
| trash        | ❌ Not Recyclable   | ❌ Non-Degradable  |
| ...          | ...                | ...                |

---

## Future Enhancements

- Drag-and-drop image upload
- Batch image classification
- User submission tracking
- Cloud deployment
- Multi-language support

---

## Demo

See [Video Demo/Video-Demo.mp4](Video%20Demo/Video-Demo.mp4) for a walkthrough
