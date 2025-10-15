import os
import numpy as np
import json
from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')

with open('class_names.json', 'r') as f:
    class_names = json.load(f)

category_mapping = {
    'battery': '❌ Not Recyclable',
    'biological': '🌱 Compostable',
    'brown-glass': '♻️ Recyclable',
    'cardboard': '♻️ Recyclable',
    'clothes': '♻️ Reusable',
    'footwear': '♻️ Reusable',
    'green-glass': '♻️ Recyclable',
    'metal': '♻️ Recyclable',
    'paper': '♻️ Recyclable',
    'plastic': '♻️ Recyclable',
    'trash': '❌ Not Recyclable',
    'white-glass': '♻️ Recyclable'
}

degradable_mapping = {
    'battery': '❌ Non-Degradable',
    'biological': '🌱 Degradable',
    'brown-glass': '❌ Non-Degradable',
    'cardboard': '🌱 Degradable',
    'clothes': '🌱 Degradable',
    'footwear': '🌱 Degradable',
    'green-glass': '❌ Non-Degradable',
    'metal': '❌ Non-Degradable',
    'paper': '🌱 Degradable',
    'plastic': '❌ Non-Degradable',
    'trash': '❌ Non-Degradable',
    'white-glass': '❌ Non-Degradable'
}

IMG_SIZE = (224, 224)
model = load_model('waste_classifier_model.keras')

def predict_image(image_path):
    img = load_img(image_path, target_size=IMG_SIZE)
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    preds = model.predict(arr)
    idx = int(np.argmax(preds, axis=1)[0])
    label = class_names[idx]
    conf = float(np.max(preds))
    return label, category_mapping.get(label, ''), degradable_mapping.get(label, ''), round(conf*100, 2)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        label, category, degradable, conf = predict_image(save_path)
        return render_template('index.html', filename=filename, predicted_class=label, category=category, degradable=degradable, confidence=conf)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)