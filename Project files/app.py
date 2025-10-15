from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = tf.keras.models.load_model('waste_classifier_model.keras')
img_size = (224, 224)
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

category_mapping = {
    'cardboard': '♻️ Recyclable',
    'glass': '♻️ Recyclable',
    'metal': '♻️ Recyclable',
    'paper': '♻️ Recyclable',
    'plastic': '♻️ Recyclable',
    'trash': '❌ Not Recyclable'
}

degradable_mapping = {
    'cardboard': '🌱 Degradable',
    'paper': '🌱 Degradable',
    'glass': '❌ Non-Degradable',
    'metal': '❌ Non-Degradable',
    'plastic': '❌ Non-Degradable',
    'trash': '❌ Non-Degradable'
}

def predict_image(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=img_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) / 255.0
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)
    return predicted_class, category_mapping[predicted_class], degradable_mapping[predicted_class], round(confidence * 100, 2)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        image = request.files['image']
        if image:
            filename = secure_filename(image.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(filepath)

            predicted_class, category, degradable, confidence = predict_image(filepath)

            return render_template("index.html", filename=filename,
                                   predicted_class=predicted_class,
                                   category=category,
                                   degradable=degradable,
                                   confidence=confidence)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
