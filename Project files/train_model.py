"""
train_model.py
---------------
Trains a MobileNetV2-based waste classifier on the 'garbage_classification/' dataset.

Expected folder structure:
garbage_classification/
    train/
        battery/
        biological/
        ...
    val/            (optional)
        battery/
        biological/
        ...
"""

import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -----------------------------
# Configuration
# -----------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'garbage_classification')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
MODEL_PATH = os.path.join(BASE_DIR, 'waste_classifier_model.keras')
CLASS_NAMES_JSON = os.path.join(BASE_DIR, 'class_names.json')

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
AUTOTUNE = tf.data.AUTOTUNE

# -----------------------------
# Load Datasets
# -----------------------------
print("📂 Loading datasets...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

class_names = train_ds.class_names
print("✅ Detected classes:", class_names)

# Save class names for Flask app
with open(CLASS_NAMES_JSON, "w") as f:
    json.dump(class_names, f, indent=2)

if os.path.exists(VAL_DIR):
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        labels="inferred",
        label_mode="categorical",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
else:
    print("⚠️ No val folder found — creating 10% split from train data.")
    total_batches = tf.data.experimental.cardinality(train_ds).numpy()
    val_batches = max(1, int(0.1 * total_batches))
    val_ds = train_ds.take(val_batches)
    train_ds = train_ds.skip(val_batches)

# -----------------------------
# Preprocessing
# -----------------------------
def preprocess(image, label):
    return preprocess_input(image), label

train_ds = train_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(preprocess, num_parallel_calls=AUTOTUNE)

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# -----------------------------
# Model Definition
# -----------------------------
print("🧠 Building model...")

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=IMG_SIZE + (3,)
)
base_model.trainable = False  # Freeze base for transfer learning

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# Training
# -----------------------------
print("🚀 Training started...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# -----------------------------
# Fine-tuning (Optional Boost)
# -----------------------------
print("🎯 Fine-tuning last layers...")
base_model.trainable = True
for layer in base_model.layers[:-40]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

fine_tune_epochs = 5
model.fit(train_ds, validation_data=val_ds, epochs=fine_tune_epochs)

# -----------------------------
# Save Model
# -----------------------------
model.save(MODEL_PATH)
print(f"✅ Model saved at: {MODEL_PATH}")
print(f"✅ Class names saved at: {CLASS_NAMES_JSON}")
