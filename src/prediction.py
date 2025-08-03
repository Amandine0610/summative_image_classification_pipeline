import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import json

IMG_HEIGHT = 150
IMG_WIDTH = 150
MODEL_PATH = "models/image_classifier_model.h5"
CLASS_INDICES_PATH = "models/class_indices.json"  # Save class indices from training

def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def load_class_indices():
    with open(CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)
    return {v: k for k, v in class_indices.items()}  # Invert to get label from index


def predict_image(img_path):
    model = load_model()
    class_labels = load_class_indices()

    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = class_labels[predicted_class_index]

    confidence = float(np.max(prediction))

    return {
        "predicted_class": predicted_class_label,
        "confidence": round(confidence, 4)
    }
