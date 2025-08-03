import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import os
from zipfile import ZipFile
from io import BytesIO

# Constants
IMG_HEIGHT = 150
IMG_WIDTH = 150
MODEL_PATH = "models/saved_model/best_model.keras"
CLASS_INDICES_PATH = "models/class_indices.json"

@st.cache(allow_output_mutation=True)
def load_model_and_classes():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)
    # invert the dict to get labels by index
    labels = {v: k for k, v in class_indices.items()}
    return model, labels

def preprocess_image(img):
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # batch dimension
    return img_array

def predict(image_data, model, labels):
    preds = model.predict(image_data)
    pred_class = np.argmax(preds, axis=1)[0]
    confidence = preds[0][pred_class]
    return labels[pred_class], confidence

def main():
    st.title("Image Classification with Retrain Option")
    
    model, labels = load_model_and_classes()
    
    st.sidebar.header("Upload Image for Prediction")
    uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        img = image.load_img(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        img_preprocessed = preprocess_image(img)
        pred_label, confidence = predict(img_preprocessed, model, labels)
        st.write(f"**Prediction:** {pred_label}")
        st.write(f"**Confidence:** {confidence:.2f}")
    
    st.sidebar.header("Retrain Model")
    uploaded_zip = st.sidebar.file_uploader("Upload ZIP of images for retraining", type=["zip"])
    
    if uploaded_zip is not None:
        st.info("Received zip file. Starting retraining...")
        
        # Extract and process zip file
        with ZipFile(BytesIO(uploaded_zip.read())) as zip_ref:
            zip_ref.extractall("retrain_data")  # You should add logic to use this data
        
        # Call your retraining function here
        st.warning("Retraining functionality is not yet implemented in this demo.")
        # e.g., retrain_model("retrain_data/train", "retrain_data/test")
        
    st.sidebar.header("Model Info")
    st.write("Model predicts classes:")
    st.write(list(labels.values()))

if __name__ == "__main__":
    main()
