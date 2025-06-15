import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# --- Constants ---
MODEL_DIR = "models"
IMAGE_SIZE = (224, 224)
CLASS_NAMES = open("class_names.txt").read().splitlines()

@st.cache_resource
def load_tflite_model(model_name):
    interpreter = tf.lite.Interpreter(model_path=os.path.join(MODEL_DIR, f"{model_name}.tflite"))
    interpreter.allocate_tensors()
    return interpreter

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize(IMAGE_SIZE)
    image = np.array(image).astype(np.float32) / 255.0  # Normalized
    return np.expand_dims(image, axis=0)  # Add batch dim

def predict_image(interpreter, image_array: np.ndarray):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    top_idx = np.argmax(predictions)
    return CLASS_NAMES[top_idx], predictions[top_idx]

# --- UI ---
st.title("🐾 Animal Classifier with TFLite")

# Load models
model_list = sorted([f.replace(".tflite", "") for f in os.listdir(MODEL_DIR) if f.endswith(".tflite")])
model_name = st.selectbox("Select a TFLite model", model_list)

uploaded_file = st.file_uploader("Upload an animal image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = preprocess_image(image)
    with st.spinner("Loading model and predicting..."):
        interpreter = load_tflite_model(model_name)
        label, confidence = predict_image(interpreter, img_array)

    st.success("Prediction complete!")
    st.markdown(f"### 🐶 Predicted Animal: `{label}`")
    st.markdown(f"**Confidence:** `{confidence:.2%}`")
