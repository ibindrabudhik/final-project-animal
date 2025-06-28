import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import (
    EfficientNetB0, EfficientNetB1, EfficientNetB2,
    EfficientNetB3, EfficientNetB4, EfficientNetB5,
    ResNet50, ResNet101
)
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess

# ===== Model & Preprocessing Setup =====
MODEL_PATHS = {
    "EfficientNetB0": "models/EfficientNetB0.h5"
    # "EfficientNetB1": "models/EfficientNetB1.h5",
    # "EfficientNetB2": "models/EfficientNetB2.h5",
    # "EfficientNetB3": "models/efficientnetb3.h5",
    # "EfficientNetB4": "models/efficientnetb4.h5",
    # "EfficientNetB5": "models/efficientnetb5.h5",
    # "ResNet50": "models/resnet50.h5",
    # "ResNet101": "models/resnet101.h5"
}

PREPROCESS_MAP = {
    "EfficientNetB0": efficientnet_preprocess
    # "EfficientNetB1": efficientnet_preprocess,
    # "EfficientNetB2": efficientnet_preprocess,
    # "EfficientNetB3": efficientnet_preprocess,
    # "EfficientNetB4": efficientnet_preprocess,
    # "EfficientNetB5": efficientnet_preprocess,
    # "ResNet50": resnet_preprocess,
    # "ResNet101": resnet_preprocess,
}

IMAGE_SIZE = (224, 224)

# ✅ Updated class labels
CLASS_NAMES = [
    "butterfly", "cat", "chicken", "cow",
    "dog", "elephant", "horse", "ragno", "sheep", "squirrel"
]

@st.cache_resource
def load_model(model_name):
    return tf.keras.models.load_model(MODEL_PATHS[model_name])

def preprocess_image(image: Image.Image, model_name: str):
    image = image.resize(IMAGE_SIZE)
    image_array = np.array(image)
    if image_array.shape[-1] == 4:
        image_array = image_array[..., :3]  # remove alpha channel if present
    image_array = np.expand_dims(image_array, axis=0)
    preprocess = PREPROCESS_MAP[model_name]
    return preprocess(image_array)

def predict_image(model, image_array, model_name):
    predictions = model.predict(image_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = float(np.max(predictions)) * 100
    return predicted_class, confidence

# ===== Streamlit UI =====
st.set_page_config(page_title="Animal Image Classifier", layout="centered")

st.title("🐾 Animal Image Classifier")
st.markdown("This study have trained several EfficientNet model and ResNet Model on Animal10 dataset")
st.markdown("""
Here are animal classes that you can use:

1. Dog 🐶  
2. Horse 🐴  
3. Elephant 🐘  
4. Butterfly 🦋  
5. Chicken 🐔  
6. Cat 🐱  
7. Cow 🐮  
8. Spider 🕷️  
9. Squirrel 🐿️  
10. Sheep 🐑
""")
st.markdown("Dataset Link: https://www.kaggle.com/datasets/alessiocorrado99/animals10")
st.markdown("Upload an animal photo and choose a model to classify it.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
model_name = st.selectbox("Select a model", list(MODEL_PATHS.keys()))

if uploaded_file and model_name:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Loading model and predicting..."):
        model = load_model(model_name)
        image_array = preprocess_image(image, model_name)
        label, confidence = predict_image(model, image_array, model_name)

    st.success(f"**Prediction:** {label} ({confidence:.2f}%)")
    st.markdown(f"Model used: **{model_name}**")
