import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Fruit Classifier",
    page_icon="🍎",
    layout="centered"
)

st.title("🍎 Fruits360 Image Classifier")
st.write("Upload a fruit image and the MobileNetV2 model will predict its class.")

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("fruits360_mobilenetv2_final.keras")

model = load_model()

# -----------------------------
# Load Class Labels
# -----------------------------
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

idx_to_class = {v: k for k, v in class_indices.items()}

IMG_SIZE = (100, 100)

# -----------------------------
# Image Preprocessing
# -----------------------------
def preprocess_image(image):

    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    img_array = np.array(image).astype("float32")

    # If your training used rescale=1./255
    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Predicting..."):

        img = preprocess_image(image)

        prediction = model.predict(img, verbose=0)

        predicted_index = np.argmax(prediction)

        confidence = prediction[0][predicted_index] * 100

        predicted_class = idx_to_class[predicted_index]

    st.success(f"Prediction: **{predicted_class}**")

    st.write(f"Confidence: **{confidence:.2f}%**")

    # -------------------------
    # Top 5 Predictions
    # -------------------------
    st.subheader("Top 5 Predictions")

    top5 = np.argsort(prediction[0])[::-1][:5]

    for idx in top5:

        st.write(
            f"{idx_to_class[idx]} : {prediction[0][idx]*100:.2f}%"
        )
