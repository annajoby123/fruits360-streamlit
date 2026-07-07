import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Fruits360 Classifier",
    page_icon="🍎",
    layout="wide"
)


# -----------------------------
# Title
# -----------------------------
st.title("🍎 Fruits360 Image Classifier")

st.markdown(
    """
Upload a fruit image and let **MobileNetV2** identify it.

**Model:** MobileNetV2 Transfer Learning
"""
)


# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "fruits360_mobilenetv2_final.keras"
    )

    return model


model = load_model()


# -----------------------------
# Load Classes
# -----------------------------
@st.cache_data
def load_classes():

    with open("class_indices.json", "r") as f:
        class_indices = json.load(f)

    return {v: k for k, v in class_indices.items()}


idx_to_class = load_classes()


# Automatically detect model input size
IMG_SIZE = (
    model.input.shape[1],
    model.input.shape[2]
)


# -----------------------------
# Preprocessing
# -----------------------------
def preprocess(image):

    image = image.convert("RGB")

    image = image.resize(
        IMG_SIZE
    )

    img = np.array(image).astype(
        np.float32
    )

    img = img / 255.0

    img = np.expand_dims(
        img,
        axis=0
    )

    return img



# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:

    st.header("Project Information")

    st.write("Model : MobileNetV2")

    st.write(
        f"Input Size : {IMG_SIZE[0]} × {IMG_SIZE[1]}"
    )

    st.write(
        "Classes :",
        len(idx_to_class)
    )

    st.write(
        "Test Accuracy : 92.78%"
    )


# -----------------------------
# Upload Image
# -----------------------------
uploaded = st.file_uploader(
    "Upload a fruit image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


if uploaded is not None:

    image = Image.open(uploaded)


    col1, col2 = st.columns(2)


    with col1:

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )


    with col2:

        with st.spinner("Predicting..."):


            img = preprocess(image)


            prediction = model.predict(
                img,
                verbose=0
            )[0]


        pred_index = np.argmax(
            prediction
        )


        pred_class = idx_to_class[
            pred_index
        ]


        confidence = (
            prediction[pred_index]
            * 100
        )


        st.success(
            f"Prediction: {pred_class}"
        )


        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )


        st.progress(
            float(
                prediction[pred_index]
            )
        )



    st.divider()


    st.subheader(
        "Top 5 Predictions"
    )


    top5 = np.argsort(
        prediction
    )[::-1][:5]


    for i in top5:

        st.write(
            f"### {idx_to_class[i]}"
        )

        st.progress(
            float(
                prediction[i]
            )
        )

        st.write(
            f"{prediction[i]*100:.2f}%"
        )



st.divider()


st.caption(
    "Developed using TensorFlow, MobileNetV2 and Streamlit."
)
