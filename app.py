import streamlit as st
import numpy as np
import os
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# --- Streamlit UI ---
st.set_page_config(page_title="Animal Classifier", layout="centered")

# --- Constants ---
IMAGE_SIZE = (224, 224)
MODEL_PATH = "best_model.keras"
GOOGLE_DRIVE_FILE_ID = "1KyP5xHClTGxZRCHm0Ha7OvT1zVD533E-"
CLASS_NAMES = ['Bear', 'Bird', 'Cat', 'Cow', 'Deer', 'Dog', 'Dolphin', 'Elephant',
               'Giraffe', 'Horse', 'Kangaroo', 'Lion', 'Panda', 'Tiger', 'Zebra']

# --- Download model ---
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading model..."):
            url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)

# --- Load Model ---
download_model()
model = load_model(MODEL_PATH, compile=False)

st.title("🧠 Animal Image Classifier")
st.markdown("Upload an animal image and let the model identify the species!")

# --- Upload Section ---
uploaded_file = st.file_uploader("📤 Upload an Image", type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'gif', 'webp'])

if uploaded_file is not None:
    # Open and preprocess image
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize(IMAGE_SIZE)
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    prediction = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = float(np.max(prediction))
    prob_dict = {CLASS_NAMES[i]: float(prediction[0][i]) for i in range(len(CLASS_NAMES))}

    # --- Display: Input and Output in a Compact Layout ---
    st.markdown("### 🔍 Prediction Results")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(img, caption='📷 Image Preview', width=150)

    with col2:
        st.success(f"🎯 **Animal:** {predicted_class}")
        st.info(f"📊 **Confidence:** {confidence:.2f}")

    st.markdown("---")
    st.subheader("📈 Class Probabilities")
    st.bar_chart(prob_dict)

else:
    st.info("🖼️ Please upload an image to get started.")
