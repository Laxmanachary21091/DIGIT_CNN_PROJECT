import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load model
model = load_model("digit_model.h5")

st.title("🧠 Handwritten Digit Recognition")

uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('L')
    img = img.resize((28,28))

    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1,28,28,1)

    prediction = model.predict(img_array)

    st.image(img, caption="Uploaded Image")
    st.write("### Predicted Digit:", np.argmax(prediction))
    st.write("Confidence:", np.max(prediction))