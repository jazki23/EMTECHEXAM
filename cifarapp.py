
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Load model
model = load_model("cifar10_model.h5")

st.title("🚀 CIFAR-10 Image Classifier")
st.write("Upload an image (32x32 or larger), and this app will classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Resize and preprocess
    image = image.resize((32, 32))
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array)[0]
    top_indices = prediction.argsort()[-3:][::-1]

    st.subheader("Top Predictions:")
    for i in top_indices:
        st.write(f"**{class_names[i]}** — {prediction[i]*100:.2f}%")
