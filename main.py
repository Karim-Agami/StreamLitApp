import streamlit as st
import torch
from PIL import Image
import numpy as np

# Load the YOLOv5 model
model_path = 'yolov5/yolov5s.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# Streamlit application
st.title("YOLOv5 Object Detection")
st.write("Upload an image to detect objects using YOLOv5")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button("Analyse Image"):
        st.write("Detecting objects...")
        img = np.array(image)
        results = model(img)

        # Extract the names of detected objects
        detected_objects = results.pandas().xyxy[0]['name']

        st.write("Detected objects:")
        for obj in detected_objects:
            st.write(f"- {obj}")
