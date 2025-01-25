import streamlit as st
from PIL import Image
import cv2
import keras_ocr
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
pipeline = keras_ocr.pipeline.Pipeline()
model = YOLO(r"runs\detect\train\weights\best.pt")

st.write("Upload an image file from your computer")


uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    prediction_groups = pipeline.recognize([image_rgb])
    fig_ocr, ax_ocr = plt.subplots(figsize=(8, 8))
    keras_ocr.tools.drawAnnotations(image_rgb, prediction_groups[0], ax=ax_ocr)
    ax_ocr.set_title("Keras OCR Result with Bounding Boxes")
    ax_ocr.axis('off')
    st.pyplot(fig_ocr)
    results = model.predict(image_cv, save=False, imgsz=640, conf=0.5)     
    fig_yolo, ax_yolo = plt.subplots(figsize=(8, 8))
    if results: 
        for result in results:
            result.save(filename="result.jpg")  
        annotated_image = Image.open("result.jpg")
        st.image(annotated_image, caption="YOLOv8 Object Detections with bounding box", use_column_width=True)
        ax_yolo.imshow(annotated_image)
    else:
        st.image(image, caption="No objects detected, displaying original image.", use_column_width=True)
        ax_yolo.imshow(image)  

