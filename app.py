import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image

# Load model and labels
model = load_model("Emotion_model_clean.keras")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

st.title("📸 Emotion Detection from Photo")
img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    image = Image.open(img_file_buffer).convert("RGB")
    img_np = np.array(image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        st.warning("No face detected.")
    else:
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = face.astype("float32") / 255.0
            face = np.expand_dims(face, axis=(0, -1))
            pred = model.predict(face, verbose=0)
            emotion = emotion_labels[np.argmax(pred)]
            st.success(f"Detected Emotion: **{emotion}**")
            break

"""
Developed by Sumanth Kumar Reddy Bijivemula
Here, 7 facial Expressions can be detected
"""
