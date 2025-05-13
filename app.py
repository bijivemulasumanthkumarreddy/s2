'''import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the model in Keras format
model = load_model("Emotion_model_clean.keras")

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Streamlit UI
st.set_page_config(page_title="Real-Time Emotion Detector", layout="centered")
st.title("🎭 Real-Time Facial Expression Detection")
emotion_placeholder = st.empty()

# Processor for video frames
class EmotionProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float32") / 255.0
            roi = np.expand_dims(roi, axis=(0, -1))
            preds = model.predict(roi, verbose=0)
            emotion = emotion_labels[np.argmax(preds)]
            emotion_placeholder.markdown(f"### 😃 Detected Emotion: **{emotion}**")
            break  # only process first face

        return frame

# Start webcam streamer
webrtc_streamer(key="emotion", video_processor_factory=EmotionProcessor)'''
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the model in Keras format
model = load_model("Emotion_model.keras")

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Streamlit UI
st.set_page_config(page_title="Real-Time Emotion Detector", layout="centered")
st.title("🎭 Real-Time Facial Expression Detection")
emotion_placeholder = st.empty()

# Processor for video frames
class EmotionProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float32") / 255.0
            roi = np.expand_dims(roi, axis=(0, -1))
            preds = model.predict(roi, verbose=0)
            emotion = emotion_labels[np.argmax(preds)]
            emotion_placeholder.markdown(f"### 😃 Detected Emotion: **{emotion}**")
            break  # only process first face

        return frame

# Start webcam streamer
webrtc_streamer(key="emotion", video_processor_factory=EmotionProcessor)
