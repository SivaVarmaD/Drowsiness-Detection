import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import mediapipe as mp
from scipy.spatial import distance

st.set_page_config(page_title="Drowsiness Detection", layout="centered")
st.title("😴 Driver Drowsiness & Yawning Detection (Streamlit + Mediapipe)")

# Mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Indices for eyes and mouth landmarks from mediapipe (refer documentation)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [78, 308, 13, 14, 312, 82, 87, 317, 88, 95]

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[3])
    B = distance.euclidean(mouth[4], mouth[8])
    C = distance.euclidean(mouth[0], mouth[1])
    return (A + B) / (2.0 * C)

EAR_THRESHOLD = 0.3
MAR_THRESHOLD = 0.6
EYE_FRAMES = 30
MOUTH_FRAMES = 10

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.eye_counter = 0
        self.mouth_counter = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            def get_coords(indices):
                return [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in indices]

            left_eye = get_coords(LEFT_EYE)
            right_eye = get_coords(RIGHT_EYE)
            mouth = get_coords(MOUTH)

            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
            mar = mouth_aspect_ratio(mouth)

            if ear < EAR_THRESHOLD:
                self.eye_counter += 1
                if self.eye_counter > EYE_FRAMES:
                    cv2.putText(img, "DROWSINESS ALERT!", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
            else:
                self.eye_counter = 0

            if mar > MAR_THRESHOLD:
                self.mouth_counter += 1
                if self.mouth_counter > MOUTH_FRAMES:
                    cv2.putText(img, "YAWNING DETECTED!", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 0, 0), 3)
            else:
                self.mouth_counter = 0

        return img

# Streamlit WebRTC
webrtc_streamer(
    key="stream",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)
