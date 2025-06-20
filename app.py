import cv2
import dlib
import av
import streamlit as st
from scipy.spatial import distance
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load shape predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()

# Thresholds
EAR_THRESHOLD = 0.3
MAR_THRESHOLD = 0.5
CONSEC_FRAMES_EYE = 40
CONSEC_FRAMES_MOUTH = 10

# EAR function
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# MAR function
def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[14], mouth[18])
    B = distance.euclidean(mouth[12], mouth[16])
    C = distance.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.eye_frame_count = 0
        self.mouth_frame_count = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            shape = predictor(gray, face)
            coords = [(p.x, p.y) for p in shape.parts()]
            left_eye = coords[42:48]
            right_eye = coords[36:42]
            mouth = coords[48:68]

            # EAR & MAR
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0
            mar = mouth_aspect_ratio(mouth)

            if avg_ear < EAR_THRESHOLD:
                self.eye_frame_count += 1
                if self.eye_frame_count >= CONSEC_FRAMES_EYE:
                    cv2.putText(img, "DROWSINESS ALERT!", (20, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            else:
                self.eye_frame_count = 0

            if mar > MAR_THRESHOLD:
                self.mouth_frame_count += 1
                if self.mouth_frame_count >= CONSEC_FRAMES_MOUTH:
                    cv2.putText(img, "YAWNING DETECTED!", (20, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
            else:
                self.mouth_frame_count = 0

        return img

# Streamlit UI
st.title("🚗 Driver Drowsiness & Yawning Detection")
st.markdown("This app detects driver drowsiness and yawning in real-time using webcam and dlib facial landmarks.")

webrtc_streamer(
    key="driver-alert",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)
