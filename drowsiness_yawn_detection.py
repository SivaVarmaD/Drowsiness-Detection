import cv2
import dlib
from scipy.spatial import distance

# --- Eye Aspect Ratio ---
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# --- Mouth Aspect Ratio ---
def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[14], mouth[18])
    B = distance.euclidean(mouth[12], mouth[16])
    C = distance.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)

# --- Constants ---
EAR_THRESHOLD = 0.3
MAR_THRESHOLD = 0.5
CONSEC_FRAMES_EYE = 40
CONSEC_FRAMES_MOUTH = 10

# --- Initialize dlib models ---
predictor_path = r"C:\Users\siva2\Desktop\Driver_Drowsiness_Prediction\shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# --- Video Capture ---
cap = cv2.VideoCapture(0)

# --- Counters ---
eye_frame_count = 0
mouth_frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        coords = [(p.x, p.y) for p in shape.parts()]

        left_eye = coords[42:48]
        right_eye = coords[36:42]
        mouth = coords[48:68]

        # EAR
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        # MAR
        mar = mouth_aspect_ratio(mouth)

        # --- Drowsiness Detection ---
        if avg_ear < EAR_THRESHOLD:
            eye_frame_count += 1
            if eye_frame_count >= CONSEC_FRAMES_EYE:
                cv2.putText(frame, "DROWSINESS ALERT!", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        else:
            eye_frame_count = 0

        # --- Yawning Detection ---
        if mar > MAR_THRESHOLD:
            mouth_frame_count += 1
            if mouth_frame_count >= CONSEC_FRAMES_MOUTH:
                cv2.putText(frame, "YAWNING DETECTED!", (20, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
        else:
            mouth_frame_count = 0

        # --- Draw facial landmarks (optional) ---
        for (x, y) in coords:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    # --- Display ---
    cv2.imshow("Driver Drowsiness & Yawning Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
