import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# ==========================
# Load Model & Label Classes
# ==========================
model = load_model("asl_keypoints_model.h5")
label_classes = np.load("label_classes.npy", allow_pickle=True)

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

# ==========================
# Predict Letter Function
# ==========================
def predict_letter(landmarks):
    # Convert list → array
    landmarks = np.array(landmarks).flatten()

    # Reshape for model
    predictions = model.predict(landmarks.reshape(1, -1), verbose=0)
    letter = label_classes[np.argmax(predictions)]
    confidence = np.max(predictions)

    return letter, confidence


# ==========================
# Capture Webcam
# ==========================
cap = cv2.VideoCapture(0)

print("📷 Camera started... Press Q to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert color → RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    h, w, _ = frame.shape

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmark coordinates
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])

            # Predict letter
            letter, conf = predict_letter(landmarks)

            # Display on screen
            text = f"{letter} ({conf*100:.1f}%)"
            cv2.putText(frame, text, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

    # Show frame
    cv2.imshow("ASL Interpreter", frame)

    # Quit key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
