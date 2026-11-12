
from pathlib import Path
from collections import deque, Counter

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

from hand_features import extract_hand_features

# ==========================
# PATHS
# ==========================

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "hand_model"

MODEL_FILE = MODEL_DIR / "sign_model.keras"
LABELS_FILE = MODEL_DIR / "labels.npy"

# ==========================
# SETTINGS
# ==========================

CONFIDENCE_THRESHOLD = 75.0
HISTORY_SIZE = 10
MIN_STABLE_FRAMES = 6

# ==========================
# LOAD MODEL
# ==========================

model = tf.keras.models.load_model(
    str(MODEL_FILE)
)

labels = np.load(
    LABELS_FILE,
    allow_pickle=True
)

if model.input_shape[-1] != 126:
    raise ValueError(
        "Expected a model with 126 input features"
    )

print("Hand model loaded successfully")
print("Available signs:", labels)

# ==========================
# MEDIAPIPE
# ==========================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Separate detector for drawing and prediction.
# This preserves the shared extractor unchanged.
draw_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==========================
# CAMERA
# ==========================

cap = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)

if not cap.isOpened():
    detector.close()
    draw_detector.close()
    raise RuntimeError("Camera not available")

# ==========================
# HISTORY
# ==========================

history = deque(maxlen=HISTORY_SIZE)

# ==========================
# MAIN LOOP
# ==========================

try:
    while True:

        success, frame = cap.read()

        if not success:
            continue

        frame = cv2.flip(frame, 1)

        # Shared extractor expects the same orientation
        # used by the Kaggle converter.
        features = extract_hand_features(
            frame,
            detector,
            flip=False
        )

        display_frame = frame.copy()

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        draw_results = draw_detector.process(rgb)

        if draw_results.multi_hand_landmarks:
            for hand_landmarks in (
                draw_results.multi_hand_landmarks
            ):
                mp_draw.draw_landmarks(
                    display_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        prediction = "Waiting..."
        confidence = 0.0

        if features is None:
            history.clear()
            prediction = "No hand detected"

        else:
            x = features.reshape(1, 126)

            probabilities = model.predict(
                x,
                verbose=0
            )[0]

            class_id = int(
                np.argmax(probabilities)
            )

            confidence = float(
                probabilities[class_id] * 100
            )

            if confidence >= CONFIDENCE_THRESHOLD:

                sign = str(labels[class_id])
                history.append(sign)

                counts = Counter(history)

                stable_sign, count = (
                    counts.most_common(1)[0]
                )

                if count >= MIN_STABLE_FRAMES:
                    prediction = stable_sign

            else:
                history.clear()
                prediction = "Low confidence"

        # ==========================
        # DISPLAY
        # ==========================

        cv2.rectangle(
            display_frame,
            (0, 0),
            (display_frame.shape[1], 100),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            display_frame,
            f"SIGN: {prediction}",
            (20, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display_frame,
            f"CONFIDENCE: {confidence:.2f}%",
            (20, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "SignAI - Hand Sign Recognition",
            display_frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

finally:
    cap.release()
    detector.close()
    draw_detector.close()
    cv2.destroyAllWindows()
