
import json
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

from movement_features import (
    extract_features,
    SEQUENCE_LENGTH,
    FEATURE_COUNT,
)

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = (
    BASE_DIR / "models" / "movement_model_candidate"
)

MODEL_PATH = MODEL_DIR / "movement_lstm.keras"
LABELS_PATH = MODEL_DIR / "labels.json"

CONFIDENCE_THRESHOLD = 0.80

print("Loading candidate model...")

model = tf.keras.models.load_model(
    str(MODEL_PATH)
)

with open(
    LABELS_PATH,
    "r",
    encoding="utf-8",
) as file:
    labels = json.load(file)

print("Loaded movements:", labels)

sequence_buffer = deque(
    maxlen=SEQUENCE_LENGTH
)

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError(
        "Unable to open webcam"
    )

print("\nCandidate webcam test started.")
print("Press Q to exit.")

try:
    while True:
        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        results = hands.process(rgb)

        features = np.asarray(
            extract_features(results),
            dtype=np.float32,
        )

        if features.shape != (FEATURE_COUNT,):
            raise ValueError(
                f"Unexpected feature shape: "
                f"{features.shape}"
            )

        has_hands = bool(
            results.multi_hand_landmarks
        )

        if not has_hands:
            sequence_buffer.clear()

        else:
            sequence_buffer.append(features)

        label = "Waiting for movement..."
        confidence = 0.0

        if len(sequence_buffer) == SEQUENCE_LENGTH:
            input_data = np.asarray(
                sequence_buffer,
                dtype=np.float32,
            ).reshape(
                1,
                SEQUENCE_LENGTH,
                FEATURE_COUNT,
            )

            probabilities = model.predict(
                input_data,
                verbose=0,
            )[0]

            top_indices = np.argsort(probabilities)[-3:][::-1]

            print("\nTop 3 predictions:")

            for index in top_indices:
               print(
                  f"{labels[index]}: "
                  f"{probabilities[index] * 100:.2f}%"
                )

            index = int(
                np.argmax(probabilities)
            )

            confidence = float(
                probabilities[index]
            )

            if confidence >= CONFIDENCE_THRESHOLD:
                label = labels[index]
            else:
                label = "Uncertain"

        cv2.putText(
            frame,
            f"Movement: {label}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            f"Confidence: {confidence:.2f}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Frames: {len(sequence_buffer)}/30",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow(
            "SignAI - Candidate LSTM Test",
            frame,
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    hands.close()
    cv2.destroyAllWindows()
