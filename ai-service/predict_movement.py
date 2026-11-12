
import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from movement_features import (
    SEQUENCE_LENGTH,
    FEATURE_COUNT,
    create_hands,
    process_frame,
    extract_features,
    draw_landmarks,
)

# ==========================================
# SignAI - Real-Time LSTM Prediction
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models" / "movement_model"

MODEL_PATH = MODEL_DIR / "movement_lstm.keras"
LABELS_PATH = MODEL_DIR / "labels.json"

CONFIDENCE_THRESHOLD = 0.80
STABILITY_FRAMES = 5

print("Loading SignAI movement model...")

model = tf.keras.models.load_model(
    str(MODEL_PATH),
    compile=False,
)

with open(LABELS_PATH, "r", encoding="utf-8") as file:
    labels = json.load(file)

print("Model loaded successfully")
print("Supported signs:", labels)

sequence = deque(maxlen=SEQUENCE_LENGTH)
prediction_history = deque(maxlen=STABILITY_FRAMES)


def predict_sequence(frames):
    data = np.asarray(frames, dtype=np.float32)

    if data.shape != (SEQUENCE_LENGTH, FEATURE_COUNT):
        raise ValueError(f"Invalid input shape: {data.shape}")

    probabilities = model.predict(
        np.expand_dims(data, axis=0),
        verbose=0,
    )[0]

    index = int(np.argmax(probabilities))

    return labels[index], float(probabilities[index])


def main():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Unable to open webcam")

    current_prediction = "Waiting for hand"
    current_confidence = 0.0

    try:
        with create_hands() as hands:
            while True:
                success, frame = camera.read()

                if not success:
                    print("Camera frame read failed")
                    break

                frame = cv2.flip(frame, 1)

                results = process_frame(frame, hands)

                detected = bool(
                    results.multi_hand_landmarks
                )

                if detected:
                    features = extract_features(results)
                    sequence.append(features)

                    draw_landmarks(frame, results)

                    if len(sequence) == SEQUENCE_LENGTH:
                        label, confidence = predict_sequence(
                            sequence
                        )

                        current_confidence = confidence
                        if (
                           confidence >= CONFIDENCE_THRESHOLD
                           and label != "NO_SIGN"
                           ):
                           prediction_history.append(label)
                        else:
                           prediction_history.clear()

                        if label == "NO_SIGN":
                           current_prediction = "No recognized sign"

                        elif confidence < CONFIDENCE_THRESHOLD:
                           current_prediction = "Uncertain"

                        elif (
                           len(prediction_history) == STABILITY_FRAMES
                           and len(set(prediction_history)) == 1
                             ):
                            current_prediction = label

                        else:
                           current_prediction = "Analyzing"
                    else:
                        current_prediction = "Collecting frames"
                        current_confidence = 0.0
                else:
                    sequence.clear()
                    prediction_history.clear()
                    current_prediction = "Waiting for hand"
                    current_confidence = 0.0

                # Prediction overlay
                cv2.rectangle(
                    frame,
                    (0, 0),
                    (frame.shape[1], 140),
                    (25, 25, 25),
                    -1,
                )

                cv2.putText(
                    frame,
                    f"Sign: {current_prediction}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Confidence: {current_confidence * 100:.1f}%",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Frames: {len(sequence)}/{SEQUENCE_LENGTH}",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow(
                    "SignAI - Dynamic Sign Recognition",
                    frame,
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
