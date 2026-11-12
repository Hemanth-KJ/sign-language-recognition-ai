
import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands

FEATURE_SIZE = 126


def extract_hand_features(image, detector, flip=False):
    if image is None:
        return None

    if flip:
        image = cv2.flip(image, 1)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False

    results = detector.process(rgb)

    features = np.zeros((2, 21, 3), dtype=np.float32)

    if not results.multi_hand_landmarks:
        return None

    for hand_landmarks, handedness in zip(
        results.multi_hand_landmarks,
        results.multi_handedness
    ):
        label = handedness.classification[0].label

        # MediaPipe Hands labels assume mirrored input.
        # Use mirrored images consistently across the pipeline.
        index = 0 if label == "Left" else 1

        landmarks = np.array(
            [
                [lm.x, lm.y, lm.z]
                for lm in hand_landmarks.landmark
            ],
            dtype=np.float32
        )

        # Normalize each hand relative to its wrist.
        wrist = landmarks[0].copy()
        landmarks -= wrist

        scale = np.max(
            np.linalg.norm(landmarks, axis=1)
        )

        if scale > 1e-6:
            landmarks /= scale

        features[index] = landmarks

    return features.flatten()
