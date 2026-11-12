
import cv2
import numpy as np
import mediapipe as mp

# ==========================================
# SignAI - Movement Feature Configuration
# ==========================================

SEQUENCE_LENGTH = 30
LANDMARKS_PER_HAND = 21
COORDINATES_PER_LANDMARK = 3
MAX_HANDS = 2

FEATURE_COUNT = (
    LANDMARKS_PER_HAND
    * COORDINATES_PER_LANDMARK
    * MAX_HANDS
)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# ==========================================
# Initialize MediaPipe Hands
# ==========================================

def create_hands():
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_HANDS,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


# ==========================================
# Process Webcam Frame
# ==========================================

def process_frame(frame, hands):
    """
    Accepts a BGR image and returns
    MediaPipe hand detection results.

    The caller should mirror the frame
    before passing it to this function.
    """

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    rgb_frame.flags.writeable = False

    results = hands.process(rgb_frame)

    rgb_frame.flags.writeable = True

    return results


# ==========================================
# Extract 126 Landmark Features
# ==========================================

def extract_features(results):
    """
    Extract normalized landmark coordinates.

    Left hand:  63 features
    Right hand: 63 features

    Output: NumPy array of shape (126,)
    """

    features = np.zeros(
        (
            MAX_HANDS,
            LANDMARKS_PER_HAND,
            COORDINATES_PER_LANDMARK
        ),
        dtype=np.float32
    )

    if (
        not results.multi_hand_landmarks
        or not results.multi_handedness
    ):
        return features.flatten()

    for hand_landmarks, handedness in zip(
        results.multi_hand_landmarks,
        results.multi_handedness
    ):

        hand_label = (
            handedness.classification[0].label
        )

        hand_index = (
            0 if hand_label == "Left" else 1
        )

        points = np.array(
            [
                [
                    landmark.x,
                    landmark.y,
                    landmark.z
                ]
                for landmark in hand_landmarks.landmark
            ],
            dtype=np.float32
        )

        # Make coordinates relative to wrist.
        wrist = points[0].copy()

        points = points - wrist

        # Normalize relative to hand size.
        distances = np.linalg.norm(
            points,
            axis=1
        )

        scale = np.max(distances)

        if scale > 1e-6:
            points = points / scale

        features[hand_index] = points

    return features.flatten()


# ==========================================
# Draw Hand Landmarks
# ==========================================

def draw_landmarks(frame, results):
    """
    Draw MediaPipe landmarks and connections.
    """

    if results.multi_hand_landmarks:

        for hand_landmarks in (
            results.multi_hand_landmarks
        ):

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    return frame
