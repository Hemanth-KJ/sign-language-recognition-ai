
import json
import time
from collections import deque
from pathlib import Path
from threading import Lock

import numpy as np
import tensorflow as tf

from movement_features import (
    SEQUENCE_LENGTH,
    FEATURE_COUNT,
)


# ==========================================
# CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = ROOT / "models" / "movement_model"

MODEL_PATH = MODEL_DIR / "movement_lstm.keras"
LABELS_PATH = MODEL_DIR / "labels.json"

CONFIDENCE_THRESHOLD = 0.80
STABILITY_FRAMES = 5

SESSION_TIMEOUT_SECONDS = 120

# Movement boundary detection
MOTION_THRESHOLD = 0.015
PAUSE_FRAMES = 5

# Minimum interval between emitted predictions
PREDICTION_COOLDOWN_SECONDS = 1.5


# ==========================================
# LOAD MODEL
# ==========================================

model = tf.keras.models.load_model(
    str(MODEL_PATH),
    compile=False,
)

with open(LABELS_PATH, "r", encoding="utf-8") as file:
    labels = json.load(file)

if model.input_shape[1:] != (
    SEQUENCE_LENGTH,
    FEATURE_COUNT,
):
    raise ValueError(
        "Unexpected LSTM input shape"
    )

if model.output_shape[-1] != len(labels):
    raise ValueError(
        "Model output does not match labels"
    )


# ==========================================
# SESSION MANAGEMENT
# ==========================================

sessions = {}

sessions_lock = Lock()
model_lock = Lock()


def create_session():
    return {
        "frames": deque(maxlen=SEQUENCE_LENGTH),
        "predictions": deque(maxlen=STABILITY_FRAMES),
        "previous_features": None,
        "pause_count": 0,
        "last_emitted_sign": None,
        "last_emitted_at": 0.0,
        "last_seen": time.monotonic(),
    }


def reset_session(session_id):
    with sessions_lock:
        sessions.pop(session_id, None)


def response(status, sign=None, confidence=0.0, frames=0):
    return {
        "status": status,
        "sign": sign,
        "confidence": float(confidence),
        "framesCollected": frames,
    }


# ==========================================
# MOTION DETECTION
# ==========================================

def calculate_motion(current, previous):
    if previous is None:
        return 0.0

    # Average displacement of normalized
    # landmark features between two frames.
    return float(
        np.mean(np.abs(current - previous))
    )


# ==========================================
# PREDICT FRAME
# ==========================================

def predict_frame(
    session_id,
    features,
    hand_detected=True,
):

    features = np.asarray(
        features,
        dtype=np.float32,
    )

    if features.shape != (FEATURE_COUNT,):
        raise ValueError(
            f"Expected {FEATURE_COUNT} features"
        )

    if not np.isfinite(features).all():
        raise ValueError(
            "Features must contain finite numbers"
        )

    with sessions_lock:

        now = time.monotonic()

        # Remove inactive sessions
        expired = [
            key
            for key, state in sessions.items()
            if now - state["last_seen"]
            > SESSION_TIMEOUT_SECONDS
        ]

        for key in expired:
            del sessions[key]

        state = sessions.setdefault(
            session_id,
            create_session(),
        )

        state["last_seen"] = now

        # ==================================
        # NO HAND DETECTED
        # ==================================

        if not hand_detected:

            state["frames"].clear()
            state["predictions"].clear()
            state["previous_features"] = None
            state["pause_count"] = 0

            return response(
                status="waiting",
            )

        # ==================================
        # CALCULATE MOTION
        # ==================================

        motion = calculate_motion(
            features,
            state["previous_features"],
        )

        state["previous_features"] = features.copy()

        if motion < MOTION_THRESHOLD:
            state["pause_count"] += 1
        else:
            state["pause_count"] = 0

        # ==================================
        # DETECT GESTURE BOUNDARY
        # ==================================

        if state["pause_count"] == PAUSE_FRAMES:

            state["frames"].clear()
            state["predictions"].clear()

            return response(
                status="collecting",
                frames=0,
            )

        # ==================================
        # ADD FRAME
        # ==================================

        state["frames"].append(features)

        count = len(state["frames"])

        if count < SEQUENCE_LENGTH:

            return response(
                status="collecting",
                frames=count,
            )

        # ==================================
        # LSTM PREDICTION
        # ==================================

        sequence = np.asarray(
            state["frames"],
            dtype=np.float32,
        )

        with model_lock:

            probabilities = model.predict(
                np.expand_dims(
                    sequence,
                    axis=0,
                ),
                verbose=0,
            )[0]

        index = int(
            np.argmax(probabilities)
        )

        label = labels[index]

        confidence = float(
            probabilities[index]
        )

        # ==================================
        # UNKNOWN GESTURE
        # ==================================

        if (
            label == "NO_SIGN"
            or confidence < CONFIDENCE_THRESHOLD
        ):

            state["predictions"].clear()

            return response(
                status="unknown",
                confidence=confidence,
                frames=count,
            )

        # ==================================
        # STABILITY CHECK
        # ==================================

        state["predictions"].append(label)

        stable = (
            len(state["predictions"]) == STABILITY_FRAMES
            and len(set(state["predictions"])) == 1
        )

        if not stable:

            return response(
                status="analyzing",
                confidence=confidence,
                frames=count,
            )

        # ==================================
        # COOLDOWN
        # ==================================

        if (
            label == state["last_emitted_sign"]
            and now - state["last_emitted_at"]
            < PREDICTION_COOLDOWN_SECONDS
        ):

            return response(
                status="analyzing",
                confidence=confidence,
                frames=count,
            )

        state["last_emitted_sign"] = label
        state["last_emitted_at"] = now

        return response(
            status="recognized",
            sign=label,
            confidence=confidence,
            frames=count,
        )
