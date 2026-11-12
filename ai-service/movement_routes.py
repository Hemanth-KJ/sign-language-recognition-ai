
from typing import List
from uuid import UUID
from threading import Lock
import base64
import binascii

import cv2
import numpy as np
import mediapipe as mp

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from movement_features import (
    FEATURE_COUNT,
    extract_features,
)

from movement_service import (
    labels,
    predict_frame,
    reset_session,
)


# ==========================================
# CONFIGURATION
# ==========================================

MAX_IMAGE_BYTES = 5 * 1024 * 1024

router = APIRouter(
    prefix="/api/movement",
    tags=["Movement Recognition"],
)


# ==========================================
# MEDIAPIPE DETECTOR
# ==========================================

image_detector_lock = Lock()

# Static image mode avoids carrying hand-tracking
# state between different client sessions.
# Detection is performed independently per image.
image_detector = mp.solutions.hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5,
)


def close_movement_detector():
    with image_detector_lock:
        image_detector.close()


# ==========================================
# REQUEST MODELS
# ==========================================

class MovementFrameRequest(BaseModel):
    sessionId: UUID

    features: List[float] = Field(
        min_length=FEATURE_COUNT,
        max_length=FEATURE_COUNT,
    )

    handDetected: bool = True


class MovementImageRequest(BaseModel):
    sessionId: UUID

    image: str = Field(
        min_length=1,
    )


class ResetRequest(BaseModel):
    sessionId: UUID


# ==========================================
# HEALTH CHECK
# ==========================================

@router.get("/health")
def movement_health():

    return {
        "success": True,
        "model": "LSTM",
        "labels": labels,
        "featuresPerFrame": FEATURE_COUNT,
        "endpoints": {
            "predict": "/api/movement/predict",
            "predictImage": "/api/movement/predict-image",
            "reset": "/api/movement/reset",
        },
    }


# ==========================================
# PREDICT FROM LANDMARK FEATURES
# ==========================================

@router.post("/predict")
def movement_predict(
    request: MovementFrameRequest,
):

    try:

        result = predict_frame(
            session_id=str(request.sessionId),
            features=request.features,
            hand_detected=request.handDetected,
        )

        return {
            "success": True,
            "data": result,
        }

    except ValueError as error:

        raise HTTPException(
            status_code=422,
            detail=str(error),
        )


# ==========================================
# DECODE BASE64 IMAGE
# ==========================================

def decode_image(image_string):

    try:

        encoded = image_string.split(",", 1)[-1]

        max_encoded_length = (
            (MAX_IMAGE_BYTES + 2) // 3
        ) * 4

        if len(encoded) > max_encoded_length:

            raise HTTPException(
                status_code=413,
                detail="Image is too large",
            )

        image_bytes = base64.b64decode(
            encoded,
            validate=True,
        )

        if len(image_bytes) > MAX_IMAGE_BYTES:

            raise HTTPException(
                status_code=413,
                detail="Image is too large",
            )

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8,
        )

        frame = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR,
        )

        if frame is None:

            raise HTTPException(
                status_code=400,
                detail="Invalid image",
            )

        return frame

    except (ValueError, binascii.Error):

        raise HTTPException(
            status_code=400,
            detail="Invalid base64 image",
        )


# ==========================================
# PREDICT FROM WEBCAM IMAGE
# ==========================================

@router.post("/predict-image")
def movement_predict_image(
    request: MovementImageRequest,
):

    # --------------------------------------
    # STEP 1: Decode image
    # --------------------------------------

    frame = decode_image(request.image)

    # --------------------------------------
    # STEP 2: Mirror frame
    # --------------------------------------

    # Match the mirrored frames used
    # during movement dataset collection.
    frame = cv2.flip(frame, 1)

    # --------------------------------------
    # STEP 3: Convert BGR to RGB
    # --------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )

    rgb_frame.flags.writeable = False

    # --------------------------------------
    # STEP 4: Detect hands
    # --------------------------------------

    with image_detector_lock:

        results = image_detector.process(
            rgb_frame
        )

        # Extract while detector results
        # are still protected by the lock.
        hand_detected = bool(
            results.multi_hand_landmarks
        )

        features = extract_features(
            results
        )

    # --------------------------------------
    # STEP 5: Validate extracted features
    # --------------------------------------

    if features.shape != (FEATURE_COUNT,):

        raise HTTPException(
            status_code=500,
            detail="Invalid movement feature shape",
        )

    # --------------------------------------
    # STEP 6: Run LSTM prediction
    # --------------------------------------

    try:

        prediction = predict_frame(
            session_id=str(request.sessionId),
            features=features,
            hand_detected=hand_detected,
        )

    except ValueError as error:

        raise HTTPException(
            status_code=422,
            detail=str(error),
        )

    # --------------------------------------
    # STEP 7: Return prediction
    # --------------------------------------

    return {
        "success": True,
        "data": prediction,
    }


# ==========================================
# RESET MOVEMENT SESSION
# ==========================================

@router.post("/reset")
def movement_reset(
    request: ResetRequest,
):

    reset_session(
        str(request.sessionId)
    )

    return {
        "success": True,
        "message": "Movement session reset",
    }
