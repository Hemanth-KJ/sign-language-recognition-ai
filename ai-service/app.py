
from pathlib import Path
from contextlib import asynccontextmanager
from threading import Lock
import base64
import binascii

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from hand_features import (
    extract_hand_features,
    FEATURE_SIZE
)

from movement_routes import (
    router as movement_router,
    close_movement_detector
)

# ==========================================
# CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

STATIC_MODEL_PATH = ROOT / "models" / "hand_model"

CONFIDENCE_THRESHOLD = 75.0
MAX_IMAGE_BYTES = 5 * 1024 * 1024

# Static sign recognition resources
model = None
labels = None
detector = None

detector_lock = Lock()
model_lock = Lock()


# ==========================================
# MODEL INITIALIZATION
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global model, labels, detector

    print("Starting SignAI AI Service...")

    try:

        # ==================================
        # LOAD STATIC SIGN MODEL
        # ==================================

        print("Loading static hand recognition model...")

        static_model_file = (
            STATIC_MODEL_PATH / "sign_model.keras"
        )

        static_labels_file = (
            STATIC_MODEL_PATH / "labels.npy"
        )

        if not static_model_file.exists():
            raise FileNotFoundError(
                f"Static model not found: {static_model_file}"
            )

        if not static_labels_file.exists():
            raise FileNotFoundError(
                f"Static labels not found: {static_labels_file}"
            )

        model = tf.keras.models.load_model(
            str(static_model_file)
        )

        labels = np.load(
            static_labels_file,
            allow_pickle=True
        )

        # ==================================
        # VALIDATE STATIC MODEL
        # ==================================

        if model.input_shape[-1] != FEATURE_SIZE:
            raise RuntimeError(
                "Static model input does not match hand features"
            )

        if model.output_shape[-1] != len(labels):
            raise RuntimeError(
                "Static model output and labels do not match"
            )

        print("Static model loaded successfully")
        print("Static classes:", labels)

        # ==================================
        # INITIALIZE MEDIAPIPE
        # ==================================

        detector = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5
        )

        print("MediaPipe hand detector initialized")

        # ==================================
        # MOVEMENT MODEL
        # ==================================

        # The movement model is loaded by
        # movement_service.py when
        # movement_routes.py is imported.

        print("Movement recognition routes registered")

        print("SignAI AI Service is ready")

        yield

    finally:

        print("Shutting down SignAI AI Service...")

        # Close static hand detector
        if detector is not None:
            detector.close()
            detector = None

        # Close dynamic movement detector
        close_movement_detector()

        # Clear static model references
        model = None
        labels = None

        print("SignAI AI Service stopped")


# ==========================================
# FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="SignAI - AI Recognition Service",
    description=(
        "AI-powered static and dynamic "
        "sign language recognition API"
    ),
    version="3.0.0",
    lifespan=lifespan
)


# ==========================================
# CORS CONFIGURATION
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_methods=[
        "GET",
        "POST"
    ],
    allow_headers=[
        "Content-Type",
        "Authorization"
    ]
)


# ==========================================
# REGISTER MOVEMENT ROUTES
# ==========================================

app.include_router(movement_router)


# ==========================================
# REQUEST MODEL
# ==========================================

class PredictRequest(BaseModel):
    image: str = Field(min_length=1)


# ==========================================
# ROOT ENDPOINT
# ==========================================

@app.get("/")
def home():

    return {
        "success": True,
        "message": "SignAI AI Recognition Service running",
        "version": "3.0.0",
        "features": {
            "staticRecognition": True,
            "movementRecognition": True
        },
        "endpoints": {
            "staticPredict": "/predict",
            "movementPredict": "/api/movement/predict",
            "movementHealth": "/api/movement/health",
            "movementReset": "/api/movement/reset",
            "documentation": "/docs"
        }
    }


# ==========================================
# STATIC MODEL HEALTH CHECK
# ==========================================

@app.get("/health")
def health():

    static_ready = (
        model is not None
        and labels is not None
        and detector is not None
    )

    return {
        "success": True,
        "model_loaded": static_ready,
        "classes": (
            len(labels)
            if labels is not None
            else 0
        ),
        "services": {
            "staticRecognition": (
                "ready" if static_ready else "not_ready"
            ),
            "movementRecognition": (
                "available"
            )
        }
    }


# ==========================================
# STATIC SIGN PREDICTION
# ==========================================

@app.post("/predict")
def predict(data: PredictRequest):

    if model is None or detector is None:

        raise HTTPException(
            status_code=503,
            detail="Static AI model is not ready"
        )

    # ======================================
    # DECODE BASE64 IMAGE
    # ======================================

    try:

        encoded = data.image.split(",", 1)[-1]

        if len(encoded) > MAX_IMAGE_BYTES * 4 // 3 + 16:

            raise HTTPException(
                status_code=413,
                detail="Image is too large"
            )

        image_bytes = base64.b64decode(
            encoded,
            validate=True
        )

        if len(image_bytes) > MAX_IMAGE_BYTES:

            raise HTTPException(
                status_code=413,
                detail="Image is too large"
            )

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        frame = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if frame is None:

            raise HTTPException(
                status_code=400,
                detail="Invalid image"
            )

    except (ValueError, binascii.Error):

        raise HTTPException(
            status_code=400,
            detail="Invalid base64 image"
        )

    # ======================================
    # EXTRACT HAND FEATURES
    # ======================================

    try:

        with detector_lock:

            features = extract_hand_features(
                frame,
                detector,
                flip=True
            )

        if features is None:

            return {
                "success": True,
                "prediction": None,
                "confidence": 0,
                "status": "no_hand"
            }

        if features.shape != (FEATURE_SIZE,):

            raise ValueError(
                "Invalid static hand feature shape"
            )

        # ==================================
        # PREPARE MODEL INPUT
        # ==================================

        x = features.reshape(
            1,
            FEATURE_SIZE
        )

        # ==================================
        # MODEL INFERENCE
        # ==================================

        with model_lock:

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

        # ==================================
        # CONFIDENCE CHECK
        # ==================================

        if confidence < CONFIDENCE_THRESHOLD:

            return {
                "success": True,
                "prediction": None,
                "confidence": round(
                    confidence,
                    2
                ),
                "status": "low_confidence"
            }

        # ==================================
        # SUCCESS RESPONSE
        # ==================================

        return {
            "success": True,
            "prediction": str(
                labels[class_id]
            ),
            "confidence": round(
                confidence,
                2
            ),
            "class_id": class_id,
            "status": "detected"
        }

    except Exception as error:

        print(
            "Static prediction error:",
            error
        )

        raise HTTPException(
            status_code=500,
            detail="Static prediction failed"
        )
