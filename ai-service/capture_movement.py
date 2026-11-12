
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from movement_features import (
    SEQUENCE_LENGTH,
    FEATURE_COUNT,
    create_hands,
    process_frame,
    extract_features,
    draw_landmarks,
)

# ==========================================
# SignAI - Dynamic Movement Dataset Capture
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "datasets" / "movements"


def display_text(frame, text, position, color=(255, 255, 255)):
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )


def capture_sign(sign_name, total_sequences, camera_index=0):

    save_dir = DATASET_DIR / sign_name
    save_dir.mkdir(parents=True, exist_ok=True)

    existing_indices = [
        int(path.stem)
        for path in save_dir.glob("*.npy")
        if path.stem.isdigit()
    ]

    next_index = max(existing_indices, default=-1) + 1

    camera = cv2.VideoCapture(camera_index)

    if not camera.isOpened():
        raise RuntimeError("Unable to open webcam")

    print(f"\nRecording sign: {sign_name}")
    print(f"Sequences to collect: {total_sequences}")
    print(f"Saving to: {save_dir}")
    print("Press S to start each recording")
    print("Press Q to quit")

    completed = 0

    try:
        with create_hands() as hands:

            while completed < total_sequences:

                # Wait until user presses S.
                while True:

                    success, frame = camera.read()

                    if not success:
                        raise RuntimeError("Camera frame read failed")

                    frame = cv2.flip(frame, 1)

                    results = process_frame(frame, hands)
                    draw_landmarks(frame, results)

                    display_text(
                        frame,
                        f"Sign: {sign_name}",
                        (20, 35),
                        (0, 255, 0),
                    )

                    display_text(
                        frame,
                        f"Sample: {completed + 1}/{total_sequences}",
                        (20, 70),
                    )

                    display_text(
                        frame,
                        "Press S to record | Q to quit",
                        (20, 105),
                    )

                    cv2.imshow(
                        "SignAI - Movement Dataset Capture",
                        frame,
                    )

                    key = cv2.waitKey(1) & 0xFF

                    if key == ord("q"):
                        print("Capture stopped by user")
                        return

                    if key == ord("s"):
                        break

                # Short countdown to prepare.
                for seconds in (3, 2, 1):

                    start_time = time.monotonic()

                    while time.monotonic() - start_time < 1:

                        success, frame = camera.read()

                        if not success:
                            raise RuntimeError("Camera read failed")

                        frame = cv2.flip(frame, 1)

                        display_text(
                            frame,
                            f"Starting in {seconds}",
                            (20, 60),
                            (0, 255, 255),
                        )

                        cv2.imshow(
                            "SignAI - Movement Dataset Capture",
                            frame,
                        )

                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            print("Capture stopped")
                            return

                # Capture 30 consecutive frames.
                sequence = []
                detected_frames = 0

                while len(sequence) < SEQUENCE_LENGTH:

                    success, frame = camera.read()

                    if not success:
                        raise RuntimeError("Camera read failed")

                    frame = cv2.flip(frame, 1)

                    results = process_frame(frame, hands)

                    features = extract_features(results)

                    if features.shape != (FEATURE_COUNT,):
                        raise RuntimeError(
                            f"Invalid feature shape: {features.shape}"
                        )

                    sequence.append(features)

                    if results.multi_hand_landmarks:
                        detected_frames += 1

                    draw_landmarks(frame, results)

                    display_text(
                        frame,
                        f"Recording: {len(sequence)}/{SEQUENCE_LENGTH}",
                        (20, 40),
                        (0, 255, 0),
                    )

                    display_text(
                        frame,
                        f"Sign: {sign_name}",
                        (20, 80),
                    )

                    cv2.imshow(
                        "SignAI - Movement Dataset Capture",
                        frame,
                    )

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("Capture stopped")
                        return

                # Reject samples where hands were mostly absent.
                if detected_frames < int(SEQUENCE_LENGTH * 0.8):
                    print(
                        "Sample rejected: hand not detected "
                        "in enough frames. Please retry."
                    )
                    continue

                sequence_array = np.asarray(
                    sequence,
                    dtype=np.float32,
                )

                file_path = save_dir / f"{next_index}.npy"

                np.save(file_path, sequence_array)

                completed += 1
                next_index += 1

                print(
                    f"Saved {file_path.name} | "
                    f"Shape: {sequence_array.shape} | "
                    f"Progress: {completed}/{total_sequences}"
                )

    finally:
        camera.release()
        cv2.destroyAllWindows()

    print(f"\nCompleted recording {sign_name}!")


def main():

    parser = argparse.ArgumentParser(
        description="SignAI movement dataset collector"
    )

    parser.add_argument(
        "--sign",
        required=True,
        help="Name of the movement sign",
    )

    parser.add_argument(
        "--sequences",
        type=int,
        default=30,
        help="Number of sequences to capture",
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index",
    )

    args = parser.parse_args()

    sign_name = args.sign.strip().upper().replace(" ", "_")

    if not sign_name or not all(
        char.isalnum() or char == "_"
        for char in sign_name
    ):
        parser.error("Invalid sign name")

    if args.sequences < 1:
        parser.error("Sequences must be at least 1")

    capture_sign(
        sign_name,
        args.sequences,
        args.camera,
    )


if __name__ == "__main__":
    main()
