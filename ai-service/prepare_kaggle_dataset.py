
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp

from hand_features import extract_hand_features

ROOT = Path(__file__).resolve().parent.parent

IMAGE_DATASET = (
    ROOT / "datasets" / "kaggle" / "asl_alphabet_train"
)

OUTPUT_DATASET = ROOT / "datasets" / "hand_features"

OUTPUT_DATASET.mkdir(
    parents=True,
    exist_ok=True
)

mp_hands = mp.solutions.hands


def main():
    if not IMAGE_DATASET.exists():
        raise FileNotFoundError(
            f"Dataset not found: {IMAGE_DATASET}"
        )

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5
    ) as detector:

        for sign_dir in sorted(IMAGE_DATASET.iterdir()):

            if not sign_dir.is_dir():
                continue

            samples = []
            total = 0

            for image_path in sign_dir.iterdir():

                if image_path.suffix.lower() not in {
                    ".jpg", ".jpeg", ".png"
                }:
                    continue

                total += 1

                image = cv2.imread(str(image_path))

                if image is None:
                    continue

                features = extract_hand_features(
                    image,
                    detector,
                    flip=True
                )

                if features is not None:
                    samples.append(features)

            if not samples:
                print(
                    f"{sign_dir.name}: no valid samples"
                )
                continue

            output = (
                OUTPUT_DATASET / f"{sign_dir.name}.npy"
            )

            np.save(
                output,
                np.stack(samples)
            )

            print(
                f"{sign_dir.name}: "
                f"{len(samples)}/{total} images converted"
            )


if __name__ == "__main__":
    main()
