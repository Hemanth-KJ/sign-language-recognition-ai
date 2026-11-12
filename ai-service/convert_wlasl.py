
import json
import argparse
from pathlib import Path
from urllib.parse import urlparse
from collections import Counter

import cv2
import numpy as np

from movement_features import (
    create_hands,
    process_frame,
    extract_features,
    SEQUENCE_LENGTH,
    FEATURE_COUNT,
)

ROOT = Path(__file__).resolve().parent.parent

WLASL_DIR = ROOT / "datasets" / "wlasl"
VIDEO_DIR = WLASL_DIR / "videos"
MANIFEST = WLASL_DIR / "selected.json"

OUTPUT_DIR = WLASL_DIR / "converted"
REPORT = WLASL_DIR / "conversion_report.json"

EXTENSIONS = [".mp4", ".webm", ".mkv", ".mov"]


def find_video(video_id):
    for ext in EXTENSIONS:
        path = VIDEO_DIR / f"{video_id}{ext}"
        if path.is_file():
            return path
    return None


def is_youtube(url):
    host = (urlparse(url).hostname or "").lower()
    return host in {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "youtu.be",
    }


def load_frames(video_path, item):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError("Cannot open video")

    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            frames.append(frame)
    finally:
        cap.release()

    if not frames:
        raise ValueError("No readable frames")

    # YouTube URLs may contain full source videos.
    # WLASL frame boundaries refer to a 25-FPS
    # annotation timeline.
    if is_youtube(item.get("url", "")):
        source_fps = 25.0
        actual_fps = fps if fps > 0 else source_fps

        start_frame = int(
            item.get("frame_start", 1)
        )

        end_frame = int(
            item.get("frame_end", -1)
        )

        start = round(
            max(0, start_frame - 1)
            / source_fps * actual_fps
        )

        if end_frame > 0:
            end = round(
                end_frame / source_fps * actual_fps
            )
        else:
            end = len(frames)

        start = max(0, min(start, len(frames)))
        end = max(start, min(end, len(frames)))

        frames = frames[start:end]

    if not frames:
        raise ValueError("Empty annotated segment")

    return frames


def convert_video(path, item, min_detected):
    frames = load_frames(path, item)

    indices = np.rint(
        np.linspace(
            0,
            len(frames) - 1,
            SEQUENCE_LENGTH,
        )
    ).astype(int)

    sequence = []
    detected_count = 0

    hands = create_hands()

    try:
        for index in indices:
            frame = frames[index]

            # process_frame must apply the same
            # preprocessing as live recognition.
            results = process_frame(frame, hands)

            if results.multi_hand_landmarks:
                detected_count += 1

            features = np.asarray(
                extract_features(results),
                dtype=np.float32,
            )

            if features.shape != (FEATURE_COUNT,):
                raise ValueError(
                    f"Invalid feature shape: {features.shape}"
                )

            if not np.isfinite(features).all():
                raise ValueError("Non-finite features")

            sequence.append(features)
    finally:
        hands.close()

    sequence = np.asarray(
        sequence,
        dtype=np.float32,
    )

    if sequence.shape != (
        SEQUENCE_LENGTH,
        FEATURE_COUNT,
    ):
        raise ValueError(
            f"Invalid sequence shape: {sequence.shape}"
        )

    if detected_count < min_detected:
        raise ValueError(
            f"Only {detected_count}/30 frames "
            "contained detected hands"
        )

    return sequence, detected_count


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--min-detected",
        type=int,
        default=15,
    )

    args = parser.parse_args()

    with open(MANIFEST, encoding="utf-8") as f:
        records = json.load(f)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = []
    counts = Counter()

    for index, item in enumerate(records, 1):
        video_id = item["video_id"]
        gloss = item["gloss"]
        split = item["split"]

        output = (
            OUTPUT_DIR
            / split
            / gloss
            / f"{video_id}.npy"
        )

        result = {
            "video_id": video_id,
            "gloss": gloss,
            "split": split,
        }

        video_path = find_video(video_id)

        if video_path is None:
            result["status"] = "missing_video"
            report.append(result)
            continue

        if output.exists():
            result["status"] = "existing"
            counts[(split, gloss)] += 1
            report.append(result)
            continue

        try:
            sequence, detected = convert_video(
                video_path,
                item,
                args.min_detected,
            )

            output.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            np.save(output, sequence)

            result["status"] = "converted"
            result["detected_frames"] = detected

            counts[(split, gloss)] += 1

            print(
                f"[{index}/{len(records)}] "
                f"{gloss} ({split}) "
                f"OK | hands {detected}/30"
            )

        except Exception as error:
            result["status"] = "failed"
            result["reason"] = str(error)

            print(
                f"[{index}/{len(records)}] "
                f"{gloss}: FAILED - {error}"
            )

        report.append(result)

    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nConversion summary")
    print("=" * 50)

    for (split, gloss), count in sorted(counts.items()):
        print(f"{split:5} {gloss:15} {count}")

    print("\nReport:", REPORT)


if __name__ == "__main__":
    main()
