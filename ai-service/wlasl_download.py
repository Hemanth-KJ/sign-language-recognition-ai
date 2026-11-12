
import argparse
import json
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import cv2
import requests


ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT / "datasets" / "wlasl"

MANIFEST_PATH = DATASET_DIR / "selected.json"
VIDEO_DIR = DATASET_DIR / "videos"
REPORT_PATH = DATASET_DIR / "download_report.json"

VIDEO_EXTENSIONS = {
    ".mp4",
    ".webm",
    ".mkv",
    ".mov",
}

MAX_DOWNLOAD_BYTES = 250 * 1024 * 1024


def is_youtube_url(url):
    host = (urlparse(url).hostname or "").lower()

    return (
        host == "youtube.com"
        or host.endswith(".youtube.com")
        or host == "youtu.be"
    )


def find_existing_video(video_id):
    for extension in VIDEO_EXTENSIONS:
        path = VIDEO_DIR / f"{video_id}{extension}"

        if path.exists() and path.stat().st_size > 0:
            return path

    return None


def validate_video(path):
    capture = cv2.VideoCapture(str(path))

    if not capture.isOpened():
        capture.release()
        return False

    success, frame = capture.read()
    capture.release()

    return (
        success
        and frame is not None
        and frame.size > 0
    )


def download_direct(url, destination):
    temporary = destination.with_suffix(".part")

    try:
        with requests.get(
            url,
            stream=True,
            timeout=(15, 60),
            headers={
                "User-Agent": "Mozilla/5.0"
            },
        ) as response:

            response.raise_for_status()

            content_type = response.headers.get(
                "Content-Type", ""
            ).lower()

            if (
                "text/html" in content_type
                or "application/json" in content_type
            ):
                raise ValueError(
                    "Source returned a webpage instead of a video"
                )

            total_bytes = 0

            with open(temporary, "wb") as file:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if not chunk:
                        continue

                    total_bytes += len(chunk)

                    if total_bytes > MAX_DOWNLOAD_BYTES:
                        raise ValueError(
                            "Download exceeds 250 MB"
                        )

                    file.write(chunk)

        if temporary.stat().st_size < 1024:
            raise ValueError(
                "Downloaded file is too small"
            )

        temporary.replace(destination)

        return destination

    finally:
        if temporary.exists():
            temporary.unlink()


def download_youtube(url, video_id):
    output_template = str(
        VIDEO_DIR / f"{video_id}.%(ext)s"
    )

    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-playlist",
        "--no-overwrites",
        "--merge-output-format",
        "mp4",
        "-f",
        "bv*+ba/b",
        "-o",
        output_template,
        url,
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=240,
    )

    if result.returncode != 0:
        raise RuntimeError(
            result.stderr[-800:]
            or "yt-dlp failed"
        )

    saved = find_existing_video(video_id)

    if saved is None:
        raise RuntimeError(
            "Download completed but video file was not found"
        )

    return saved


def save_report(report):
    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum selected records to process; 0 means all",
    )

    parser.add_argument(
        "--skip-youtube",
        action="store_true",
    )

    args = parser.parse_args()

    if not MANIFEST_PATH.exists():
        raise SystemExit(
            "selected.json not found. "
            "Run python wlasl_select.py first."
        )

    VIDEO_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        MANIFEST_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        records = json.load(file)

    if args.limit > 0:
        records = records[:args.limit]

    report = {}

    if REPORT_PATH.exists():
        with open(
            REPORT_PATH,
            "r",
            encoding="utf-8",
        ) as file:
            report = json.load(file)

    for index, item in enumerate(records, start=1):
        video_id = item["video_id"]
        url = item.get("url", "")

        print(
            f"\n[{index}/{len(records)}] "
            f"{item['gloss']} | {video_id}"
        )

        existing = find_existing_video(video_id)

        if existing and validate_video(existing):
            print("Already downloaded")

            report[video_id] = {
                "status": "downloaded",
                "path": str(existing),
            }

            save_report(report)
            continue

        if existing:
            print("Invalid existing video, retrying")
            existing.unlink()

        if not url:
            print("Missing source URL")

            report[video_id] = {
                "status": "failed",
                "reason": "Missing URL",
            }

            save_report(report)
            continue

        if args.skip_youtube and is_youtube_url(url):
            print("Skipping YouTube source")
            continue

        try:
            if is_youtube_url(url):
                saved = download_youtube(
                    url,
                    video_id,
                )
            else:
                saved = download_direct(
                    url,
                    VIDEO_DIR / f"{video_id}.mp4",
                )

            if not validate_video(saved):
                saved.unlink(missing_ok=True)

                raise ValueError(
                    "Downloaded video could not be decoded"
                )

            report[video_id] = {
                "status": "downloaded",
                "path": str(saved),
            }

            print("Downloaded successfully")

        except Exception as error:
            report[video_id] = {
                "status": "failed",
                "reason": str(error),
            }

            print(f"Download failed: {error}")

        save_report(report)

    selected_ids = {
        item["video_id"]
        for item in records
    }

    successful = sum(
        report.get(video_id, {}).get("status")
        == "downloaded"
        for video_id in selected_ids
    )

    failed = sum(
        report.get(video_id, {}).get("status")
        == "failed"
        for video_id in selected_ids
    )

    print("\n" + "=" * 50)
    print("WLASL DOWNLOAD SUMMARY")
    print("=" * 50)
    print("Selected records:", len(records))
    print("Downloaded:", successful)
    print("Failed:", failed)
    print("Report:", REPORT_PATH)


if __name__ == "__main__":
    main()
