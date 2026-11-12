
import argparse
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

ANNOTATIONS_PATH = (
    ROOT
    / "external-datasets"
    / "WLASL"
    / "start_kit"
    / "WLASL_v0.3.json"
)

OUTPUT_PATH = (
    ROOT
    / "datasets"
    / "wlasl"
    / "selected.json"
)

DEFAULT_WORDS = [
    "yes",
    "no",
    "please",
    "sorry",
    "help",
    "eat",
    "drink",
    "good",
    "bad",
    "love",
]


def normalize_word(word):
    return " ".join(
        str(word).strip().lower().split()
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--words",
        nargs="+",
        default=DEFAULT_WORDS,
        help="WLASL glosses to select",
    )

    args = parser.parse_args()

    if not ANNOTATIONS_PATH.exists():
        raise SystemExit(
            "WLASL annotations not found:\n"
            f"{ANNOTATIONS_PATH}\n"
            "Clone the WLASL repository first."
        )

    with open(
        ANNOTATIONS_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        annotations = json.load(file)

    requested = {
        normalize_word(word)
        for word in args.words
    }

    available = {
        normalize_word(entry["gloss"])
        for entry in annotations
    }

    missing = requested - available

    if missing:
        print("\nWords not found in WLASL:")
        for word in sorted(missing):
            print(f"  - {word}")

    selected = []

    for entry in annotations:
        gloss = normalize_word(entry["gloss"])

        if gloss not in requested:
            continue

        label = gloss.upper().replace(" ", "_")

        for instance in entry.get("instances", []):
            split = str(
                instance.get("split", "")
            ).lower()

            if split not in {
                "train",
                "val",
                "test",
            }:
                continue

            selected.append({
                "gloss": label,
                "original_gloss": entry["gloss"],
                "video_id": str(
                    instance["video_id"]
                ),
                "url": instance.get("url", ""),
                "split": split,
                "signer_id": instance.get(
                    "signer_id"
                ),
                "frame_start": instance.get(
                    "frame_start", 1
                ),
                "frame_end": instance.get(
                    "frame_end", -1
                ),
                "fps": instance.get(
                    "fps", 25
                ),
            })

    if not selected:
        raise SystemExit(
            "No matching WLASL records found."
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            selected,
            file,
            indent=2,
        )

    counts = Counter(
        (
            item["gloss"],
            item["split"],
        )
        for item in selected
    )

    labels = sorted({
        item["gloss"]
        for item in selected
    })

    print("\nWLASL Movement Selection")
    print("=" * 60)

    print(
        f"{'Movement':15}"
        f"{'Train':>10}"
        f"{'Val':>10}"
        f"{'Test':>10}"
        f"{'Total':>10}"
    )

    print("-" * 60)

    for label in labels:
        train = counts[(label, "train")]
        val = counts[(label, "val")]
        test = counts[(label, "test")]

        print(
            f"{label:15}"
            f"{train:>10}"
            f"{val:>10}"
            f"{test:>10}"
            f"{train + val + test:>10}"
        )

    print("=" * 60)
    print(f"Selected movements: {len(labels)}")
    print(f"Annotated videos: {len(selected)}")
    print(f"Manifest: {OUTPUT_PATH}")

    print(
        "\nNote: Counts represent annotations, "
        "not successfully downloaded videos."
    )


if __name__ == "__main__":
    main()
