
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "datasets" / "wlasl"

with open(
    DATA_DIR / "selected.json",
    encoding="utf-8",
) as file:
    records = json.load(file)

with open(
    DATA_DIR / "download_report.json",
    encoding="utf-8",
) as file:
    report = json.load(file)

summary = defaultdict(
    lambda: {
        "train": 0,
        "val": 0,
        "test": 0,
    }
)

for item in records:
    video_id = item["video_id"]

    result = report.get(video_id, {})

    if result.get("status") != "downloaded":
        continue

    path = Path(result.get("path", ""))

    if not path.is_file():
        continue

    summary[item["gloss"]][item["split"]] += 1

print("\nWLASL Downloaded Videos")
print("=" * 55)

print(
    f"{'Movement':15}"
    f"{'Train':>10}"
    f"{'Val':>10}"
    f"{'Test':>10}"
    f"{'Total':>10}"
)

for gloss, counts in sorted(summary.items()):
    total = sum(counts.values())

    print(
        f"{gloss:15}"
        f"{counts['train']:>10}"
        f"{counts['val']:>10}"
        f"{counts['test']:>10}"
        f"{total:>10}"
    )

print("=" * 55)
