
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

DATASET = (
    ROOT / "datasets" / "wlasl" / "converted"
)

counts = defaultdict(lambda: defaultdict(int))
invalid = 0

for split in ["train", "val", "test"]:
    folder = DATASET / split

    if not folder.exists():
        continue

    for file in folder.rglob("*.npy"):
        try:
            data = np.load(
                file,
                allow_pickle=False,
            )

            if (
                data.shape != (30, 126)
                or not np.isfinite(data).all()
            ):
                raise ValueError("Invalid array")

            counts[file.parent.name][split] += 1

        except Exception as error:
            invalid += 1
            print(f"INVALID: {file} - {error}")

print("\nConverted WLASL Dataset")
print("=" * 55)

print(
    f"{'Movement':15}"
    f"{'Train':>10}"
    f"{'Val':>10}"
    f"{'Test':>10}"
)

for label in sorted(counts):
    print(
        f"{label:15}"
        f"{counts[label]['train']:>10}"
        f"{counts[label]['val']:>10}"
        f"{counts[label]['test']:>10}"
    )

print("=" * 55)
print("Invalid sequences:", invalid)
