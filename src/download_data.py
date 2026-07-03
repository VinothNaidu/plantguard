"""
PlantGuard — dataset download helper.

Downloads the PlantVillage dataset into data/PlantVillage/<class>/*.jpg.

Two options:

1) KaggleHub (recommended, no manual files):
       pip install kagglehub
       python -m src.download_data --source kaggle
   Uses the widely-mirrored "abdallahalidev/plantvillage-dataset"
   (color images, 38 classes, ~54k images).

2) Manual:
   Download from either of these and unzip so that the class folders sit
   directly under data/PlantVillage/:
       https://github.com/spMohanty/PlantVillage-Dataset
       https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset

After downloading, build the split:
       python -m src.data_prep --build-split
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from . import config as C


def _flatten_to_target(found_root: Path, target: Path):
    """Move <found_root>/<class>/* into data/PlantVillage/<class>/*."""
    target.mkdir(parents=True, exist_ok=True)
    moved = 0
    for cls_dir in found_root.iterdir():
        if cls_dir.is_dir():
            dest = target / cls_dir.name
            if dest.exists():
                continue
            shutil.move(str(cls_dir), str(dest))
            moved += 1
    print(f"Linked {moved} class folders into {target}")


def download_kaggle():
    try:
        import kagglehub
    except ImportError:
        raise SystemExit("Run: pip install kagglehub")

    print("Downloading PlantVillage via kagglehub ...")
    path = Path(kagglehub.dataset_download(
        "abdallahalidev/plantvillage-dataset"))
    print(f"Downloaded to cache: {path}")

    # The Kaggle archive nests color/grayscale/segmented variants.
    # Find the 'color' directory that holds the class folders.
    candidates = list(path.rglob("color"))
    src_root = candidates[0] if candidates else path
    # If 'color' wraps a single 'PlantVillage' folder, dive in.
    inner = list(src_root.glob("*"))
    if len(inner) == 1 and inner[0].is_dir():
        src_root = inner[0]

    target = C.DATA_DIR / "PlantVillage"
    _flatten_to_target(src_root, target)
    n = sum(1 for _ in target.rglob("*.jpg")) + sum(1 for _ in target.rglob("*.JPG"))
    print(f"Done. ~{n} jpg images under {target}")
    print("Next: python -m src.data_prep --build-split")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["kaggle"], default="kaggle")
    args = ap.parse_args()
    if args.source == "kaggle":
        download_kaggle()
