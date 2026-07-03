"""
PlantGuard — dataset preparation and tf.data input pipelines.

Responsibilities (proposal Stage 1 & 2 + Section 7):
  * Build a stratified 70/15/15 train/val/test split from a flat
    class-folder dataset (PlantVillage layout).
  * Provide tf.data pipelines with resize -> [0,1] normalize ->
    augmentation (train only).

Expected raw layout:
    data/PlantVillage/
        Apple___Apple_scab/ *.jpg
        Apple___healthy/    *.jpg
        ...

Run directly to create the split:
    python -m src.data_prep --build-split
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import tensorflow as tf

from . import config as C

AUTOTUNE = tf.data.AUTOTUNE
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ----------------------------------------------------------------------
# Stratified split
# ----------------------------------------------------------------------
def _gather_by_class(raw_dir: Path) -> dict[str, list[Path]]:
    classes = defaultdict(list)
    for cls_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        for img in cls_dir.iterdir():
            if img.suffix.lower() in IMG_EXTS:
                classes[cls_dir.name].append(img)
    if not classes:
        raise FileNotFoundError(
            f"No class folders with images found under {raw_dir}. "
            "Expected data/PlantVillage/<class>/*.jpg"
        )
    return classes


def build_stratified_split(
    raw_dir: Path | None = None,
    out_dir: Path | None = None,
    copy: bool = True,
) -> None:
    """Create train/val/test folders with a per-class stratified split."""
    raw_dir = raw_dir or (C.DATA_DIR / "PlantVillage")
    out_dir = out_dir or C.SPLIT_DIR
    random.seed(C.SEED)

    classes = _gather_by_class(raw_dir)
    print(f"Found {len(classes)} classes, "
          f"{sum(len(v) for v in classes.values())} images.")

    if out_dir.exists():
        shutil.rmtree(out_dir)

    counts = {"train": 0, "val": 0, "test": 0}
    for cls, files in classes.items():
        random.shuffle(files)
        n = len(files)
        n_train = int(n * C.TRAIN_RATIO)
        n_val = int(n * C.VAL_RATIO)
        partitions = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:],
        }
        for split, items in partitions.items():
            dest_dir = out_dir / split / cls
            dest_dir.mkdir(parents=True, exist_ok=True)
            for src in items:
                dest = dest_dir / src.name
                if copy:
                    shutil.copy2(src, dest)
                else:
                    # symlink to save disk space on large datasets
                    dest.symlink_to(src.resolve())
            counts[split] += len(items)

    print(f"Split complete -> {dict(counts)}")
    print(f"Written to: {out_dir}")

    # Persist canonical class order
    class_names = sorted(classes.keys())
    C.CLASS_NAMES_JSON.write_text(json.dumps(class_names, indent=2))
    print(f"Saved {len(class_names)} class names -> {C.CLASS_NAMES_JSON}")


# ----------------------------------------------------------------------
# Augmentation (proposal Section 7.1)
# ----------------------------------------------------------------------
def _augmenter() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(15 / 360.0),     # ±15°
            tf.keras.layers.RandomZoom(0.10),               # 10% zoom
            tf.keras.layers.RandomBrightness(0.15, value_range=(0.0, 1.0)),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="augmentation",
    )


# ----------------------------------------------------------------------
# tf.data pipelines
# ----------------------------------------------------------------------
def _normalize(image, label):
    image = tf.cast(image, tf.float32) / 255.0   # -> [0, 1]
    return image, label


def make_datasets(split_dir: Path | None = None):
    """Return (train_ds, val_ds, test_ds, class_names)."""
    split_dir = split_dir or C.SPLIT_DIR
    if not split_dir.exists():
        raise FileNotFoundError(
            f"{split_dir} not found. Run: python -m src.data_prep --build-split"
        )

    common = dict(
        image_size=(C.IMG_SIZE, C.IMG_SIZE),
        batch_size=C.BATCH_SIZE,
        label_mode="int",
        seed=C.SEED,
    )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        split_dir / "train", shuffle=True, **common)
    class_names = train_ds.class_names

    val_ds = tf.keras.utils.image_dataset_from_directory(
        split_dir / "val", shuffle=False, **common)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        split_dir / "test", shuffle=False, **common)

    aug = _augmenter()
    train_ds = (
        train_ds.map(_normalize, num_parallel_calls=AUTOTUNE)
        .map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )
    val_ds = val_ds.map(_normalize, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    test_ds = test_ds.map(_normalize, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    # Keep class order canonical and persisted
    C.CLASS_NAMES_JSON.write_text(json.dumps(class_names, indent=2))
    return train_ds, val_ds, test_ds, class_names


def load_class_names() -> list[str]:
    if C.CLASS_NAMES_JSON.exists():
        return json.loads(C.CLASS_NAMES_JSON.read_text())
    raise FileNotFoundError(
        "class_names.json missing. Build the split or train first."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PlantGuard data prep")
    parser.add_argument("--build-split", action="store_true",
                        help="Create stratified train/val/test split")
    parser.add_argument("--raw", type=str, default=None,
                        help="Raw dataset dir (default data/PlantVillage)")
    parser.add_argument("--symlink", action="store_true",
                        help="Symlink instead of copy (saves disk)")
    args = parser.parse_args()

    if args.build_split:
        build_stratified_split(
            raw_dir=Path(args.raw) if args.raw else None,
            copy=not args.symlink,
        )
    else:
        parser.print_help()
