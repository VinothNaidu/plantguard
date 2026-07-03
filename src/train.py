"""
PlantGuard — training (proposal Stage 4, Section 8.3).

Two-phase transfer learning for MobileNetV2:
    Phase 1 — freeze backbone, train the new classification head.
    Phase 2 — unfreeze the top of the backbone and fine-tune at a low LR.

Optionally also trains the from-scratch baseline CNN used for the
baseline comparison in the report.

Examples:
    python -m src.train                 # train MobileNetV2 only
    python -m src.train --baseline      # also train the baseline CNN
    python -m src.train --epochs-head 6 --epochs-finetune 10
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import tensorflow as tf

from . import config as C
from .data_prep import make_datasets
from .model import (
    build_baseline_cnn,
    build_mobilenetv2,
    compile_head_phase,
    unfreeze_for_finetune,
)


def _callbacks(ckpt_path, patience=4):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            str(ckpt_path), monitor="val_accuracy",
            save_best_only=True, mode="max", verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=patience,
            restore_best_weights=True, mode="max", verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1,
        ),
    ]


def _merge_history(h1, h2):
    out = {}
    for k in h1.history:
        out[k] = h1.history[k] + h2.history.get(k, [])
    return out


def train_mobilenet(train_ds, val_ds, class_names,
                    epochs_head, epochs_finetune):
    model = build_mobilenetv2(len(class_names))
    compile_head_phase(model)
    model.summary()

    print("\n=== Phase 1: training classification head (frozen backbone) ===")
    h1 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=epochs_head, callbacks=_callbacks(C.MOBILENET_MODEL),
    )

    print("\n=== Phase 2: fine-tuning top of MobileNetV2 ===")
    unfreeze_for_finetune(model)
    h2 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=epochs_finetune, callbacks=_callbacks(C.MOBILENET_MODEL),
    )

    model.save(C.MOBILENET_MODEL)
    history = _merge_history(h1, h2)
    C.HISTORY_JSON.write_text(json.dumps(history, indent=2))
    print(f"\nSaved MobileNetV2 -> {C.MOBILENET_MODEL}")
    print(f"Saved history     -> {C.HISTORY_JSON}")
    return model, history


def train_baseline(train_ds, val_ds, class_names, epochs):
    print("\n=== Baseline CNN (from scratch, no transfer learning) ===")
    model = build_baseline_cnn(len(class_names))
    model.fit(
        train_ds, validation_data=val_ds,
        epochs=epochs, callbacks=_callbacks(C.BASELINE_MODEL, patience=5),
    )
    model.save(C.BASELINE_MODEL)
    print(f"Saved baseline CNN -> {C.BASELINE_MODEL}")
    return model


def main():
    ap = argparse.ArgumentParser(description="Train PlantGuard models")
    ap.add_argument("--epochs-head", type=int, default=C.HEAD_EPOCHS)
    ap.add_argument("--epochs-finetune", type=int, default=C.FINETUNE_EPOCHS)
    ap.add_argument("--epochs-baseline", type=int, default=C.BASELINE_EPOCHS)
    ap.add_argument("--baseline", action="store_true",
                    help="Also train the from-scratch baseline CNN")
    ap.add_argument("--baseline-only", action="store_true")
    args = ap.parse_args()

    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs available: {gpus or 'none (running on CPU)'}")

    t0 = time.time()
    train_ds, val_ds, test_ds, class_names = make_datasets()
    print(f"Classes ({len(class_names)}): {class_names[:5]} ...")

    if not args.baseline_only:
        train_mobilenet(train_ds, val_ds, class_names,
                        args.epochs_head, args.epochs_finetune)

    if args.baseline or args.baseline_only:
        train_baseline(train_ds, val_ds, class_names, args.epochs_baseline)

    print(f"\nTotal wall-clock: {(time.time() - t0) / 60:.1f} min")
    print("Next: python -m src.evaluate")


if __name__ == "__main__":
    main()
