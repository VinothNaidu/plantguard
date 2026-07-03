"""
PlantGuard — evaluation (proposal Section 8).

Quantitative  (8.1): accuracy, precision, recall, F1 (macro + weighted),
                      per-class report, confusion matrix, mean inference time.
Qualitative   (8.2): Grad-CAM panels + side-by-side prediction grids;
                      a misclassified-sample gallery (failure modes).
Baseline      (8.3): compares MobileNetV2 vs the from-scratch baseline CNN.

Outputs are written to outputs/:
    metrics.json, classification_report.txt, confusion_matrix.png,
    gradcam_examples.png, misclassified_gallery.png, baseline_comparison.json

Run:
    python -m src.evaluate
    python -m src.evaluate --no-gradcam      # faster, skip heatmaps
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from . import config as C
from .data_prep import make_datasets
from .gradcam import GradCAM


# ----------------------------------------------------------------------
def _gather_predictions(model, test_ds):
    y_true, y_pred, y_prob = [], [], []
    t_total, n = 0.0, 0
    for images, labels in test_ds:
        t0 = time.perf_counter()
        probs = model.predict(images, verbose=0)
        t_total += time.perf_counter() - t0
        n += images.shape[0]
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(np.argmax(probs, axis=1).tolist())
        y_prob.extend(probs.tolist())
    ms_per_image = 1000.0 * t_total / max(n, 1)
    return (np.array(y_true), np.array(y_pred), np.array(y_prob), ms_per_image)


def quantitative(model, test_ds, class_names) -> dict:
    print("Running quantitative evaluation ...")
    y_true, y_pred, y_prob, ms = _gather_predictions(model, test_ds)

    acc = accuracy_score(y_true, y_pred)
    p_m, r_m, f_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)

    report_txt = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0, digits=4)
    (C.OUTPUTS_DIR / "classification_report.txt").write_text(report_txt)

    metrics = {
        "accuracy": round(float(acc), 4),
        "precision_macro": round(float(p_m), 4),
        "recall_macro": round(float(r_m), 4),
        "f1_macro": round(float(f_m), 4),
        "precision_weighted": round(float(p_w), 4),
        "recall_weighted": round(float(r_w), 4),
        "f1_weighted": round(float(f_w), 4),
        "inference_ms_per_image": round(float(ms), 3),
        "n_test": int(len(y_true)),
        "n_classes": len(class_names),
    }
    (C.OUTPUTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    _plot_confusion(y_true, y_pred, class_names)
    return metrics, y_true, y_pred, y_prob


def _plot_confusion(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.35), max(7, n * 0.35)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("PlantGuard — Confusion Matrix (test set)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    short = [c.replace("___", " | ")[:22] for c in class_names]
    ax.set_xticklabels(short, rotation=90, fontsize=6)
    ax.set_yticklabels(short, fontsize=6)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = C.OUTPUTS_DIR / "confusion_matrix.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"Saved {out}")


# ----------------------------------------------------------------------
def qualitative_gradcam(model, test_ds, class_names, n=8):
    print("Generating Grad-CAM examples ...")
    cam = GradCAM(model)
    images, labels = next(iter(test_ds))
    images = images[:n].numpy(); labels = labels[:n].numpy()

    cols = 3
    fig, axes = plt.subplots(n, cols, figsize=(cols * 3, n * 3))
    for i in range(n):
        batch = images[i:i + 1]
        probs = model.predict(batch, verbose=0)[0]
        pred = int(np.argmax(probs))
        hm = cam.heatmap(batch, pred)
        overlay = cam.overlay((images[i] * 255).astype(np.uint8), hm)

        axes[i, 0].imshow(images[i]); axes[i, 0].set_title(
            f"True: {class_names[labels[i]].split('___')[-1][:14]}", fontsize=8)
        axes[i, 1].imshow(hm, cmap="jet"); axes[i, 1].set_title("Grad-CAM", fontsize=8)
        ok = "✓" if pred == labels[i] else "✗"
        axes[i, 2].imshow(overlay); axes[i, 2].set_title(
            f"{ok} {class_names[pred].split('___')[-1][:14]}\n{probs[pred]:.2f}",
            fontsize=8)
        for j in range(cols):
            axes[i, j].axis("off")
    fig.tight_layout()
    out = C.OUTPUTS_DIR / "gradcam_examples.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"Saved {out}")


def misclassified_gallery(model, test_ds, class_names, y_true, y_pred, max_show=20):
    print("Building misclassified-sample gallery ...")
    # Re-collect images aligned with predictions (test_ds is unshuffled)
    imgs = []
    for batch_imgs, _ in test_ds:
        imgs.extend(batch_imgs.numpy())
        if len(imgs) >= len(y_true):
            break
    imgs = np.array(imgs[:len(y_true)])

    wrong = np.where(y_true != y_pred)[0][:max_show]
    if len(wrong) == 0:
        print("No misclassifications on the test set — skipping gallery.")
        return
    cols = 4
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for k, idx in enumerate(wrong):
        ax = axes[k // cols, k % cols]
        ax.imshow(imgs[idx]); ax.axis("off")
        ax.set_title(
            f"T:{class_names[y_true[idx]].split('___')[-1][:12]}\n"
            f"P:{class_names[y_pred[idx]].split('___')[-1][:12]}",
            fontsize=7, color="crimson")
    fig.suptitle("Misclassified samples (failure-mode analysis)", fontsize=12)
    fig.tight_layout()
    out = C.OUTPUTS_DIR / "misclassified_gallery.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"Saved {out}")


# ----------------------------------------------------------------------
def baseline_comparison(test_ds, class_names, mobilenet_metrics):
    if not C.BASELINE_MODEL.exists():
        print("No baseline model found — skipping comparison "
              "(train with: python -m src.train --baseline).")
        return
    print("Evaluating baseline CNN for comparison ...")
    base = tf.keras.models.load_model(C.BASELINE_MODEL)
    y_true, y_pred, _, ms = _gather_predictions(base, test_ds)
    acc = accuracy_score(y_true, y_pred)
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)

    comparison = {
        "MobileNetV2_transfer": {
            "accuracy": mobilenet_metrics["accuracy"],
            "f1_macro": mobilenet_metrics["f1_macro"],
            "inference_ms_per_image": mobilenet_metrics["inference_ms_per_image"],
        },
        "Baseline_CNN_scratch": {
            "accuracy": round(float(acc), 4),
            "f1_macro": round(float(f1), 4),
            "inference_ms_per_image": round(float(ms), 3),
        },
        "literature_reference": {
            "Simon_et_al_2020_basic_CNN_accuracy": 0.88
        },
    }
    out = C.OUTPUTS_DIR / "baseline_comparison.json"
    out.write_text(json.dumps(comparison, indent=2))
    print(json.dumps(comparison, indent=2))
    print(f"Saved {out}")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Evaluate PlantGuard")
    ap.add_argument("--no-gradcam", action="store_true")
    args = ap.parse_args()

    _, _, test_ds, class_names = make_datasets()
    model = tf.keras.models.load_model(C.MOBILENET_MODEL)

    metrics, y_true, y_pred, _ = quantitative(model, test_ds, class_names)

    if not args.no_gradcam:
        qualitative_gradcam(model, test_ds, class_names)
    misclassified_gallery(model, test_ds, class_names, y_true, y_pred)
    baseline_comparison(test_ds, class_names, metrics)

    print("\nEvaluation complete. See the outputs/ folder.")


if __name__ == "__main__":
    main()
