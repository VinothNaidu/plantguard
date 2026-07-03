"""
PlantGuard — end-to-end inference (proposal Stages 1-5).

Wraps the full prediction pipeline so both the Streamlit app and the
evaluation script use identical logic:

    raw image -> (optional segmentation) -> resize/normalize ->
    MobileNetV2 -> top-k classes + confidence -> Grad-CAM overlay
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from . import config as C
from .disease_info import get_disease_info
from .gradcam import GradCAM
from .segmentation import segment_leaf


@lru_cache(maxsize=2)
def load_model(path: str | None = None) -> tf.keras.Model:
    path = Path(path) if path else C.MOBILENET_MODEL
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Train first: python -m src.train"
        )
    return tf.keras.models.load_model(path)


@lru_cache(maxsize=1)
def load_class_names() -> tuple[str, ...]:
    if not C.CLASS_NAMES_JSON.exists():
        raise FileNotFoundError("class_names.json missing — train first.")
    return tuple(json.loads(C.CLASS_NAMES_JSON.read_text()))


def preprocess(rgb: np.ndarray, do_segment: bool = True):
    """Return (model_input (1,H,W,3) float[0,1], display_rgb uint8, mask)."""
    rgb = np.asarray(rgb)[:, :, :3].astype(np.uint8)
    mask = None
    seg = rgb
    if do_segment:
        seg, mask = segment_leaf(rgb)
    resized = cv2.resize(seg, (C.IMG_SIZE, C.IMG_SIZE))
    norm = resized.astype(np.float32) / 255.0
    return np.expand_dims(norm, 0), resized, mask


def predict(
    rgb: np.ndarray,
    model: tf.keras.Model | None = None,
    do_segment: bool = True,
    do_gradcam: bool = True,
    top_k: int = 3,
) -> dict:
    """Run the full pipeline on one RGB image and return a result dict."""
    model = model or load_model()
    class_names = load_class_names()

    batch, display_rgb, mask = preprocess(rgb, do_segment=do_segment)

    probs = model.predict(batch, verbose=0)[0]
    order = np.argsort(probs)[::-1][:top_k]
    top = [
        {
            "label": class_names[i],
            "confidence": float(probs[i]),
            **get_disease_info(class_names[i]),
        }
        for i in order
    ]

    result = {
        "top_prediction": top[0],
        "topk": top,
        "all_probs": probs.tolist(),
        "segmented_input": display_rgb,
        "mask": mask,
        "gradcam_overlay": None,
    }

    if do_gradcam:
        try:
            cam = GradCAM(model)
            hm = cam.heatmap(batch, class_index=int(order[0]))
            result["gradcam_overlay"] = cam.overlay(display_rgb, hm)
        except Exception as e:           # never let viz break inference
            result["gradcam_error"] = str(e)

    return result
