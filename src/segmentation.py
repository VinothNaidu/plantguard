"""
PlantGuard — leaf segmentation / background removal (proposal Stage 3).

The literature (Chowdhury et al. 2021; Ahmed et al. 2021) shows that
removing background before classification reduces the lab-to-field
generalization gap. We provide a fast, dependency-light segmentation
that does NOT require a trained U-Net, so the demo runs anywhere:

    1. HSV thresholding to keep green/yellow/brown leaf-coloured pixels.
    2. GrabCut refinement seeded by that mask.
    3. Morphological cleanup + largest-component selection.

`segment_leaf` returns an RGB image with the background set to a neutral
colour, ready to be fed to the classifier. If anything fails it returns
the original image (fail-safe), so the pipeline never crashes on an
unusual input.
"""
from __future__ import annotations

import cv2
import numpy as np


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected foreground blob."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return mask
    # index 0 is background; pick largest of the rest by area
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest, 255, 0).astype(np.uint8)


def _hsv_leaf_mask(rgb: np.ndarray) -> np.ndarray:
    """Coarse leaf mask from HSV colour thresholding."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    # Green-ish healthy tissue
    green = cv2.inRange(hsv, (25, 30, 30), (95, 255, 255))
    # Yellow / chlorotic tissue
    yellow = cv2.inRange(hsv, (15, 30, 30), (35, 255, 255))
    # Brown / necrotic lesions (low-mid hue, lower value)
    brown = cv2.inRange(hsv, (5, 30, 20), (25, 255, 200))

    mask = cv2.bitwise_or(cv2.bitwise_or(green, yellow), brown)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def segment_leaf(
    rgb: np.ndarray,
    bg_color: tuple[int, int, int] = (0, 0, 0),
    use_grabcut: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Isolate the leaf from the background.

    Args:
        rgb: H×W×3 uint8 RGB image.
        bg_color: colour to paint the removed background.
        use_grabcut: refine the HSV mask with GrabCut (slower, cleaner).

    Returns:
        (segmented_rgb, mask) where mask is a uint8 {0,255} foreground mask.
    """
    try:
        assert rgb.ndim == 3 and rgb.shape[2] == 3
        rgb = rgb.astype(np.uint8)
        h, w = rgb.shape[:2]

        coarse = _hsv_leaf_mask(rgb)
        if coarse.sum() < 0.02 * 255 * h * w:
            # Too little leaf detected — bail out, return original
            return rgb, np.full((h, w), 255, np.uint8)

        if use_grabcut:
            gc_mask = np.where(coarse > 0, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
            # Seed strong foreground at eroded core, strong background at borders
            core = cv2.erode(coarse, np.ones((15, 15), np.uint8))
            gc_mask[core > 0] = cv2.GC_FGD
            border = np.ones((h, w), np.uint8)
            border[10:h - 10, 10:w - 10] = 0
            gc_mask[(border > 0) & (coarse == 0)] = cv2.GC_BGD

            bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
            try:
                cv2.grabCut(rgb, gc_mask, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
                mask = np.where(
                    (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
                ).astype(np.uint8)
            except cv2.error:
                mask = coarse
        else:
            mask = coarse

        mask = _largest_component(mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        out = rgb.copy()
        out[mask == 0] = bg_color
        return out, mask
    except Exception:
        # Absolute fail-safe — never break the pipeline
        h, w = rgb.shape[:2]
        return rgb, np.full((h, w), 255, np.uint8)
