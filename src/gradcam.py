"""
PlantGuard — Grad-CAM visualization (proposal Stage 5 / Section 8.2).

Produces a class-activation heatmap that highlights the leaf regions
driving a prediction, so users can verify the model attends to lesions
rather than background artefacts.

Keras 3 note: the MobileNetV2 backbone is a *nested* model, so the last
conv feature map is not directly reachable from the top model's symbolic
graph. We therefore split the network: run the backbone up to the target
conv layer, then re-apply the post-backbone head, so a single
GradientTape connects the prediction back to the feature map.

Usage:
    cam = GradCAM(model)
    heatmap = cam.heatmap(img_batch, class_index)   # H×W float [0,1]
    overlay = cam.overlay(rgb_uint8, heatmap)        # H×W×3 uint8
"""
from __future__ import annotations

import cv2
import numpy as np
import tensorflow as tf

from . import config as C


class GradCAM:
    def __init__(self, model: tf.keras.Model, layer_name: str | None = None):
        self.model = model
        self.backbone = self._find_backbone()
        self.layer_name = layer_name or self._resolve_layer_name()
        self._pre_layers = []
        self._top_layers = []
        self.conv_model = self._build_conv_model()

    def _find_backbone(self):
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.Model):
                return layer
        return None

    def _resolve_layer_name(self) -> str:
        if self.backbone is not None:
            try:
                self.backbone.get_layer(C.GRADCAM_LAYER)
                return C.GRADCAM_LAYER
            except ValueError:
                for layer in reversed(self.backbone.layers):
                    if hasattr(layer, "output") and len(layer.output.shape) == 4:
                        return layer.name
        for layer in reversed(self.model.layers):
            if hasattr(layer, "output") and len(layer.output.shape) == 4:
                return layer.name
        raise ValueError("No 4D conv layer found for Grad-CAM.")

    def _build_conv_model(self):
        if self.backbone is not None:
            target = self.backbone.get_layer(self.layer_name)
            # pre-backbone layers (e.g. Rescaling) and post-backbone head
            seen = False
            for layer in self.model.layers:
                if layer is self.backbone:
                    seen = True
                    continue
                if isinstance(layer, tf.keras.layers.InputLayer):
                    continue
                (self._top_layers if seen else self._pre_layers).append(layer)
            return tf.keras.models.Model(
                self.backbone.inputs,
                [target.output, self.backbone.output],
            )
        target = self.model.get_layer(self.layer_name)
        return tf.keras.models.Model(
            self.model.inputs, [target.output, self.model.output])

    def heatmap(self, img_batch: np.ndarray, class_index: int | None = None) -> np.ndarray:
        """img_batch: (1, H, W, 3) normalized [0,1]. Returns H×W heatmap [0,1]."""
        img_tensor = tf.convert_to_tensor(img_batch, dtype=tf.float32)
        with tf.GradientTape() as tape:
            if self.backbone is not None:
                x = img_tensor
                for layer in self._pre_layers:
                    x = layer(x, training=False)
                conv_out, backbone_out = self.conv_model(x, training=False)
                preds = backbone_out
                for layer in self._top_layers:
                    preds = layer(preds, training=False)
            else:
                conv_out, preds = self.conv_model(img_tensor, training=False)
            if class_index is None:
                class_index = int(tf.argmax(preds[0]))
            loss = preds[:, class_index]

        grads = tape.gradient(loss, conv_out)
        if grads is None:
            return np.zeros((C.IMG_SIZE, C.IMG_SIZE), np.float32)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam = tf.reduce_sum(conv_out[0] * pooled, axis=-1)
        cam = tf.nn.relu(cam)
        cam = cam / (tf.reduce_max(cam) + 1e-8)
        cam = cv2.resize(cam.numpy(), (C.IMG_SIZE, C.IMG_SIZE))
        return cam.astype(np.float32)

    @staticmethod
    def overlay(rgb_uint8: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """Blend a [0,1] heatmap over an RGB image."""
        hm = np.uint8(255 * heatmap)
        hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
        hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
        if rgb_uint8.shape[:2] != hm.shape[:2]:
            rgb_uint8 = cv2.resize(rgb_uint8, (hm.shape[1], hm.shape[0]))
        return cv2.addWeighted(rgb_uint8, 1 - alpha, hm, alpha, 0)
