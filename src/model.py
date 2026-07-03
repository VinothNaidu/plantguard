"""
PlantGuard — model architectures (proposal Stage 4 / Section 6).

  * build_mobilenetv2(): MobileNetV2 backbone (ImageNet weights) +
    custom classification head. Supports two-phase transfer learning.
  * build_baseline_cnn(): a small CNN trained from scratch, used for the
    baseline comparison in Section 8.3.

Inputs are expected already normalized to [0, 1]. MobileNetV2 needs
[-1, 1], so a Rescaling layer is baked into the model — the same model
therefore works directly inside the Streamlit app on [0,1] inputs.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models

from . import config as C


# ----------------------------------------------------------------------
# MobileNetV2 transfer-learning model
# ----------------------------------------------------------------------
def build_mobilenetv2(num_classes: int) -> tf.keras.Model:
    inputs = layers.Input(shape=(C.IMG_SIZE, C.IMG_SIZE, C.CHANNELS), name="input_image")

    # [0,1] -> [-1,1] for MobileNetV2 preprocessing
    x = layers.Rescaling(2.0, offset=-1.0, name="to_pm1")(inputs)

    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(C.IMG_SIZE, C.IMG_SIZE, C.CHANNELS),
        include_top=False,
        weights="imagenet",
    )
    backbone.trainable = False           # phase 1: frozen
    x = backbone(x, training=False)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(C.DROPOUT, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name="PlantGuard_MobileNetV2")
    model._backbone_name = backbone.name   # convenience handle
    return model


def compile_head_phase(model: tf.keras.Model) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(C.HEAD_LR),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )


def unfreeze_for_finetune(model: tf.keras.Model) -> None:
    """Unfreeze the top of the MobileNetV2 backbone for fine-tuning."""
    backbone = model.get_layer("mobilenetv2_1.00_224")
    backbone.trainable = True
    for layer in backbone.layers[:C.FINETUNE_AT]:
        layer.trainable = False
    # BatchNorm layers stay frozen for stable fine-tuning
    for layer in backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(C.FINETUNE_LR),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )


# ----------------------------------------------------------------------
# Baseline CNN (from scratch — no transfer learning)
# ----------------------------------------------------------------------
def build_baseline_cnn(num_classes: int) -> tf.keras.Model:
    model = models.Sequential(
        [
            layers.Input(shape=(C.IMG_SIZE, C.IMG_SIZE, C.CHANNELS)),
            layers.Conv2D(32, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),

            layers.Conv2D(64, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),

            layers.Conv2D(128, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),

            layers.Conv2D(128, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),

            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.4),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="Baseline_CNN",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(C.BASELINE_LR),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model
