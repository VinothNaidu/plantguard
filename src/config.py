"""
PlantGuard — central configuration.

All paths, hyperparameters, and constants live here so that training,
evaluation, and the Streamlit app stay in sync.
"""
from pathlib import Path

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"                 # expects data/PlantVillage/<class>/*.jpg
SPLIT_DIR = ROOT / "data_split"          # auto-created: train/ val/ test/
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

for _d in (MODELS_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Image / model parameters
# ----------------------------------------------------------------------
IMG_SIZE = 224                 # MobileNetV2 native input
CHANNELS = 3
BATCH_SIZE = 32
SEED = 42

# Stratified split ratios (proposal: 70 / 15 / 15)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
# Two-phase transfer learning
HEAD_EPOCHS = 8                # phase 1: frozen backbone, train head
FINETUNE_EPOCHS = 12           # phase 2: unfreeze top layers
FINETUNE_AT = 100              # unfreeze MobileNetV2 layers from this index up
HEAD_LR = 1e-3
FINETUNE_LR = 1e-5
DROPOUT = 0.3
LABEL_SMOOTHING = 0.1

# Baseline CNN (trained from scratch, no transfer learning)
BASELINE_EPOCHS = 20
BASELINE_LR = 1e-3

# ----------------------------------------------------------------------
# Artifact filenames
# ----------------------------------------------------------------------
MOBILENET_MODEL = MODELS_DIR / "plantguard_mobilenetv2.keras"
BASELINE_MODEL = MODELS_DIR / "baseline_cnn.keras"
CLASS_NAMES_JSON = MODELS_DIR / "class_names.json"
HISTORY_JSON = OUTPUTS_DIR / "training_history.json"

# Last conv layer used for Grad-CAM on MobileNetV2
GRADCAM_LAYER = "Conv_1"
