# 🌿 PlantGuard — Intelligent Plant Disease Recognition System

**CDS6334 Visual Information Processing · Track A · Group 17 (TT2L)**

PlantGuard takes a leaf image and returns the predicted disease class, a
confidence score, a Grad-CAM explanation, and a basic management
recommendation. It uses **MobileNetV2 transfer learning** fine-tuned on the
**PlantVillage** dataset (38 classes, 14 crops), with leaf **segmentation**
for background removal and a from-scratch **baseline CNN** for comparison.

---

## 1. Pipeline (matches the proposal)

```
RGB leaf image
   └─▶ Stage 2  Preprocess   resize 224×224, normalize [0,1], augment
        └─▶ Stage 3  Segment      HSV + GrabCut leaf isolation
             └─▶ Stage 4  Classify    MobileNetV2 (transfer learning) → softmax
                  └─▶ Stage 5  Output     class + confidence + Grad-CAM + advice
```

---

## 2. Project structure

```
plantguard/
├── src/
│   ├── config.py          # all paths & hyperparameters
│   ├── download_data.py    # fetch PlantVillage (kagglehub)
│   ├── data_prep.py        # stratified 70/15/15 split + tf.data pipelines
│   ├── segmentation.py     # HSV + GrabCut leaf isolation
│   ├── model.py            # MobileNetV2 transfer model + baseline CNN
│   ├── gradcam.py          # Grad-CAM (Keras-3 nested-backbone safe)
│   ├── disease_info.py     # class → crop/disease/treatment metadata
│   ├── inference.py        # end-to-end predict() used by app + eval
│   ├── train.py            # two-phase transfer learning + baseline
│   └── evaluate.py         # quantitative + qualitative + baseline compare
├── app/
│   └── streamlit_app.py    # interactive demo (upload / camera)
├── notebooks/
│   └── PlantGuard_Colab.ipynb   # one-click GPU training on Colab
├── models/                 # trained .keras files + class_names.json
├── outputs/                # metrics, confusion matrix, Grad-CAM figures
├── requirements.txt
├── PEER_EVALUATION_TEMPLATE.md
└── README.md
```

---

## 3. Setup

```bash
git clone <your-repo-url> plantguard
cd plantguard
python -m venv venv && source venv/bin/activate     # optional
pip install -r requirements.txt
```

> **GPU strongly recommended.** On Colab, just open
> `notebooks/PlantGuard_Colab.ipynb` and Run All.

---

## 4. Get the dataset

**Option A — automatic (kagglehub):**
```bash
pip install kagglehub
python -m src.download_data --source kaggle
```

**Option B — manual:** download from
<https://github.com/spMohanty/PlantVillage-Dataset> or the Kaggle mirror,
and unzip so the class folders sit directly under `data/PlantVillage/`:

```
data/PlantVillage/Apple___Apple_scab/*.jpg
data/PlantVillage/Tomato___healthy/*.jpg
...
```

Then build the stratified split:
```bash
python -m src.data_prep --build-split
```

---

## 5. Train

```bash
# MobileNetV2 (two-phase transfer learning)
python -m src.train

# also train the from-scratch baseline CNN for comparison
python -m src.train --baseline
```

Training is two-phase: (1) frozen backbone + new head, then (2) fine-tune
the top of MobileNetV2 at a low learning rate. The best model is saved to
`models/plantguard_mobilenetv2.keras`.

Useful flags: `--epochs-head`, `--epochs-finetune`, `--epochs-baseline`,
`--baseline-only`.

---

## 6. Evaluate

```bash
python -m src.evaluate
```

Produces in `outputs/`:

| File | Content |
|------|---------|
| `metrics.json` | accuracy, precision/recall/F1 (macro & weighted), inference ms/image |
| `classification_report.txt` | per-class precision/recall/F1 |
| `confusion_matrix.png` | full 38×38 confusion matrix |
| `gradcam_examples.png` | input / heatmap / overlay panels |
| `misclassified_gallery.png` | up to 20 failure cases for analysis |
| `baseline_comparison.json` | MobileNetV2 vs scratch CNN vs literature (Simon et al. 2020, 88%) |

---

## 7. Run the demo app

```bash
streamlit run app/streamlit_app.py
```

Upload or capture a leaf → get prediction, confidence, top-3, segmentation
preview, Grad-CAM heatmap, and a management recommendation. Background
removal and Grad-CAM are toggleable in the sidebar.

---

## 8. Mapping to evaluation requirements

- **Quantitative** (8.1): accuracy, precision, recall, F1, confusion matrix, inference time → `src/evaluate.py`.
- **Qualitative** (8.2): Grad-CAM panels + misclassified-sample failure-mode gallery.
- **Baseline comparison** (8.3): scratch CNN + Simon et al. (2020) reference.
- **Dataset** (Sec 7): PlantVillage, 54k images ≫ 400 minimum, preprocessing documented.
- **Every member contributes technically** — see Task Distribution in the proposal.

---

## 9. Notes & ethics

- Treatment recommendations are **educational summaries**, not professional
  agricultural advice — confirm with a local extension officer before
  applying chemicals.
- Pretrained MobileNetV2 (ImageNet) weights are used under their standard
  license; this is acknowledged as external pretrained-model use.
- Real-world generalization is a known open challenge (Ahmed et al. 2021
  saw 98.79% → 82.47% lab→field); segmentation + aggressive augmentation
  are included to narrow that gap.

---

## 10. Team (Group 17)

| Member | Primary responsibility |
|--------|------------------------|
| Muhammad Iz'aan Khan Bin Mubarak | Model selection, transfer-learning training, tuning, evaluation |
| Dinesh Waren a/l Rajasingam | Dataset, preprocessing pipeline, augmentation, split |
| Paviteran A/L Arumugam | System integration, Streamlit UI, Grad-CAM implementation |
| Vinoth Naidu A/L Sri Padmanathan | Quantitative evaluation, confusion-matrix analysis, report, slides |
