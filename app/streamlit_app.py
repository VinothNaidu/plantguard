"""
PlantGuard — Streamlit demo app (proposal Stage 5 / Expected Outcomes).

Upload or capture a leaf image and get:
  * predicted disease class + confidence
  * top-3 alternatives
  * segmentation preview (background removed)
  * Grad-CAM heatmap showing the regions that drove the prediction
  * a basic management recommendation

Run:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# Make `src` importable when run via `streamlit run app/streamlit_app.py`
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config as C            # noqa: E402
from src.inference import predict, load_model, load_class_names  # noqa: E402

st.set_page_config(
    page_title="PlantGuard — Plant Disease Recognition",
    page_icon="🌿",
    layout="wide",
)

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.markdown(
    """
    <div style="padding:0.5rem 0 1rem 0">
      <h1 style="margin-bottom:0">🌿 PlantGuard</h1>
      <p style="color:#4b7d52;font-size:1.05rem;margin-top:0.2rem">
        Intelligent Plant Disease Recognition System — MobileNetV2 · PlantVillage · Grad-CAM
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    do_segment = st.toggle("Background removal (segmentation)", value=True)
    do_gradcam = st.toggle("Grad-CAM explanation", value=True)
    top_k = st.slider("Show top-K predictions", 1, 5, 3)
    st.divider()
    st.caption(
        "Model: MobileNetV2 (transfer learning, fine-tuned on PlantVillage).\n\n"
        "Recommendations are educational summaries, not professional "
        "agricultural advice."
    )

    model_ok = C.MOBILENET_MODEL.exists()
    if model_ok:
        st.success("Model loaded ✓")
    else:
        st.error(
            "No trained model found.\n\n"
            "Train first:\n`python -m src.train`\n\n"
            "then place the `.keras` file in `models/`."
        )

# ----------------------------------------------------------------------
# Input
# ----------------------------------------------------------------------
tab_upload, tab_camera = st.tabs(["📁 Upload image", "📷 Use camera"])
with tab_upload:
    uploaded = st.file_uploader(
        "Upload a leaf image", type=["jpg", "jpeg", "png", "bmp"])
with tab_camera:
    snapshot = st.camera_input("Capture a leaf")

image_file = uploaded or snapshot


def _confidence_color(c: float) -> str:
    return "#2e7d32" if c >= 0.75 else "#f9a825" if c >= 0.5 else "#c62828"


# ----------------------------------------------------------------------
# Run inference
# ----------------------------------------------------------------------
if image_file is not None and model_ok:
    rgb = np.array(Image.open(image_file).convert("RGB"))

    with st.spinner("Analysing leaf ..."):
        result = predict(
            rgb,
            do_segment=do_segment,
            do_gradcam=do_gradcam,
            top_k=top_k,
        )

    top = result["top_prediction"]
    conf = top["confidence"]

    # --- headline result ---
    c1, c2 = st.columns([1, 1])
    with c1:
        st.image(rgb, caption="Input image", use_container_width=True)
    with c2:
        st.markdown(
            f"""
            <div style="border:1px solid #d7e6d9;border-radius:14px;padding:1.1rem 1.3rem;background:#f6fbf6">
              <div style="color:#6b6b6b;font-size:0.85rem">Predicted</div>
              <div style="font-size:1.5rem;font-weight:700;color:#234a28">
                {top['crop']} — {top['disease']}
              </div>
              <div style="margin:0.6rem 0">
                <span style="color:{_confidence_color(conf)};font-size:1.2rem;font-weight:700">
                  {conf*100:.1f}% confidence
                </span>
                <span style="color:#888"> · {top['status']}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(conf, 1.0))
        st.markdown("**Recommendation**")
        st.info(top["recommendation"])
        if conf < 0.5:
            st.warning(
                "Low confidence — try a clearer, well-lit close-up of a single "
                "leaf, or confirm with an expert."
            )

    st.divider()

    # --- top-K table ---
    st.subheader("Top predictions")
    for p in result["topk"]:
        st.markdown(
            f"- **{p['crop']} — {p['disease']}** · "
            f"{p['confidence']*100:.1f}%"
        )

    # --- visual diagnostics ---
    st.subheader("Visual diagnostics")
    cols = st.columns(3)
    cols[0].image(result["segmented_input"],
                  caption="Segmented input (model sees this)",
                  use_container_width=True)
    if result.get("mask") is not None:
        cols[1].image(result["mask"], caption="Leaf mask",
                      use_container_width=True, clamp=True)
    if result.get("gradcam_overlay") is not None:
        cols[2].image(result["gradcam_overlay"],
                      caption="Grad-CAM (red = most influential)",
                      use_container_width=True)
    elif "gradcam_error" in result:
        cols[2].caption(f"Grad-CAM unavailable: {result['gradcam_error']}")

elif image_file is None:
    st.info("👆 Upload or capture a leaf image to begin.")
