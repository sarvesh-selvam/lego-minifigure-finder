import io

import requests
import streamlit as st
from PIL import Image

API_URL = "http://localhost:8000"

st.set_page_config(page_title="LEGO Minifigure Finder", page_icon="🧱", layout="centered")
st.title("LEGO Minifigure Finder")
st.caption("Upload an image to check whether it contains a LEGO minifigure.")

# ---------------------------------------------------------------------------
# Sidebar — API health check
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("API Status")
    try:
        resp = requests.get(f"{API_URL}/health", timeout=2)
        if resp.ok and resp.json().get("model_loaded"):
            st.success("API online, model loaded")
        else:
            st.warning("API online but model not loaded")
    except requests.exceptions.ConnectionError:
        st.error("API offline — start the FastAPI server first")
        st.code("uvicorn src.app.app:app --reload", language="bash")

# ---------------------------------------------------------------------------
# Main — image upload and prediction
# ---------------------------------------------------------------------------
uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption=uploaded.name, use_container_width=True)

    if st.button("Run prediction", type="primary"):
        with st.spinner("Classifying..."):
            try:
                img_bytes = io.BytesIO()
                img.save(img_bytes, format="JPEG")
                img_bytes.seek(0)

                resp = requests.post(
                    f"{API_URL}/predict",
                    files={"file": (uploaded.name, img_bytes, "image/jpeg")},
                    timeout=10,
                )
                resp.raise_for_status()
                result = resp.json()

            except requests.exceptions.ConnectionError:
                st.error("Cannot reach the API. Is the FastAPI server running?")
                st.stop()
            except requests.exceptions.HTTPError as e:
                st.error(f"API error: {e.response.text}")
                st.stop()

        # Results
        label = result["pred_label"]
        confidence = result["positive_prob"]
        is_positive = result["is_positive"]

        if is_positive:
            st.success(f"Minifigure detected!")
        else:
            st.info("No minifigure detected.")

        col1, col2 = st.columns(2)
        col1.metric("Prediction", label)
        col2.metric("Confidence", f"{confidence:.1%}")

        with st.expander("Full probability breakdown"):
            class_names = ["not_minifig", "minifig"]
            for name, prob in zip(class_names, result["probs"]):
                st.write(f"**{name}**")
                st.progress(prob, text=f"{prob:.1%}")
