"""
Streamlit app: Digit drawer + Gradient Boosting Classifier prediction
File: streamlit_gbc_app.py
Model file expected: gbc.pkl (place in same directory)

Requirements (put in requirements.txt):
streamlit
streamlit-drawable-canvas
scikit-learn
numpy
pillow

How to run:
1. Place your trained model file `gbc.pkl` in the same folder as this script.
2. Install requirements: `pip install -r requirements.txt` (or individually)
3. Run: `streamlit run streamlit_gbc_app.py`

Description:
- Lets the user draw a digit on a canvas.
- Converts the drawing to the same 8x8 feature representation as sklearn.datasets.load_digits
  (values scaled to 0-16 per pixel), then uses gbc.pkl to predict the digit.
- Shows top-3 predicted classes with probabilities.

Notes:
- This code assumes gbc.pkl was trained on sklearn's digits (8x8, 0-16 scale) like in your snippet.
- If your model expects a different preprocessing pipeline, adapt the `preprocess_image()` function.
"""

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image, ImageOps
import pickle
import io

# --- Config ---
MODEL_PATH = "gbc.pkl"
CANVAS_SIZE = 280  # pixels (canvas will be CANVAS_SIZE x CANVAS_SIZE)
RESIZE_TO = (8, 8)  # target for sklearn digits

st.set_page_config(page_title="Digit Recognizer (GBC)", layout="centered")

st.title("✍️ Draw a digit — Gradient Boosting Classifier predicts it")
st.markdown("Draw a single digit (0-9) inside the box, then click Predict.\n\nThe app converts your drawing to the 8x8 representation used by sklearn's `load_digits` before predicting.")

# Load model
@st.cache_resource
def load_model(path):
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Could not load model from {path}: {e}")
        return None

model = load_model(MODEL_PATH)

# Canvas
st.subheader("Draw here")
canvas_result = st_canvas(
    fill_color="rgba(0,0,0,0)",  # Transparent background
    stroke_width=20,
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=CANVAS_SIZE,
    width=CANVAS_SIZE,
    drawing_mode="freedraw",
    key="canvas",
)

# Utility: convert canvas image data to model input
def preprocess_image(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to the 8x8 features used by sklearn digits (0-16 scale).

    Steps:
    - Convert to grayscale
    - Invert colors so strokes are bright (like sklearn digits)
    - Resize to 8x8
    - Scale pixel range to 0..16
    - Flatten to shape (64,)
    """
    # Convert to grayscale
    image = image.convert("L")

    # Invert so drawn strokes become bright (255) and background dark (0)
    image = ImageOps.invert(image)

    # Resize to 8x8 with antialiasing
    image = image.resize(RESIZE_TO, resample=Image.LANCZOS)

    arr = np.asarray(image).astype(np.float32)

    # Normalize to 0-16 (digits dataset uses integer values 0..16)
    # Current arr is 0..255
    arr = (arr / 255.0) * 16.0

    # Flatten
    flat = arr.flatten()
    return flat

# Predict button
st.subheader("Predict")
col1, col2 = st.columns([2, 1])
with col1:
    if st.button("Predict"):
        if canvas_result.image_data is None:
            st.warning("Please draw something on the canvas first.")
        elif model is None:
            st.error("Model not loaded. Make sure gbc.pkl is present in the app directory.")
        else:
            # Convert the canvas image (RGBA) to PIL image
            img_data = canvas_result.image_data  # numpy array HxWx4
            # Convert to PIL; ensure white background
            pil_img = Image.fromarray((img_data * 255).astype(np.uint8)) if img_data.max() <= 1.0 else Image.fromarray(img_data.astype(np.uint8))

            # If the canvas includes alpha, composite over white background
            if pil_img.mode == "RGBA":
                bg = Image.new("RGBA", pil_img.size, (255, 255, 255, 255))
                pil_img = Image.alpha_composite(bg, pil_img).convert("RGB")

            # Convert to grayscale and preprocess
            sample = preprocess_image(pil_img)

            # The model expects shape (n_samples, n_features)
            sample = sample.reshape(1, -1)

            try:
                pred = model.predict(sample)[0]
                probs = None
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(sample)[0]

                st.success(f"Predicted digit: **{pred}**")

                if probs is not None:
                    # show top 3
                    top3_idx = np.argsort(probs)[::-1][:3]
                    st.write("Top predictions:")
                    for i in top3_idx:
                        st.write(f"{i}: {probs[i]:.3f}")

                # Show the preprocessed 8x8 image for debugging
                st.subheader("Preprocessed 8x8 image")
                arr8 = sample.reshape(8, 8)
                # Scale 0..16 back to 0..255 for display
                disp = ((arr8 / 16.0) * 255).astype(np.uint8)
                st.image(disp, width=160, caption="8x8 scaled view")

            except Exception as e:
                st.error(f"Prediction failed: {e}")
with col2:
    if st.button("Clear"):
        # A simple way to clear is to rerun; the canvas will be empty initially
        st.experimental_rerun()

# Extra: allow uploading an image
st.subheader("Or upload an image file")
uploaded = st.file_uploader("Upload a handwritten digit image (PNG/JPG)", type=["png", "jpg", "jpeg"])
if uploaded is not None:
    try:
        image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
        st.image(image, caption="Uploaded image", width=200)
        if st.button("Predict from upload"):
            if model is None:
                st.error("Model not loaded. Make sure gbc.pkl is present in the app directory.")
            else:
                sample = preprocess_image(image)
                sample = sample.reshape(1, -1)
                pred = model.predict(sample)[0]
                st.success(f"Predicted digit: **{pred}**")
    except Exception as e:
        st.error(f"Could not read uploaded image: {e}")

st.markdown("---")
st.write("Model file expected: `gbc.pkl`. If you need, run your training script to produce that pickle (you provided the training snippet).")

