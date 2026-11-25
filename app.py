import streamlit as st
import numpy as np
import joblib
from PIL import Image

st.set_page_config(page_title="Handwritten Digit Recognition")
st.title("Handwritten Digit Recognition")

# choose between load_digits (8x8) and MNIST (28x28)
dataset_choice = st.selectbox("Select Model:", ["MNIST (28x28)", "load_digits (8x8)"])

samples = np.load("sample_digits.npz")
if dataset_choice == "MNIST (28x28)":
    model = joblib.load("mnist_model.pkl")
    sample_images = samples["mnist_images"]
    sample_labels = samples["mnist_labels"]
    img_size = 28
else:
    model = joblib.load("digits_model.pkl")
    sample_images = samples["digits_images"]
    sample_labels = samples["digits_labels"]
    img_size = 8

mode = st.radio("Choose input method:", ["Select a sample digit", "Upload an image"])

if mode == "Select a sample digit":
    idx = st.slider("Pick sample index", 0, len(sample_images) - 1, 0)
    img = sample_images[idx]
    true_label = sample_labels[idx]

    st.image(img, caption=f"Actual digit: {true_label}", width=150, clamp=True)

    x_input = img.flatten().reshape(1, -1)
    probs = model.predict_proba(x_input)[0]
    pred = int(np.argmax(probs))
    conf = probs[pred] * 100

    st.subheader(f"Predicted: {pred} ({conf:.2f}% confidence)")
    st.bar_chart(probs)

else:
    uploaded = st.file_uploader("Upload a digit image (png/jpg)", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        img = Image.open(uploaded).convert("L")
        st.image(img, caption="Uploaded image", width=150)

        # invert if background is white/light
        arr = np.array(img, dtype=np.float32)
        if np.mean(arr) > 127:
            arr = 255.0 - arr

        # remove gray paper background and boost contrast
        arr = arr - np.percentile(arr, 60)
        arr = np.clip(arr, 0, None)
        if arr.max() > 0:
            arr = (arr / arr.max()) * 255.0

        # crop around the digit
        coords = np.argwhere(arr > 30)
        if len(coords) > 0:
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0) + 1
            arr = arr[y0:y1, x0:x1]

        if img_size == 28:
            digit_20 = Image.fromarray(arr.astype(np.uint8)).resize((20, 20), Image.Resampling.LANCZOS)
            canvas = Image.new("L", (28, 28), 0)
            canvas.paste(digit_20, (4, 4))
            final_img = canvas
        else:
            final_img = Image.fromarray(arr.astype(np.uint8)).resize((8, 8), Image.Resampling.LANCZOS)

        x_input = (np.array(final_img, dtype=np.float32).flatten() / 255.0).reshape(1, -1)
        probs = model.predict_proba(x_input)[0]
        pred = int(np.argmax(probs))
        conf = probs[pred] * 100

        st.subheader(f"Predicted: {pred} ({conf:.2f}% confidence)")
        st.bar_chart(probs)
