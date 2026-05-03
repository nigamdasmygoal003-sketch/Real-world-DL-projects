import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image

from src.model import CNN

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="Digit Recognizer", layout="centered")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    model = CNN().to(DEVICE)
    model.load_state_dict(torch.load("models/cnn_model.pth", map_location=DEVICE))
    model.eval()
    return model


model = load_model()


# -------------------------------
# PREPROCESS FUNCTION (FIXED)
# -------------------------------
def preprocess_image(image):
    import numpy as np
    import cv2
    import torch

    img = np.array(image)

    # -------------------------------
    # 1. Convert to grayscale safely
    # -------------------------------
    if len(img.shape) == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # -------------------------------
    # 2. Normalize contrast (🔥 KEY FIX)
    # -------------------------------
    gray = cv2.equalizeHist(gray)

    # -------------------------------
    # 3. Light blur (NOT heavy)
    # -------------------------------
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # -------------------------------
    # 4. Otsu threshold (clean, not aggressive)
    # -------------------------------
    _, bw = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # -------------------------------
    # 5. Find largest contour
    # -------------------------------
    contours, _ = cv2.findContours(
        bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        digit = bw[y:y+h, x:x+w]
    else:
        digit = bw

    # -------------------------------
    # 6. Resize with padding (KEEP SHAPE)
    # -------------------------------
    h, w = digit.shape
    scale = 20.0 / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    digit = cv2.resize(digit, (new_w, new_h))

    canvas = np.zeros((28, 28), dtype=np.uint8)

    y_off = (28 - new_h) // 2
    x_off = (28 - new_w) // 2

    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = digit

    # -------------------------------
    # 7. Normalize (MATCH TRAINING)
    # -------------------------------
    canvas = canvas / 255.0
    canvas = (canvas - 0.5) / 0.5

    tensor = torch.tensor(canvas).float().unsqueeze(0).unsqueeze(0)

    return tensor, canvas

# -------------------------------
# UI
# -------------------------------
st.title("🧠 Handwritten Digit Recognizer")
st.write("Upload a handwritten digit image (0–9)")

uploaded_file = st.file_uploader("📤 Upload Image", type=["png", "jpg", "jpeg"])


# -------------------------------
# PREDICTION
# -------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.subheader("📷 Uploaded Image")
    st.image(image, width=200)

    # Preprocess
    input_tensor, processed_img = preprocess_image(image)
    input_tensor = input_tensor.to(DEVICE)

    st.subheader("🧪 Processed Image (Model Input)")
    st.image(processed_img, width=150, clamp=True)

    # Prediction
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]

    pred = int(np.argmax(probs))
    confidence = float(probs[pred])

    # -------------------------------
    # RESULTS
    # -------------------------------
    st.subheader("🎯 Prediction")
    st.markdown(f"### Digit: **{pred}**")
    st.markdown(f"### Confidence: **{confidence * 100:.2f}%**")

    # -------------------------------
    # TOP-3
    # -------------------------------
    st.subheader("🔝 Top Predictions")
    top3_idx = np.argsort(probs)[-3:][::-1]

    for i in top3_idx:
        st.write(f"{i} → {probs[i]*100:.2f}%")

    # -------------------------------
    # PROBABILITY CHART
    # -------------------------------
    st.subheader("📊 Class Probabilities")
    prob_dict = {str(i): float(probs[i]) for i in range(10)}
    st.bar_chart(prob_dict)

    # -------------------------------
    # WARNING
    # -------------------------------
    if confidence < 0.6:
        st.warning("⚠️ Low confidence. Try clearer handwriting.")