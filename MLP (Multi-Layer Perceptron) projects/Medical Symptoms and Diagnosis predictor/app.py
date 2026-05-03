import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import joblib

# -----------------------------
# Model Definition (same as train.py)
# -----------------------------
class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )

    def forward(self, x):
        return self.model(x)

# -----------------------------
# Load Artifacts
# -----------------------------
preprocessor = joblib.load("model/preprocessor.pkl")

# Dummy input to get input size
dummy = pd.DataFrame([{
    "age": 0,
    "gender": 1,
    "fever": 0,
    "cough": 0,
    "fatigue": 0,
    "headache": 0,
    "muscle_pain": 0,
    "nausea": 0,
    "vomiting": 0,
    "diarrhea": 0,
    "skin_rash": 0,
    "loss_smell": 0,
    "loss_taste": 0,
    "systolic_bp": 120,
    "diastolic_bp": 80,
    "heart_rate": 70,
    "temperature_c": 36.5,
    "oxygen_saturation": 98,
    "wbc_count": 7,
    "hemoglobin": 13,
    "platelet_count": 250,
    "crp_level": 5,
    "glucose_level": 100
}])

input_size = preprocessor.transform(dummy).shape[1]

# Load model
model = MLP(input_size)
model.load_state_dict(torch.load("model/model.pth"))
model.eval()

# Class labels
labels = ["Dengue", "Influenza", "COVID-19", "Malaria", "Pneumonia"]

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Disease Prediction", layout="centered")

st.title("🧠 Disease Prediction System")
st.warning("⚠️ This model has low accuracy (~20%) due to weak dataset signal. Predictions are not reliable.")
st.write("Enter patient details")

# -----------------------------
# Inputs
# -----------------------------
age = st.slider("Age", 0, 100, 30)
gender = st.selectbox("Gender", ["Male", "Female"])

fever = st.slider("Fever", 0, 3, 0)
cough = st.slider("Cough", 0, 3, 0)
fatigue = st.slider("Fatigue", 0, 3, 0)
headache = st.slider("Headache", 0, 3, 0)
muscle_pain = st.slider("Muscle Pain", 0, 3, 0)
nausea = st.slider("Nausea", 0, 3, 0)
vomiting = st.slider("Vomiting", 0, 3, 0)
diarrhea = st.slider("Diarrhea", 0, 3, 0)

skin_rash = st.slider("Skin Rash", 0, 3, 0)
loss_smell = st.slider("Loss of Smell", 0, 3, 0)
loss_taste = st.slider("Loss of Taste", 0, 3, 0)

systolic_bp = st.number_input("Systolic BP", 80, 200, 120)
diastolic_bp = st.number_input("Diastolic BP", 40, 120, 80)
heart_rate = st.number_input("Heart Rate", 40, 150, 70)
temperature_c = st.number_input("Temperature (°C)", 30.0, 45.0, 36.5)
oxygen_saturation = st.number_input("Oxygen Saturation", 80.0, 100.0, 98.0)

wbc_count = st.number_input("WBC Count", 1.0, 20.0, 7.0)
hemoglobin = st.number_input("Hemoglobin", 5.0, 20.0, 13.0)
platelet_count = st.number_input("Platelet Count", 50.0, 500.0, 250.0)
crp_level = st.number_input("CRP Level", 0.0, 50.0, 5.0)
glucose_level = st.number_input("Glucose Level", 50.0, 300.0, 100.0)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):

    input_df = pd.DataFrame([{
        "age": age,
        "gender": 1 if gender == "Male" else 0,
        "fever": fever,
        "cough": cough,
        "fatigue": fatigue,
        "headache": headache,
        "muscle_pain": muscle_pain,
        "nausea": nausea,
        "vomiting": vomiting,
        "diarrhea": diarrhea,
        "skin_rash": skin_rash,
        "loss_smell": loss_smell,
        "loss_taste": loss_taste,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "heart_rate": heart_rate,
        "temperature_c": temperature_c,
        "oxygen_saturation": oxygen_saturation,
        "wbc_count": wbc_count,
        "hemoglobin": hemoglobin,
        "platelet_count": platelet_count,
        "crp_level": crp_level,
        "glucose_level": glucose_level
    }])

    processed = preprocessor.transform(input_df)
    x_tensor = torch.tensor(processed, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(x_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()

    st.success(f"Predicted Disease: {labels[pred_class]}")

    # Show probabilities
    st.subheader("Prediction Probabilities")
    for i, label in enumerate(labels):
        st.write(f"{label}: {probs[0][i]:.2f}")