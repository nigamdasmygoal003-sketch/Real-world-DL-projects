import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import joblib

# -----------------------------
# Define Model (must match train.py)
# -----------------------------
class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# -----------------------------
# Load preprocessor
# -----------------------------
preprocessor = joblib.load("model/preprocessor.pkl")

# Create dummy input to determine input size
dummy = pd.DataFrame([{
    " no_of_dependents": 0,
    " education": " Graduate",
    " self_employed": " No",
    " income_annum": 0,
    " loan_amount": 0,
    " loan_term": 0,
    " cibil_score": 0,
    " residential_assets_value": 0,
    " commercial_assets_value": 0,
    " luxury_assets_value": 0,
    " bank_asset_value": 0
}])

input_size = preprocessor.transform(dummy).shape[1]

# -----------------------------
# Load model weights
# -----------------------------
model = MLP(input_size)
model.load_state_dict(torch.load("model/model.pth"))
model.eval()

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Loan Approval Predictor", layout="centered")

st.title("🏦 Loan Approval Prediction")
st.write("Enter applicant details")

# Inputs
no_of_dependents = st.slider("Dependents", 0, 10, 2)
education = st.selectbox("Education", [" Graduate", " Not Graduate"])
self_employed = st.selectbox("Self Employed", [" Yes", " No"])

income_annum = st.number_input("Annual Income", 100000, 10000000, 500000)
loan_amount = st.number_input("Loan Amount", 100000, 50000000, 1000000)
loan_term = st.slider("Loan Term (years)", 1, 30, 10)
cibil_score = st.slider("CIBIL Score", 300, 900, 700)

residential_assets_value = st.number_input("Residential Assets", 0, 50000000, 1000000)
commercial_assets_value = st.number_input("Commercial Assets", 0, 50000000, 1000000)
luxury_assets_value = st.number_input("Luxury Assets", 0, 50000000, 1000000)
bank_asset_value = st.number_input("Bank Assets", 0, 50000000, 1000000)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):

    input_df = pd.DataFrame([{
        " no_of_dependents": no_of_dependents,
        " education": education,
        " self_employed": self_employed,
        " income_annum": income_annum,
        " loan_amount": loan_amount,
        " loan_term": loan_term,
        " cibil_score": cibil_score,
        " residential_assets_value": residential_assets_value,
        " commercial_assets_value": commercial_assets_value,
        " luxury_assets_value": luxury_assets_value,
        " bank_asset_value": bank_asset_value
    }])

    # Preprocess
    processed = preprocessor.transform(input_df)

    # Convert to tensor
    x_tensor = torch.tensor(processed, dtype=torch.float32)

    # Predict
    with torch.no_grad():
        prob = model(x_tensor).item()

    # Output
    if prob > 0.5:
        st.success(f"✅ Loan Approved (Confidence: {prob:.2f})")
    else:
        st.error(f"❌ Loan Rejected (Confidence: {prob:.2f})")

    # Optional visualization
    st.progress(float(prob))