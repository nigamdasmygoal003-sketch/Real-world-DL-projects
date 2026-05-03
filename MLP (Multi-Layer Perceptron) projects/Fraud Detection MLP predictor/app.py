import streamlit as st
import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd

# -------------------------
# Load Preprocessor
# -------------------------
preprocessor = joblib.load("model/preprocessor.pkl")

# -------------------------
# Model Definition (same as train.py)
# -------------------------
class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# -------------------------
# Load Model
# -------------------------
input_size = preprocessor.transform(pd.DataFrame([{
    "Transaction_Amount": 0,
    "Transaction_Type": "ATM Withdrawal",
    "Time_of_Transaction": 0,
    "Device_Used": "Mobile",
    "Location": "New York",
    "Previous_Fraudulent_Transactions": 0,
    "Account_Age": 1,
    "Number_of_Transactions_Last_24H": 1,
    "Payment_Method": "Credit Card"
}])).shape[1]

model = MLP(input_size)
model.load_state_dict(torch.load("model/model.pth"))
model.eval()

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Fraud Detection", layout="centered")

st.title("💳 Fraud Detection Predictor (MLP)")
st.write("Enter transaction details to check if it's fraudulent.")

# -------------------------
# Input Fields
# -------------------------
transaction_amount = st.number_input("Transaction Amount", min_value=0.0)
transaction_type = st.selectbox("Transaction Type", ["ATM Withdrawal", "POS Payment", "Bill Payment"])
time_of_transaction = st.number_input("Time of Transaction (0–24)", min_value=0.0, max_value=24.0)
device_used = st.selectbox("Device Used", ["Mobile", "Desktop", "Tablet"])
location = st.selectbox("Location", ["New York", "San Francisco", "Chicago"])
previous_fraud = st.number_input("Previous Fraudulent Transactions", min_value=0)
account_age = st.number_input("Account Age (days)", min_value=0)
num_tx_24h = st.number_input("Transactions in Last 24H", min_value=0)
payment_method = st.selectbox("Payment Method", ["Credit Card", "Debit Card", "UPI"])

# -------------------------
# Prediction
# -------------------------
if st.button("Predict"):

    input_dict = {
        "Transaction_Amount": transaction_amount,
        "Transaction_Type": transaction_type,
        "Time_of_Transaction": time_of_transaction,
        "Device_Used": device_used,
        "Location": location,
        "Previous_Fraudulent_Transactions": previous_fraud,
        "Account_Age": account_age,
        "Number_of_Transactions_Last_24H": num_tx_24h,
        "Payment_Method": payment_method
    }

    input_df = pd.DataFrame([input_dict])

    # Preprocess
    input_processed = preprocessor.transform(input_df)

    input_tensor = torch.tensor(input_processed, dtype=torch.float32)

    with torch.no_grad():
        output = model(input_tensor)
        prob = output.item()
        pred = int(prob > 0.5)

    # -------------------------
    # Output
    # -------------------------
    st.subheader("Result:")

    if pred == 1:
        st.error(f"⚠️ Fraud Detected! (Confidence: {prob:.2f})")
    else:
        st.success(f"✅ Legitimate Transaction (Confidence: {1 - prob:.2f})")