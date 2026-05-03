import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pickle

# -----------------------------
# Load model and preprocessing data
# -----------------------------
model = joblib.load("model.pkl")
X_max = np.load("X_max.npy")
y_max = np.load("y_max.npy")

with open("columns.pkl", "rb") as f:
    train_columns = pickle.load(f)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="House Price Predictor", layout="centered")

st.title("🏠 House Price Prediction (MLP from Scratch)")
st.write("Fill the details below to predict house price")

# -----------------------------
# Inputs
# -----------------------------
area = st.number_input("Area (sq ft)", 500, 10000, 2000)
bedrooms = st.slider("Bedrooms", 1, 10, 3)
bathrooms = st.slider("Bathrooms", 1, 5, 2)
floors = st.slider("Floors", 1, 5, 2)
age = st.slider("Age (years)", 0, 50, 10)
distance = st.slider("Distance to city center", 1, 50, 10)

garage = st.selectbox("Garage", [0, 1])
parking = st.selectbox("Parking", [0, 1])
garden = st.selectbox("Garden", [0, 1])
security = st.selectbox("Security", [0, 1])

school_nearby = st.selectbox("School Nearby", [0, 1])
hospital_nearby = st.selectbox("Hospital Nearby", [0, 1])
shopping_mall_nearby = st.selectbox("Shopping Mall Nearby", [0, 1])
public_transport = st.selectbox("Public Transport", [0, 1])

crime_rate = st.number_input("Crime Rate", 0.0, 10.0, 5.0)
population_density = st.number_input("Population Density", 500, 20000, 5000)

location = st.selectbox("Location", ["low", "medium", "premium"])
income_level = st.selectbox("Income Level", ["low", "mid"])

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Price"):

    # Create input dataframe
    input_df = pd.DataFrame([{
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "floors": floors,
        "age": age,
        "distance": distance,
        "garage": garage,
        "parking": parking,
        "garden": garden,
        "security": security,
        "school_nearby": school_nearby,
        "hospital_nearby": hospital_nearby,
        "shopping_mall_nearby": shopping_mall_nearby,
        "public_transport": public_transport,
        "crime_rate": crime_rate,
        "population_density": population_density,
        "location": location,
        "income_level": income_level
    }])

    # One-hot encoding
    input_df = pd.get_dummies(input_df)

    # Align with training columns
    input_df = input_df.reindex(columns=train_columns, fill_value=0)

    # Convert to numpy
    input_array = input_df.values.astype(float)

    # Normalize
    input_array = input_array / (X_max + 1e-8)

    # Predict
    pred = model.forward(input_array)
    price = pred[0][0] * y_max

    st.success(f"💰 Predicted Price: ₹ {price:,.2f}")