import numpy as np
import pandas as pd
import joblib
import pickle

from model import MLP

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("house_price.csv")

# One-hot encoding
data = pd.get_dummies(data, columns=["location", "income_level"], drop_first=True)

# Ensure all numeric
data = data.astype(float)

# -----------------------------
# Split features and target
# -----------------------------
X = data.drop(columns=["price"]).values
y = data["price"].values.reshape(-1, 1)

# -----------------------------
# Normalization
# -----------------------------
X_max = np.max(X, axis=0)
X = X / (X_max + 1e-8)

y_max = np.max(y)
y = y / y_max

# -----------------------------
# Initialize Model
# -----------------------------
model = MLP(input_size=X.shape[1], hidden_size=16, output_size=1)

# -----------------------------
# Training Loop
# -----------------------------
epochs = 2000

for epoch in range(epochs):
    y_pred = model.forward(X)
    loss = model.compute_loss(y_pred, y)

    model.backward(X, y, y_pred, lr=0.01)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# -----------------------------
# Save EVERYTHING (IMPORTANT)
# -----------------------------

# Save model
joblib.dump(model, "model.pkl")

# Save normalization values
np.save("X_max.npy", X_max)
np.save("y_max.npy", y_max)

# Save column names (for encoding alignment)
with open("columns.pkl", "wb") as f:
    pickle.dump(data.drop(columns=["price"]).columns.tolist(), f)

print("\n✅ Training complete. Files saved:")
print("✔ model.pkl")
print("✔ X_max.npy")
print("✔ y_max.npy")
print("✔ columns.pkl")