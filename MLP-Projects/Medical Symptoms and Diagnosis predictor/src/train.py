import pandas as pd
import joblib
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from torch.utils.data import TensorDataset, DataLoader

# -----------------------------
# Load Data
# -----------------------------
data = pd.read_csv("data/medical_symptoms_dataset.csv")

# -----------------------------
# Encoding
# -----------------------------
data["gender"] = data["gender"].map({"Male": 1, "Female": 0})

data["diagnosis"] = data["diagnosis"].map({
    "Dengue": 0,
    "Influenza": 1,
    "COVID-19": 2,
    "Malaria": 3,
    "Pneumonia": 4
})

# -----------------------------
# Split X, y
# -----------------------------
X = data.drop(columns=["diagnosis"])
y = data["diagnosis"]

# -----------------------------
# Preprocessing
# -----------------------------
num_features = X.select_dtypes(include=["int64", "float64"]).columns
cat_features = X.select_dtypes(include=["object"]).columns

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features)
])

# -----------------------------
# Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# -----------------------------
# Baseline Model (RandomForest)
# -----------------------------
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

rf_acc = rf.score(X_test, y_test)
print(f"\n🌲 RandomForest Accuracy: {rf_acc:.4f}")

# -----------------------------
# Convert to Tensors
# -----------------------------
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.long)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.long)

# -----------------------------
# DataLoader
# -----------------------------
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# -----------------------------
# Model
# -----------------------------
class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5),
            nn.Softmax()
        )

    def forward(self, x):
        return self.model(x)

model = MLP(input_size=X_train.shape[1])

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# Training
# -----------------------------
epochs = 100

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        outputs = model(xb)
        loss = criterion(outputs, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

# -----------------------------
# Evaluation
# -----------------------------
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    preds = torch.argmax(outputs, dim=1)

    acc = (preds == y_test).float().mean()

print(f"\n🧠 MLP Test Accuracy: {acc.item():.4f}")

# -----------------------------
# Prediction Distribution
# -----------------------------
print("\n🔍 Prediction Distribution:")
print(torch.bincount(preds))

# -----------------------------
# Save Artifacts
# -----------------------------
torch.save(model.state_dict(), "model/model.pth")
joblib.dump(preprocessor, "model/preprocessor.pkl")

print("\n✅ Model saved: model/model.pth")
print("✅ Preprocessor saved: model/preprocessor.pkl")