import pandas as pd
import joblib
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from torch.utils.data import TensorDataset, DataLoader

# -----------------------------
# Fix column names (IMPORTANT)
# -----------------------------
data = pd.read_csv("data/loan_approval_dataset.csv")


# Target encoding
data[" loan_status"] = data[" loan_status"].map({" Approved": 1, " Rejected": 0})

# Split X, y
X = data.drop(columns=["loan_id", " loan_status"])
y = data[" loan_status"]

# -----------------------------
# Preprocessing
# -----------------------------
num_features = X.select_dtypes(include=["int64"]).columns
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

# Train/Test split BEFORE fit (important)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# -----------------------------
# Convert to tensors
# -----------------------------
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# -----------------------------
# DataLoader (mini-batch)
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
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

model = MLP(input_size=X_train.shape[1])

criterion = nn.BCELoss()
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

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

# -----------------------------
# Evaluation
# -----------------------------
model.eval()
with torch.no_grad():
    preds = model(X_test)
    predicted = (preds > 0.5).float()
    accuracy = (predicted == y_test).float().mean()

print("Test Accuracy:", accuracy.item())

# -----------------------------
# Save artifacts (IMPORTANT)
# -----------------------------
# Save model weights
torch.save(model.state_dict(), "model/model.pth")

# Save preprocessor
joblib.dump(preprocessor, "model/preprocessor.pkl")

print("✅ Model saved: model/model.pth")
print("✅ Preprocessor saved: model/preprocessor.pkl")