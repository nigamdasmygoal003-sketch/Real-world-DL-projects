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

data = pd.read_csv("data/FraudDetectionDataset.csv")

x = data.drop(columns=["Transaction_ID","User_ID","Fraudulent"])
y = data["Fraudulent"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


num_features = x.select_dtypes(include=["int64","float64"]).columns
cat_features = x.select_dtypes(include=["object"]).columns

num_pipeline = Pipeline([
    ("imputer",SimpleImputer(strategy="mean")),
    ("scaler",StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer",SimpleImputer(strategy="most_frequent")),
    ("encoder",OneHotEncoder(handle_unknown="ignore",sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num",num_pipeline,num_features),
    ("cat",cat_pipeline,cat_features)
])

x_train_preprocessed = preprocessor.fit_transform(x_train)
x_test_preprocessed = preprocessor.transform(x_test)

x_train = torch.tensor(x_train_preprocessed, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1,1)

x_test = torch.tensor(x_test_preprocessed, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1,1)

from torch.utils.data import TensorDataset,DataLoader

train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

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
    
model = MLP(input_size=x_train.shape[1])

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.0005)

epochs = 1201

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for xb,yb in train_loader:
        output = model(xb)
        loss = criterion(output,yb)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f} , avg_loss: {avg_loss:.4f}")    
        
        
with torch.no_grad():
    preds = model(x_test)
    predicted = (preds > 0.5).float()
    
print(predicted[:10])    

accuracy = (predicted == y_test).float().mean()
print("Accuracy:", accuracy.item())


torch.save(model.state_dict(), "model/model.pth")

# Save preprocessor
joblib.dump(preprocessor, "model/preprocessor.pkl")

print("✅ Model saved: model/model.pth")
print("✅ Preprocessor saved: model/preprocessor.pkl")        