# 💳 Fraud Detection MLP Predictor (Deep Learning Project)

## 📌 Overview

This project builds a **Fraud Detection System** using a **Multi-Layer Perceptron (MLP)** implemented in **PyTorch**.

The model predicts whether a financial transaction is **fraudulent or legitimate** based on transaction details such as amount, device, location, and user behavior.

---

## 🚀 Features

- End-to-end Deep Learning pipeline
- Data preprocessing using `ColumnTransformer`
- Handling missing values (numerical + categorical)
- One-hot encoding for categorical features
- MLP model built using PyTorch
- Batch training using DataLoader
- Streamlit web application for real-time predictions
- Model + Preprocessor saving for deployment

---

## 🧠 Model Architecture


Input Layer → 128 → ReLU → 64 → ReLU → Output → Sigmoid


- Loss Function: Binary Cross Entropy (`BCELoss`)
- Optimizer: Adam
- Epochs: 1200+
- Learning Rate: 0.0005

---

## 📊 Dataset

The dataset contains **51,000 transactions** with features:

- Transaction_Amount
- Transaction_Type
- Time_of_Transaction
- Device_Used
- Location
- Previous_Fraudulent_Transactions
- Account_Age
- Number_of_Transactions_Last_24H
- Payment_Method

Target:
- `Fraudulent` (0 = Legitimate, 1 = Fraud)

---

## ⚙️ Project Structure

```
Fraud Detection MLP predictor/
│
├── data/
│ └── FraudDetectionDataset.csv
│
├── Image/
│ └── appImage.png
│
├── model/
│ ├── model.pth
│ └── preprocessor.pkl
│
├── notebook/
│ └── experiment.ipynb
│
├── src/
│ └── train.py
│
├── app.py
├── requirements.txt
├── README.md
```
### 1. Clone the repository

```bash
git clone git clone https://github.com/nigamdasmygoal003-sketch/Real-world-DL-projects.git
cd Real-world-DL-projects
```
---

## 🧪 Training

Run the training script:

```bash
python src/train.py

This will:

Train the MLP model
Save the trained model to model/model.pth
Save the preprocessing pipeline to model/preprocessor.pkl
🌐 Run the App
streamlit run app.py
🖥️ App Features
User-friendly input interface
Real-time fraud prediction
Confidence score display
Handles missing and categorical inputs
📈 Model Performance
Accuracy: ~90.85%

⚠️ Note: Accuracy alone is not enough for fraud detection.
Future improvements can include:

Precision / Recall / F1-score
ROC-AUC
Handling class imbalance
🔧 Future Improvements
Add class imbalance handling (SMOTE / Weighted Loss)
Deploy using FastAPI + Docker
Add batch prediction (CSV upload)
Improve UI with charts and analytics
Use advanced models (XGBoost, Transformers for tabular)
🧠 Skills Demonstrated
Deep Learning (PyTorch)
Data Preprocessing Pipelines
Feature Engineering
Model Training & Evaluation
Deployment using Streamlit
End-to-End ML System Design
📬 Author

Nigam Das
B.Tech AI/ML Student
Building real-world Machine Learning & Deep Learning projects 🚀

⭐ If you like this project

Give it a ⭐ on GitHub and follow for more AI projects!
