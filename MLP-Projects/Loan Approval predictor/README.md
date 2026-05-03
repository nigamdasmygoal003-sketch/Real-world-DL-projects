# 🏦 Loan Approval Prediction using MLP (PyTorch)

This project builds a **Loan Approval Prediction system** using:

- 🧠 PyTorch (Multi-Layer Perceptron)
- 🔧 scikit-learn (Preprocessing Pipeline)
- 🌐 Streamlit (Web App)

It is a **complete end-to-end Machine Learning project** including:
data preprocessing → model training → evaluation → deployment.

---

## 🚀 Features

- ✅ MLP Neural Network (PyTorch)
- ✅ ColumnTransformer Pipeline (sklearn)
- ✅ Handles categorical + numerical data
- ✅ Train/Test split (real evaluation)
- ✅ Model saving using `state_dict`
- ✅ Streamlit UI for live prediction

---

## 🧠 Model Architecture

Input → Hidden Layer → Hidden Layer → Output

- Linear (input → 64) + ReLU  
- Linear (64 → 32) + ReLU  
- Linear (32 → 1) + Sigmoid  

Loss Function: Binary CrossEntropy  
Optimizer: Adam  

---

## 📂 Project Structure
```
Loan Approval predictor/
│
├── data/
│ └── loan_approval_dataset.csv
├── Image/
| └──appImage.png
│
├── model/
│ ├── model.pth
│ └── preprocessor.pkl
│
├── notebook/
| └── experiment.ipynb
├──src/
| └──train.py
├── app.py
├── requirements.txt
├── README.md
```
---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone git clone https://github.com/nigamdasmygoal003-sketch/Real-world-DL-projects.git
cd Real-world-DL-projects
pip install -r requirements.txt
```
