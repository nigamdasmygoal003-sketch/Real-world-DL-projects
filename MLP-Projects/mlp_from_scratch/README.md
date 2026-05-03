# 🏠 House Price Prediction using MLP (From Scratch)

This project implements a **Multi-Layer Perceptron (MLP) from scratch using NumPy** to predict house prices based on various features.

It includes:
- Custom neural network (no PyTorch / TensorFlow)
- Full training pipeline
- Data preprocessing
- Streamlit web application for predictions

---

## 🚀 Features

- 🧠 MLP implemented from scratch (forward + backpropagation)
- 📊 Handles real-world dataset with categorical + numerical features
- 🔄 One-hot encoding for categorical variables
- 📉 Data normalization for stable training
- 💾 Model saving (joblib + numpy + pickle)
- 🌐 Streamlit UI for live predictions

---

## 🧠 Model Architecture

The model follows:

Input Layer → Hidden Layer (ReLU) → Output Layer

Mathematically:

y = W₂ · ReLU(W₁x + b₁) + b₂

- Activation: ReLU  
- Loss Function: Mean Squared Error (MSE)  
- Optimization: Gradient Descent  

---

## 📂 Project Structure
```
mlp_from_scratch/
│── model.py # MLP implementation
│── train.py # Training script
│── app.py # Streamlit UI
│── house_price.csv # Dataset
│── model.pkl # Saved model
│── X_max.npy # Feature normalization
│── y_max.npy # Target scaling
│── columns.pkl # Feature alignment
│── requirements.txt
│── README.md
```
---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone git clone https://github.com/nigamdasmygoal003-sketch/Real-world-DL-projects.git
cd Real-world-DL-projects
pip install -r requirements.txt
```
▶️ Train the Model
python train.py

This will generate:

model.pkl
X_max.npy
y_max.npy
columns.pkl
🌐 Run the Web App
streamlit run app.py
📊 Input Features
Area
Bedrooms
Bathrooms
Floors
Age
Distance to city
Garage, Parking, Garden, Security
Nearby facilities (school, hospital, mall, transport)
Crime rate
Population density
Location (categorical)
Income level (categorical)
⚠️ Important Notes
Input data is normalized using training statistics
Categorical features are one-hot encoded
Feature alignment is handled using columns.pkl
Without proper preprocessing, predictions will be incorrect
📈 Future Improvements
Add validation split and loss visualization
Implement regularization (Dropout / L2)
Convert to PyTorch for scalability
Deploy on Streamlit Cloud / HuggingFace Spaces
🧠 Learning Outcomes

This project demonstrates:

Neural network fundamentals
Backpropagation from scratch
Real-world data preprocessing
Model deployment pipeline
Debugging and ML engineering skills
