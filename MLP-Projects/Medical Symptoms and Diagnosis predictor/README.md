# 🧠 Disease Prediction using MLP (PyTorch)

This project builds a **multi-class disease prediction system** using:

- 🧠 PyTorch (Multi-Layer Perceptron)
- 🔧 scikit-learn (Preprocessing Pipeline)
- 🌐 Streamlit (Web App)

The goal was to classify patients into one of 5 diseases:

- Dengue  
- Influenza  
- COVID-19  
- Malaria  
- Pneumonia  

---

## 🚀 Features

- ✅ Multi-class classification (5 classes)
- ✅ Neural Network (PyTorch MLP)
- ✅ Preprocessing pipeline (scaling + encoding)
- ✅ RandomForest baseline comparison
- ✅ Streamlit web app for predictions
- ⚠️ Real-world insight: dataset limitations

---

## 🧠 Model Architecture
Input → Linear(128) → ReLU
→ Linear(64) → ReLU
→ Linear(5)


- Loss Function: CrossEntropyLoss  
- Optimizer: Adam  

---

## 📂 Project Structure

```
Medical Symptoms and Diagnosis predictor/
│
├── data/
│ └── medical_symptoms_dataset.csv
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
---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone git clone https://github.com/nigamdasmygoal003-sketch/Real-world-DL-projects.git
cd Real-world-DL-projects
pip install -r requirements.txt
```

▶️ Train the Model
python src/train.py
🌐 Run the App
streamlit run app.py
📊 Dataset Overview
3000 samples
23 input features (symptoms + medical metrics)
5 target classes

Features include:

Symptoms (fever, cough, fatigue, etc.)
Vital signs (BP, heart rate, temperature)
Lab results (WBC, CRP, glucose)
📈 Results
Model	Accuracy
MLP (PyTorch)	~20%
RandomForest	~21%

👉 Both models perform near random baseline (20%)

⚠️ Key Insight (IMPORTANT)

This project demonstrates a critical ML lesson:

❝ Model performance depends more on data quality than model complexity ❞

Observations:
Weak separation between classes
Overlapping symptom patterns
Low discriminative signal
Possible synthetic/noisy dataset
Conclusion:

The dataset does not contain enough information for reliable classification.
