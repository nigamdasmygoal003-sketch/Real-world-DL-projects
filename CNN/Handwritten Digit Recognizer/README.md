# 🧠 Handwritten Digit Recognizer (CNN + Streamlit)

A deep learning project that predicts handwritten digits (0–9) using a Convolutional Neural Network (CNN) trained on the EMNIST dataset, enhanced with synthetic data for better real-world performance.

---

## 🚀 Features

- 🔢 Digit classification (0–9)
- 🧠 CNN model built with PyTorch
- 🎨 Streamlit web app for image upload
- 🧪 Advanced preprocessing pipeline
- 📊 Confidence score + probability distribution
- 🔝 Top-3 predictions
- ⚡ Works with real handwritten images (not just MNIST)

---

## 📁 Project Structure


Handwritten Digit Recognizer/
│
├── data/
│ ├── custom/ # Synthetic / custom images
│
├── models/
│ └── cnn_model.pth # Trained model
│
├── src/
│ ├── init.py
│ ├── data_loader.py
│ ├── custom_dataset.py
│ ├── model.py
│ ├── train.py
│ ├── evaluate.py
│
├── app.py # Streamlit app
├── generate_synthetic.py # Synthetic data generator
├── requirements.txt
├── README.md


---

## 🧠 Model Details

- Architecture: CNN (Conv → ReLU → BatchNorm → FC)
- Input: 28x28 grayscale image
- Output: 10 classes (digits 0–9)
- Loss: CrossEntropyLoss
- Optimizer: Adam

---

## 📊 Dataset

- **EMNIST (digits split)** — base dataset
- **Synthetic dataset (OpenCV generated)** — improves generalization

---

## 🧪 Preprocessing Pipeline

- Grayscale conversion
- Contrast normalization
- Noise reduction (Gaussian blur)
- Adaptive thresholding
- Contour detection & cropping
- Centering in 28×28 canvas
- Normalization to match training distribution

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone git clone https://github.com/nigamdasmygoal003-sketch/Real-world-DL-projects.git
cd Real-world-DL-projects
```

pip install -r requirements.txt
🏋️ Train Model
python -m src.train
📈 Evaluate Model
python -m src.evaluate
🎨 Run Web App
streamlit run app.py
🖼️ How to Use
Upload an image of a handwritten digit
Model processes the image
Output:
Predicted digit
Confidence score
Top-3 predictions
Probability chart
⚠️ Limitations
Works best with bold, high-contrast handwriting
Thin strokes may require better lighting or preprocessing
Model trained on EMNIST + synthetic data (not full real-world dataset)
🚀 Future Improvements
Add drawing canvas input
Train with real handwritten dataset
Improve preprocessing for thin strokes
Deploy on cloud (Streamlit Cloud / Hugging Face Spaces)
Mobile-friendly UI
🧠 Key Learnings
Data distribution > model complexity
Preprocessing consistency is critical
Synthetic data improves real-world performance
CNNs are sensitive to input format mismatch
👨‍💻 Author

Nigam Das
B.Tech AI/ML Student
Building real-world ML & DL projects 🚀

⭐ If you like this project

Give it a ⭐ on GitHub!
