# 📝 Handwritten Notes OCR System (Image → Text → PDF)

A production-ready OCR system that converts **handwritten notes images into clean digital text and downloadable PDF files** using deep learning.

---

## 🚀 Features

* 📷 Upload handwritten image
* 🧠 OCR using **TrOCR (Transformer-based model)**
* 🧹 Automatic text cleaning (post-processing)
* 📄 Export extracted text as **PDF**
* 🌐 Interactive UI using **Streamlit**
* ⚡ Modular, scalable architecture

---

## 🧠 Tech Stack

* **Python**
* **OpenCV** – image preprocessing
* **Hugging Face Transformers (TrOCR)** – handwriting recognition
* **PyTorch** – deep learning backend
* **ReportLab** – PDF generation
* **Streamlit** – UI

---

## 📂 Project Structure

```
Handwritten Notes OCR System/
│
├── data/
│   └── samples/              # Sample test images
│
├── outputs/
│   ├── text/                 # Extracted text files
│   └── pdf/                  # Generated PDFs
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py         # Image preprocessing
│   ├── ocr_engine.py         # TrOCR-based OCR
│   ├── postprocess.py        # Text cleaning
│   ├── pdf_generator.py      # PDF creation
│   ├── pipeline.py           # End-to-end pipeline
│
├── app.py                   # Streamlit UI
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone git clone https://github.com/nigamdasmygoal003-sketch/Real-world-DL-projects.git
cd Real-world-DL-projects
```

---

### 2. Create virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate     # Linux / Mac
venv\Scripts\activate        # Windows
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

Then open your browser → upload image → get text + PDF ✅

---

## 🧪 Example Workflow

1. Upload handwritten image
2. Click **Run OCR**
3. View extracted text
4. Download PDF

---

## 📸 Sample Result

**Input:**
Handwritten note image

**Output:**

```
Hello my name is Nigam Das
```

---

## ⚠️ Notes

* First run will download the TrOCR model (~1–2GB)
* GPU optional (CPU works fine but slower)
* Works best with:

  * Clear handwriting
  * Good lighting
  * Minimal noise

---

## 🔥 Future Improvements

* 📦 Batch image processing
* 🌍 Multi-language OCR
* ✂️ Line & word segmentation
* 📊 Confidence scoring visualization
* ☁️ Deployment (Streamlit Cloud / Docker)
* 🧠 Custom-trained OCR model

---

## 💡 Key Learnings

* OCR performance depends heavily on:

  * preprocessing
  * model choice
  * postprocessing

* Transformer-based OCR (TrOCR) significantly outperforms traditional methods for handwriting.

---

## 🤝 Contributing

Contributions are welcome!
Feel free to open issues or submit pull requests.

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Nigam Das**

---

## ⭐ If you found this useful

Give this repo a ⭐ and share it!
