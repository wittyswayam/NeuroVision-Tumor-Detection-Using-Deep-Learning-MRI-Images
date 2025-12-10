# 📘 **NeuroVision – Brain Tumor Detection Using Deep Learning (MRI Images)**

NeuroVision is an advanced deep learning–powered medical imaging system designed to automatically classify brain MRI scans into tumor and non-tumor categories.
The project leverages a Convolutional Neural Network (CNN) architecture trained on a curated dataset of brain MRI images, enabling the model to detect subtle patterns and abnormalities that may indicate the presence of a tumor.

By integrating AI-driven analysis with traditional radiological workflows, NeuroVision assists radiologists in making faster, more accurate diagnostic decisions.
Its goal is to reduce human error, improve early tumor detection, and ultimately enhance patient outcomes through precise, data-backed medical imaging insights.

# 🔥 **Project Overview**

NeuroVision is a deep-learning–based system that detects the presence of brain tumors from MRI scans using a **Convolutional Neural Network (CNN)**.
The repository contains:

* Preprocessing scripts
* Training pipeline
* Final saved model
* Prediction script
* API version (FastAPI/Flask)

---

# 🧠 **Goal**

To build an accurate, reliable, and deployment-ready tumor detection system that can support radiologists in early diagnosis.

---

# 🗂️ Dataset Description

* Two classes: **yes/** (tumor) and **no/** (no tumor)
* ~3000 images
* Mixed resolutions
* Requires resizing & normalization
* Contains noise → handled via augmentation

---

# 🔧 **Text-Based Flowcharts (No Images Needed)**

## **1️⃣ End-to-End ML Pipeline**

```
RAW MRI IMAGES
       │
       ▼
[Data Preprocessing]
       │
       ▼
[Train-Test Split]
       │
       ▼
[CNN Model Training]
       │
       ▼
[Evaluation → Accuracy, Loss, CM]
       │
       ▼
[SAVED MODEL (.h5)]
       │
       ▼
[Prediction Script / API]
```

---

## **2️⃣ Data Preprocessing Workflow**

```
Load Image → Resize (150x150)
          → Normalize (0-1)
          → Augment (rotate/flip/zoom)
          → Convert to Array
          → Store in Dataset
```

---

## **3️⃣ CNN Architecture**

```
Input Layer (150x150x3)
        │
Conv2D → ReLU → MaxPool
        │
Conv2D → ReLU → MaxPool
        │
Flatten
        │
Dense → Dropout
        │
Output Layer (Softmax)
```

---

## **4️⃣ Prediction Pipeline**

```
User Image (.jpg/.png)
        │
        ▼
Preprocessing (resize → scale)
        │
        ▼
Model Predicts (0 or 1)
        │
        ▼
Final Output:
"Tumor Detected" / "No Tumor Detected"
```

---

# 🧪 **Model Performance Summary**

| Metric     | Value                       |
| ---------- | --------------------------- |
| Accuracy   | ~94–96%                     |
| Recall     | High (good tumor detection) |
| Precision  | Good (low false positives)  |
| Loss Curve | Stable after 15–20 epochs   |

---

# 📁 **Repository Structure**

```
NeuroVision/
│── dataset/
│   ├── yes/
│   ├── no/
│
│── model/
│   └── tumor_model.h5
│
│── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│
│── notebooks/
│   └── training.ipynb
│
│── api/
│   └── app.py
│
│── README.md
```

---

# 🚀 **API Version (FastAPI or Flask)**

### FastAPI Example

```python
from fastapi import FastAPI, UploadFile
from utils import load_model, predict_image

app = FastAPI()
model = load_model()

@app.post("/predict")
async def predict(file: UploadFile):
    result = predict_image(file, model)
    return {"prediction": result}
```

### Features

* Upload MRI → returns tumor result
* Automatic preprocessing
* Fast inference
* Deployable on Render, Railway, or Docker

---

# ▶️ **How to Run**

### Install packages

```
pip install -r requirements.txt
```

### Train model

```
python src/train.py
```

### Predict

```
python src/predict.py --image sample.jpg
```

### Run API

```
uvicorn api.app:app --reload
```

---
