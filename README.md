# CleanTalk — Multi-Label Toxic Comment Detection System

CleanTalk is a **multi-label toxic comment detection system** designed to identify and classify harmful user-generated content (UGC) such as toxic, abusive, or offensive comments.  
The project combines **modern NLP sentence embeddings** with **traditional machine learning (SVM)** and includes both **library-based** and **from-scratch implementations** for learning and comparison purposes.

This repository is intended as a **technical showcase project** for AI / ML engineering, with a clear and modular structure that can be extended to deployment or further research.

---

## Project Goals

- Build an end-to-end NLP pipeline for toxic comment classification
- Explore the effectiveness of **Sentence Embeddings + SVM**
- Implement **custom SVM kernels** to deepen algorithmic understanding
- Maintain a clean, scalable project structure suitable for real-world systems

---

## Repository Structure

```
CleanTalk/
├── data/                     # Raw and processed datasets
├── kernel_svm/               # Custom SVM (kernel-based) implementations
├── notebooks/                # Jupyter notebooks for experimentation
├── src/                      # Core NLP and ML pipeline
│   ├── preprocess.py         # Text cleaning & normalization
│   ├── embed.py              # Sentence embedding generation
│   ├── svm_sklearn.py        # SVM using scikit-learn
│   ├── svm_scratch.py        # SVM implemented from scratch
│   ├── utils.py              # Helper functions
│   └── (other modules)
├── web/                      # Optional web interface / backend
├── example_SVM.ipynb         # Demonstration notebook
├── main.py                   # Pipeline entry point
├── requirements.txt          # Python dependencies
├── .gitignore
└── README.md                 # Project documentation
```

---

## System Pipeline Overview

### 1. Text Preprocessing
Raw text data is normalized and cleaned to reduce noise and improve downstream model performance.

Typical steps include:
- Lowercasing
- Regex-based cleanup
- Normalization of URLs, numbers, and special tokens
- Removal of duplicates and malformed entries

**Goal:** produce clean, semantically meaningful text for embedding.

---

### 2. Sentence Embeddings
CleanTalk converts text into dense vector representations using **Sentence Transformers**.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-distilroberta-v1")
embeddings = model.encode(texts)
```

---

### 3. Model Training

#### a) SVM with scikit-learn
- Linear SVM for multi-label classification
- Fast training and inference
- Strong baseline for text classification with dense embeddings

#### b) Custom SVM (from scratch)
Located in `kernel_svm/`, this implementation reimplements SVM optimization logic manually and supports kernel-based experimentation.

---

### 4. Optional Web Interface
The `web/` directory is reserved for potential deployment such as REST APIs or moderation UIs.

---

## Installation

```bash
git clone https://github.com/gitHuyNgo/CleanTalk.git
cd CleanTalk
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## Usage
### CleanTalk
#### Prepare Data
Change direction to `CleanTalk/src/data`
```bash
cd CleanTalk/src/data
python preprocess.py
python embeddings.py --inp "./data/processed/train.csv" --out "./data/embeddings/train.npy"
python embeddings.py --inp "./data/processed/test.csv"  --out "./data/embeddings/test.npy"
```
---
#### Train Model
Change direction to `CleanTalk/src/models`
```bash
cd CleanTalk/src/models
# Train Linear SVM using scikit-learn
python svm.py              # Uncomment prediction code in main() for Kaggle submission
# Train Soft Margin SVM implemented from scratch (Pegasos SVM)
python svm_scratch_SGD.py  # Uncomment prediction code in main() for Kaggle submission
```
---
#### Main CleanTalk Webpage
Change direction to `CleanTalk/web`
```bash
cd CleanTalk/web
pip install -r requirements.txt
# Copy the trained model from output to the backend model store:
cp PATH/CleanTalk/data/output/svm_model.pkl backend/models_store/CleanTalk1/
# Turn on the backend server:
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000
# Open the frontend UI by following this link:
http://127.0.0.1:3000/web/frontend/public/index.html
```

### Draw Digit 
Change direction to CleanTalk/kernel_svm
```bash
python draw_digit_app.py
```

---

## Future Work

- Add FastAPI-based inference service
- Introduce real-time moderation UI
- Improve evaluation metrics and logging
- Compare against fine-tuned transformer models