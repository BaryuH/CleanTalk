# Clean Talk – Toxic Comment Classification

Clean Talk is a machine learning project that classifies toxic comments using the Jigsaw dataset on Kaggle.  
The system includes a robust NLP preprocessing pipeline, vector-based text embeddings, traditional ML modeling (SVM), and experimental LLM-based few-shot & zero-shot evaluation.

---

## 🚀 Features

- End-to-end toxic comment classification system.
- Robust preprocessing: regex normalization, text standardization (`<URL>`, `<DATE>`, etc.), deduplication.
- High-quality text embeddings using **all-distilroberta-v1**.
- SVM classifier using **scikit-learn** achieving **97% accuracy** on Kaggle.
- Custom SVM implemented from scratch achieving **95% accuracy**.
- Experimental evaluation using **GPT-5** for zero-shot & few-shot classification.

---

## 📂 Project Structure

```
CleanTalk/
│
├── data/                     # Raw and processed datasets
├── notebooks/                # Jupyter notebooks for experiments
├── src/
│   ├── preprocess.py         # Data cleaning & text normalization
│   ├── embed.py              # Sentence embedding generation
│   ├── svm_sklearn.py        # SVM using scikit-learn
│   ├── svm_scratch.py        # Custom SVM implementation
│   ├── gpt_eval.py           # GPT-5 zero-shot / few-shot evaluation
│   └── utils.py              # Helper functions
│
├── models/                   # Saved models & vectors
├── results/                  # Accuracy reports, logs
├── requirements.txt
└── README.md
```

---

## 🧹 Preprocessing Pipeline

The preprocessing pipeline includes:

- Lowercasing  
- Regex cleaning  
- Standardizing tokens (`<URL>`, `<DATE>`, `<NUMBER>`)  
- Reducing character repetition (`"meeeee"` → `"mee"`)  
- Removing duplicates  
- Removing non-informative samples  

This ensures the embeddings capture meaningful semantic patterns.

---

## 🔡 Text Embeddings

We use **all-distilroberta-v1** (Sentence Transformers) to convert comments into fixed-size embeddings.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-distilroberta-v1")
embeddings = model.encode(comments, batch_size=32)
```

---

## 🧠 Model Training

### **1. SVM (scikit-learn version)**  
Achieved **97% accuracy** on Kaggle.

```python
from sklearn.svm import LinearSVC

svm = LinearSVC()
svm.fit(X_train, y_train)
```

### **2. Custom SVM (from scratch)**  
A hand-built implementation for learning purposes.  
Achieved **95% accuracy** with efficient inference.

---

## 🤖 LLM Evaluation (GPT-5)

We also test zero-shot & few-shot classification using the GPT-5 API:

- Zero-shot toxicity classification  
- 3-shot prompt templates  
- Performance comparison against SVM baselines  

This module is exploratory and showcases traditional ML vs modern LLM behavior.

---

## 📊 Results

| Model                   | Accuracy |
|------------------------|----------|
| SVM (scikit-learn)     | **97%**  |
| Custom SVM (scratch)   | **95%**  |
| GPT-5 Zero-shot        | TBD      |
| GPT-5 Few-shot         | TBD      |

---

## 🛠 Tech Stack

- **Python**
- **scikit-learn**
- **Sentence Transformers**
- **NumPy / Pandas**
- **OpenAI GPT-5 API**
- **Regex-based preprocessing**

---

## 📦 Installation

```bash
git clone https://github.com/gitHuyNgo/CleanTalk
cd CleanTalk
pip install -r requirements.txt
```

---

## ▶️ Usage

### Preprocess data:
```bash
python src/preprocess.py
```

### Generate embeddings:
```bash
python src/embed.py
```

### Train SVM (scikit-learn):
```bash
python src/svm_sklearn.py
```

### Train custom SVM:
```bash
python src/svm_scratch.py
```

### Evaluate with GPT-5:
```bash
python src/gpt_eval.py
```

---

## 📌 Future Improvements

- Deploy model with FastAPI  
- Add web UI for real-time classification  
- Improve GPT-5 evaluation metrics  
- Add adversarial toxicity detection  
- Enhance data augmentation pipeline  