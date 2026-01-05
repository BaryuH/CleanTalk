# REPORT INSTRUCTION

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

## Environment Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage (Notebook Guide)

Please run the notebooks **from top to bottom** to ensure all plots are generated correctly.

### Hard Margin, Soft Margin, and Dual Maximization
To clearly observe and compare:
- Hard-margin SVM
- Soft-margin SVM

run the following notebook:

```
CleanTalk/LinearSVM/hard_soft_SVM.ipynb
```



### Kernel Margin Visualization
To better understand:
- Kernelized SVM
- Non-linear decision boundaries
- Margin behavior under different kernels
- Kernel-SVM in hand-written number predict 

run the following notebook:

```
CleanTalk/KernelSVM/kernel_svm_visuallization.ipynb
CleanTalk/KernelSVM/MNIST.ipynb
```



### CleanTalk Dataset EDA 
run the following notebook:
```
CleanTalk/notebooks/eda.ipynb
```