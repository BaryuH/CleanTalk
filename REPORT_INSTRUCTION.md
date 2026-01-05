# REPORT INSTRUCTION

## Repository Structure

```
CleanTalk/
├── KernelSVM/                # Kernel-SVM implementations
├── LinearSVM/                # Linear-SVM implementations
├── data/                     # Raw and processed datasets
├── notebooks/                # Jupyter notebooks for experimentation
├── src/                      # Core NLP and ML pipeline
│   ├── preprocess.py         # Text cleaning & normalization
│   ├── embed.py              # Sentence embedding generation
│   ├── svm.py                # SVM using scikit-learn (LinearSVC)
│   ├── svm_scratch_SGD.py    # SVM implemented from scratch
│   ├── utils.py              # Helper functions
│   └── (other modules)
├── web/                      # Optional web interface / backend
├── main.py                   # Pipeline entry point
├── requirements.txt          # Python dependencies
├── .gitignore
├── README.md                 # Project documentation
└── REPORT_INSTRUCTION.md     # CS115 Report instruction
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