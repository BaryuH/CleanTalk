import os
import numpy as np
import joblib
from tqdm import tqdm
import re
import unicodedata
from sentence_transformers import SentenceTransformer

os.chdir('.')
print(os.getcwd())

LABELS = ["toxic", "severe_toxic", "obscene",
          "threat", "insult", "identity_hate"]
MODEL_PATH = './data/output/svm_model.pkl'
MODEL_NAME = 'sentence-transformers/all-distilroberta-v1'
weights = {}
for power, col in enumerate(reversed(LABELS), start=1):
    weights[col] = 2 ** power

threshold_warning = 10
threshold_ban = 50


def compute_points(pred_vector):
    total = 0
    for label, value in zip(LABELS, pred_vector):
        if value == 1:
            total += weights[label]
    return total

def preprocess(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?'\-\"<>\[\]()/#&:%]", "", text)
    text = text.replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*\(\s*\)\s*", " ", text)
    text = re.sub(r"\s@\s*", ' ', text)
    text = re.sub(r"http\S+", "<URL>", text)
    text = re.sub(r"\S+@\S+", "<EMAIL>", text)
    text = re.sub(r"@\w+", "<USER>", text)
    text = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", "<IP>", text)
    text = re.sub(r'([!?.,;:\"\'])\1+', r'\1', text)
    time_pattern = re.compile(
        r"""
        (?:
            \b\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}\b
            | \b\d{4}[\/\-.]\d{1,2}[\/\-.]\d{1,2}\b
            | \b\d{1,2}\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s,]*\d{2,4}\b
            | \b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*\d{1,2},?\s*\d{2,4}\b
        )
        (?:\s*\(?(?:UTC|GMT|PST|EST|CET|IST)\)?)?
        | \b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|UTC|GMT)?\b
        """,
        flags=re.IGNORECASE | re.VERBOSE
    )
    text = re.sub(time_pattern, "<DATE>", text)
    text = re.sub(r"\"\s+", "", text)
    return text.strip()

def classify(points):
    if points >= threshold_ban:
        return "ban"
    elif points >= threshold_warning:
        return "warning"
    else:
        return "safe"


def main():
    print("Loading embedding model and SVM model...")
    encoder = SentenceTransformer(MODEL_NAME)
    svm_model = joblib.load(MODEL_PATH)['svm']
    print("Loaded successfully!\n")

    print("Nhập câu tiếng Anh để kiểm tra toxicity.")
    print("Gõ 'quit' / 'exit' để thoát.\n")

    while True:
        text = input(">> Enter comment: ").strip()
        if text.lower() in ["quit", "exit", "q"]:
            print("Bye!")
            break

        if not text:
            print("Empty input, thử lại.\n")
            continue

        text = preprocess(text)
        emb = encoder.encode([text])

        y_pred = svm_model.predict(emb)
        y_pred = np.array(y_pred)[0]

        points = compute_points(y_pred)
        final_label = classify(points)

        print("\n===== RESULT =====")
        print(f"Comment: {text}")
        print("Labels (0 = no, 1 = yes):")
        for label, value in zip(LABELS, y_pred):
            print(f"  {label:13s}: {int(value)}")

        print(f"\nPoints: {points}")
        print(f"Final label: {final_label.upper()}")
        print("====================\n")


if __name__ == "__main__":
    main()
