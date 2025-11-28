import os
import numpy as np
import joblib
from tqdm import tqdm
import re
import unicodedata
from scipy.special import expit
from sentence_transformers import SentenceTransformer

os.chdir('.')
print(os.getcwd())

LABELS = ["toxic", "severe_toxic", "obscene",
          "threat", "insult", "identity_hate"]

MODEL_PATH = './data/output/svm_model.pkl'
MODEL_NAME = 'sentence-transformers/all-distilroberta-v1'



MODEL_PATH = './data/output/svm_model.pkl'
MODEL_NAME = 'sentence-transformers/all-distilroberta-v1'


weights = {
    "toxic": 1,
    "obscene": 1,
    "insult": 1,
    "severe_toxic": 3,
    "identity_hate": 3,
    "threat": 4
}

SAFE_MAX = 2
WARNING_MAX = 4


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
    text = re.sub(r"([A-Za-z])\1{2,}", r"\1", text)
    text = re.sub(r"([A-Za-z])\1{1}", r"\1\1", text)
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


def classify(points, pred_vector):

    toxic, severe_toxic, obscene,  threat,  insult,  identity_hate, = pred_vector

    light_count = int(toxic) + int(obscene) + int(insult)
    heavy_count = int(severe_toxic) + int(identity_hate) + int(threat)
    total_on = light_count + heavy_count

    if total_on == 3:
        if heavy_count == 1 and light_count == 2:
            return "warning"
        if heavy_count == 2 and light_count == 1:
            return "ban"

    if threat == 1:
        return "ban"

    if points <= SAFE_MAX:
        return "safe"

    if points <= WARNING_MAX:
        return "warning"

    return "ban"


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

        text_proc = preprocess(text)
        emb = encoder.encode([text_proc])

        n_samples = emb.shape[0]
        n_labels = len(LABELS)
        scores = np.zeros((n_samples, n_labels), dtype=np.float32)
        raw_scores = np.zeros((n_samples, n_labels), dtype=float)

        for i, est in enumerate(svm_model.estimators_):
            score = est.decision_function(emb)[0]
            raw_scores[:, i] = score
            scores[:, i] = 1 if expit(score) >= 0.5 else 0

        scores = np.array(scores, dtype=int)
        pred_vector = scores[0]
        points = compute_points(pred_vector)
        final_label = classify(points, pred_vector)

        print("\n===== RESULT =====")
        print(f"Original: {text}")
        print(f"Preproc : {text_proc}")
        print("Raw scores (decision_function):")

        for label, value in zip(LABELS, raw_scores[0]):
            print(f"  {label:13s}: {value:.4f}")

        print("Labels (0 = no, 1 = yes):")
        for label, value in zip(LABELS, pred_vector):
            print(f"  {label:13s}: {int(value)}")

        print(f"\nPoints: {points}")
        print(f"Final label: {final_label.upper()}")
        print("====================\n")


if __name__ == "__main__":
    main()
