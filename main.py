import os
import numpy as np
import joblib
from tqdm import tqdm
from scipy.special import expit
from src.utils.preprocess import preprocess
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
