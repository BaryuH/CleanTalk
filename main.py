import os
import numpy as np
import joblib
from tqdm import tqdm
from src.utils.preprocess import preprocess
from sentence_transformers import SentenceTransformer

os.chdir('.')

LABELS = ["toxic", "severe_toxic", "obscene",
          "threat", "insult", "identity_hate"]

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
    # print("Loading embedding model and SVM model...")
    # encoder = SentenceTransformer(MODEL_NAME)
    # svm_model = joblib.load(MODEL_PATH)['svm']
    # print("Loaded successfully!\n")

    # print("Nhập câu tiếng Anh để kiểm tra toxicity.")
    # print("Gõ 'quit' / 'exit' để thoát.\n")

    # while True:
    #     text = input(">> Enter comment: ").strip()

    #     if text.lower() in ["quit", "exit", "q"]:
    #         print("Bye!")
    #         break

    #     if not text:
    #         print("Empty input, thử lại.\n")
    #         continue

    #     text_proc = preprocess(text)
    #     emb = encoder.encode([text_proc])

    #     y_pred = svm_model.predict(emb)
    #     y_pred = np.array(y_pred)[0]

    #     points = compute_points(y_pred)
    #     final_label = classify(points, y_pred)

    #     print("\n===== RESULT =====")
    #     print(f"Original: {text}")
    #     print(f"Preproc : {text_proc}")
    #     print("Labels (0 = no, 1 = yes):")
    #     for label, value in zip(LABELS, y_pred):
    #         print(f"  {label:13s}: {int(value)}")

    #     print(f"\nPoints: {points}")
    #     print(f"Final label: {final_label.upper()}")
    #     print("====================\n")

    text = 'you are a nigga!!! 18/10/2025'
    text = preprocess(text)
    print(text)

if __name__ == "__main__":
    main()
