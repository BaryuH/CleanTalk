import os
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer

os.chdir('.')

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

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

def classify(points):
    if points >= threshold_ban:
        return "ban"
    elif points >= threshold_warning:
        return "warning"
    else:
        return "safe"
    

def main():
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-distilroberta-v1"
    SVM_MODEL_PATH = "svm_model.pkl"  

    print("Loading embedding model and SVM model...")
    encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    svm_model = joblib.load(SVM_MODEL_PATH)
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