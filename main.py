import os
import re
import unicodedata
import numpy as np
from pathlib import Path

from src.models.svm import Trainer as LibTrainer, Inference as LibInference


LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

LIB_MODEL_PATH = "./data/output/svm_model.pkl"

weights = {
    "toxic": 1,
    "obscene": 1,
    "insult": 1,
    "severe_toxic": 3,
    "identity_hate": 3,
    "threat": 4,
}

SAFE_MAX = 2
WARNING_MAX = 4


def preprocess(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?'\-\"<>\[\]()/#&:%]", "", text)
    text = text.replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*\(\s*\)\s*", " ", text)
    text = re.sub(r"\s@\s*", " ", text)
    text = re.sub(r"http\S+", "<URL>", text)
    text = re.sub(r"\S+@\S+", "<EMAIL>", text)
    text = re.sub(r"@\w+", "<USER>", text)
    text = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", "<IP>", text)
    text = re.sub(r"([!?.,;:\"\'])\1+", r"\1", text)
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
        flags=re.IGNORECASE | re.VERBOSE,
    )

    text = re.sub(time_pattern, "<DATE>", text)
    text = re.sub(r"\"\s+", "", text)
    return text.strip()


def compute_points(pred_vector):
    total = 0
    for label, value in zip(LABELS, pred_vector):
        if int(value) == 1:
            total += weights[label]
    return total


def classify(points, pred_vector):
    toxic, severe_toxic, obscene, threat, insult, identity_hate = pred_vector

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


def print_single(a):
    a_bins = np.asarray(a["bins"], dtype=int)

    a_points = compute_points(a_bins)
    a_final = classify(a_points, a_bins)

    print("\n========= RESULT ===========")
    print("label           score     bin")
    for lab, s, b in zip(LABELS, a["scores"], a_bins):
        print(f"{lab:14s} {float(s):7.4f}    {int(b)}")

    print(f"\nPoints: {a_points} | Final: {a_final.upper()}")
    print("============================\n")


def main():
    root = Path(__file__).resolve().parent
    os.chdir(root)

    svm_a, norm_a = LibTrainer.load_model(LIB_MODEL_PATH)
    infer_a = LibInference(svm_a, norm_a)

    print("CleanTalk - CLI inference")

    while True:
        print("Type 'quit' / 'exit' / 'q' to stop.\n")
        text = input(">> Enter comment: ").strip()
        if text.lower() in ["quit", "exit", "q"]:
            print("Bye!")
            break
        if not text:
            print("Empty input, again.\n")
            continue

        text_proc = preprocess(text)

        a = infer_a.predict_single(text_proc)

        print("\n========== INPUT ===========")
        print(f"Original: {text}")
        print(f"Preproc : {text_proc}")

        print_single(a)


if __name__ == "__main__":
    main()
