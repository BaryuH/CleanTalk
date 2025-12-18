import pandas as pd
from tqdm import tqdm
import re
import unicodedata
import os
from pathlib import Path
import argparse


ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing Data")
    parser.add_argument(
        "--inp",
        type=str,
        default="./data/raw/train.csv",
        help="Path to raw data",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="./data/processed/train.csv",
        help="Path to processed data",
    )
    args = parser.parse_args()
    INPUT_PATH = args.inp
    OUTPUT_PATH = args.out
    data = pd.read_csv(INPUT_PATH)
    name = "comment_text"
    tqdm.pandas()
    data[name] = data[name].progress_apply(preprocess)
    data.to_csv(OUTPUT_PATH, index=False)
