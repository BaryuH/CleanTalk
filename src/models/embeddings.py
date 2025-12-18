from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
MODEL_NAME = "sentence-transformers/all-distilroberta-v1"
BATCH_SIZE = 128

parser = argparse.ArgumentParser(description="Generate sentence embeddings")
parser.add_argument(
    "--inp",
    type=str,
    default="./data/processed/train.csv",
    help="Path to input CSV file",
)
parser.add_argument(
    "--out",
    type=str,
    default="./data/embeddings/train.npy",
    help="Path to output embeddings",
)

args = parser.parse_args()
INPUT_PATH = args.inp
OUTPUT_PATH = args.out
# PROCESS
model = SentenceTransformer(MODEL_NAME)
df = pd.read_csv(INPUT_PATH)
texts = df["comment_text"].fillna("").tolist()
embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)
ids = df["id"].astype(str).values if "id" in df.columns else np.arange(len(df))
embeddings_with_id = np.column_stack((ids, embeddings))
np.save(OUTPUT_PATH, embeddings_with_id)
print(f"Saved embeddings to {OUTPUT_PATH}")
print(f"Shape: {embeddings_with_id.shape} (id + {embeddings.shape[1]} dims)")
