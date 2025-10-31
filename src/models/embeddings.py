from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import os
os.chdir('..')

# ==== PATH CONFIG ====
INPUT_PATH = '../data/processed/train.csv'
OUTPUT_PATH = '../data/embeddings/train.npy'
MODEL_NAME = 'sentence-transformers/all-distilroberta-v1'

# ==== LOAD MODEL ====
model = SentenceTransformer(MODEL_NAME)

# ==== LOAD DATA ====
df = pd.read_csv(INPUT_PATH)
texts = df['comment_text'].fillna("").tolist()

# ==== ENCODE ====
embeddings = model.encode(
    texts,
    batch_size=128,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

# ==== MAPPING ID AND SAVE ====
ids = df['id'].values if 'id' in df.columns else np.arange(len(df))
embeddings_with_id = np.column_stack((ids, embeddings))

np.save(OUTPUT_PATH, embeddings_with_id)

print(f"✅ Saved embeddings to {OUTPUT_PATH}")
print(f"   Shape: {embeddings_with_id.shape} (id + {embeddings.shape[1]} dims)")