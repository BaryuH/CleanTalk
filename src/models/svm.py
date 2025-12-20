import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.multioutput import MultiOutputClassifier
from scipy.special import expit
from sklearn.svm import LinearSVC
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)

TRAIN_EMB_PATH = "./data/embeddings/train.npy"
TEST_EMB_PATH = "./data/embeddings/test.npy"
RAW_PATH = "./data/processed/train.csv"
TEST_PATH = "./data/processed/test.csv"
RES_PATH = "./data/output/submit_lib.csv"
MODEL_PATH = "./data/output/svm_model.pkl"
MODEL_NAME = "sentence-transformers/all-distilroberta-v1"
LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
TEST_SIZE = 0.2
RANDOM_STATE = 42
C_PARAM = 0.05
MAX_ITER = 1000
CLASS_WEIGHT = "balanced"
BATCH_SIZE = 128


class Trainer:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.normalizer = Normalizer(norm="l2")
        self.svm = None
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None

    def load_data(self):
        emb_data = np.load(TRAIN_EMB_PATH, allow_pickle=True)
        raw_data = pd.read_csv(RAW_PATH)
        ids = emb_data[:, 0].astype(str)
        X_full = emb_data[:, 1:].astype(np.float32)
        df_emb = pd.DataFrame({"id": ids, "emb_idx": np.arange(len(ids))})
        df_merged = raw_data[["id"] + LABEL_COLS].merge(df_emb, on="id", how="inner")
        matched_indices = df_merged["emb_idx"].values
        X = X_full[matched_indices]
        y = df_merged[LABEL_COLS].values.astype(int)
        print(f"Matched: {len(df_merged)}")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        return X, y

    def preprocess(self, X, y):
        X = self.normalizer.fit_transform(X)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        print(f"Train: {self.X_train.shape[0]}")
        print(f"Val:   {self.X_val.shape[0]}")

    def train(self):
        print("\nTraining")
        print(f"C={C_PARAM}, max_iter={MAX_ITER}, class_weight={CLASS_WEIGHT}")
        base = LinearSVC(C=C_PARAM, max_iter=MAX_ITER, class_weight=CLASS_WEIGHT)
        self.svm = MultiOutputClassifier(base, n_jobs=-1)
        self.svm.fit(self.X_train, self.y_train)

        model_dir = os.path.dirname(MODEL_PATH)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)

        with open(MODEL_PATH, "wb") as f:
            pickle.dump({"svm": self.svm, "normalizer": self.normalizer}, f)

        print(f"Model saved: {MODEL_PATH}")
        return self.svm, self.normalizer

    @staticmethod
    def load_model(model_path=MODEL_PATH):
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        return data["svm"], data["normalizer"]


class Inference:
    def __init__(self, svm, normalizer):
        self.model = SentenceTransformer(MODEL_NAME)
        self.svm = svm
        self.normalizer = normalizer

    def predict_file(self, test_file, output_file, test_emb=None):
        df_test = pd.read_csv(test_file)
        print(f"Predict on: {len(df_test)} row")
        test_ids = df_test["id"].astype(str).values

        if test_emb and os.path.exists(test_emb):
            emb_data = np.load(test_emb, allow_pickle=True)
            emb_ids = emb_data[:, 0].astype(str)
            X_full = emb_data[:, 1:].astype(np.float32)

            id2idx = {emb_id: i for i, emb_id in enumerate(emb_ids)}
            idx = np.empty(len(test_ids), dtype=np.int64)
            for i, _id in enumerate(test_ids):
                if _id not in id2idx:
                    raise KeyError(f"ID {_id} not found in embedding file")
                idx[i] = id2idx[_id]

            embeddings = X_full[idx]
            print(f"Loaded embeddings: {test_emb}")
            print(f"Embeddings shape: {embeddings.shape}")
        else:
            texts = df_test["comment_text"].fillna("").tolist()
            embeddings = self.model.encode(
                texts,
                batch_size=BATCH_SIZE,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

        embeddings = self.normalizer.transform(embeddings)
        scores = np.column_stack(
            [est.decision_function(embeddings) for est in self.svm.estimators_]
        )
        probs = expit(scores)
        result = pd.DataFrame({"id": df_test["id"]})
        for i, label in enumerate(LABEL_COLS):
            result[label] = probs[:, i]

        out_dir = os.path.dirname(output_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        result.to_csv(output_file, index=False)
        return result

    def predict_single(self, text: str):
        embedding = self.model.encode(
            [text], convert_to_numpy=True, normalize_embeddings=True
        )
        embedding = self.normalizer.transform(embedding)

        scores = np.array(
            [est.decision_function(embedding)[0] for est in self.svm.estimators_],
            dtype=float,
        )
        probs = expit(scores)
        bins = (scores >= 0).astype(int)
        return {"scores": scores, "probs": probs, "bins": bins}


if __name__ == "__main__":
    not_retrain = True
    if os.path.exists(MODEL_PATH) and not_retrain:
        svm, normalizer = Trainer.load_model(MODEL_PATH)
        print(f"Loaded model: {MODEL_PATH}")
    else:
        trainer = Trainer()
        X, y = trainer.load_data()
        trainer.preprocess(X, y)
        svm, normalizer = trainer.train()
    # Uncomment for predicting file
    # inference = Inference(svm, normalizer)
    # inference.predict_file(TEST_PATH, RES_PATH, TEST_EMB_PATH)
    # print(f"saved {RES_PATH}")

