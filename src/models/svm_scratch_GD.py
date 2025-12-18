import os
import numpy as np
import pandas as pd
import pickle

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import Normalizer
from scipy.special import expit
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)

TRAIN_EMB_PATH = "./data/embeddings/train.npy"
RAW_PATH = "./data/processed/train.csv"
TEST_PATH = "./data/processed/test2.csv"
TEST_EMB_PATH = "./data/embeddings/test.npy"
RES_PATH = "./data/output/submit_GD.csv"
MODEL_PATH = "./data/output/svm_gd_model.pkl"

MODEL_NAME = "sentence-transformers/all-distilroberta-v1"
LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

BATCH_SIZE = 128


class SVM_classifier:
    def __init__(
        self,
        learning_rate=5e-5,
        no_of_epochs=30,
        lambda_parameter=1e-4,
        batch_size=512,
        shuffle=True,
        class_weight=None,
    ):
        self.learning_rate = learning_rate
        self.no_of_epochs = no_of_epochs
        self.lambda_parameter = lambda_parameter
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.class_weight = class_weight

        self.w = None
        self.b = None
        self.m = None
        self.n = None

    def fit(self, X, Y):
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y).ravel()

        self.m, self.n = X.shape
        self.w = np.zeros(self.n, dtype=np.float32)
        self.b = 0.0

        y_label = np.where(Y <= 0, -1, 1).astype(np.float32)

        if self.class_weight == "balanced":
            m = len(Y)
            pos_count = (Y == 1).sum()
            neg_count = (Y == 0).sum()
            w_pos = m / (2.0 * max(pos_count, 1))
            w_neg = m / (2.0 * max(neg_count, 1))
            sample_weight = np.where(Y == 1, w_pos, w_neg).astype(np.float32)
        else:
            sample_weight = np.ones_like(Y, dtype=np.float32)

        lr = self.learning_rate

        for epoch in range(self.no_of_epochs):
            if self.shuffle:
                idx = np.random.permutation(self.m)
                X_epoch = X[idx]
                y_epoch = y_label[idx]
                sw_epoch = sample_weight[idx]
            else:
                X_epoch = X
                y_epoch = y_label
                sw_epoch = sample_weight

            for start in range(0, self.m, self.batch_size):
                end = start + self.batch_size
                X_batch = X_epoch[start:end]
                y_batch = y_epoch[start:end]
                sw_batch = sw_epoch[start:end]

                if X_batch.shape[0] == 0:
                    continue

                margins = y_batch * (X_batch @ self.w + self.b)
                mis_idx = margins < 1.0

                if not np.any(mis_idx):
                    grad_w = self.lambda_parameter * self.w
                    grad_b = 0.0
                else:
                    X_mis = X_batch[mis_idx]
                    y_mis = y_batch[mis_idx]
                    sw_mis = sw_batch[mis_idx]

                    sw_sum = np.sum(sw_batch)

                    grad_w_hinge = -(X_mis.T @ (sw_mis * y_mis)) / max(sw_sum, 1e-8)
                    grad_b_hinge = -np.sum(sw_mis * y_mis) / max(sw_sum, 1e-8)

                    grad_w = self.lambda_parameter * self.w + grad_w_hinge
                    grad_b = grad_b_hinge

                self.w = self.w - lr * grad_w
                self.b = self.b - lr * grad_b

        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float32)
        return X @ self.w + self.b

    def predict(self, X):
        scores = self.decision_function(X)
        y_sign = np.sign(scores)
        y_hat = np.where(y_sign <= 0, 0, 1)
        return y_hat


class Trainer:
    def __init__(self):
        self.normalizer = Normalizer(norm="l2")
        self.models = {}

    def load_data(self):
        print(f"Loading embeddings from {TRAIN_EMB_PATH}")
        emb_data = np.load(TRAIN_EMB_PATH, allow_pickle=True)
        ids = emb_data[:, 0].astype(str)
        X_full = emb_data[:, 1:].astype(np.float32)

        raw_data = pd.read_csv(RAW_PATH)
        print(f"Loaded raw train from {RAW_PATH}, shape = {raw_data.shape}")

        df_emb = pd.DataFrame({"id": ids, "emb_idx": np.arange(len(ids))})
        df_merged = raw_data[["id"] + LABEL_COLS].merge(df_emb, on="id", how="inner")

        matched_indices = df_merged["emb_idx"].values
        X = X_full[matched_indices]
        y_df = df_merged[LABEL_COLS].astype(int)

        print(f"Matched: {len(df_merged)} samples")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y_df.shape}")

        return X, y_df

    def train_all(self):
        X, y_df = self.load_data()

        X_norm = self.normalizer.fit_transform(X)
        print("X normalized with L2.")

        for col in LABEL_COLS:
            print(f"\n========== Training label: {col} ==========")
            y = y_df[col].values

            model = SVM_classifier(
                learning_rate=5e-5,
                no_of_epochs=50,
                lambda_parameter=1e-4,
                batch_size=1024,
                shuffle=True,
                class_weight="balanced",
            )

            model.fit(X_norm, y)
            self.models[col] = model

        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({"svm": self.models, "normalizer": self.normalizer}, f)

        print(f"\nAll models saved to: {MODEL_PATH}")
        return self.models, self.normalizer

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
        print(f"Predict on: {len(df_test)} rows")

        test_ids = df_test["id"].astype(str).values

        if test_emb is not None and os.path.exists(test_emb):
            emb_data = np.load(test_emb, allow_pickle=True)
            emb_ids = emb_data[:, 0].astype(str)
            X_full = emb_data[:, 1:].astype(np.float32)

            id2idx = {}
            for i, emb_id in enumerate(emb_ids):
                id2idx[emb_id] = i

            idx = np.empty(len(test_ids), dtype=np.int64)
            for i, _id in enumerate(test_ids):
                idx[i] = id2idx[_id]  

            embeddings = X_full[idx]
            print(f"Loaded embeddings from: {test_emb}")
            print(f"Embeddings shape: {embeddings.shape}")

        else:
            print("No test embedding found. Encoding test texts from scratch...")
            texts = df_test["comment_text"].fillna("").tolist()
            embeddings = self.model.encode(
                texts,
                batch_size=BATCH_SIZE,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

        embeddings = self.normalizer.transform(embeddings)

        scores = np.zeros((embeddings.shape[0], len(LABEL_COLS)), dtype=np.float32)
        for j, label in enumerate(LABEL_COLS):
            scores[:, j] = self.svm[label].decision_function(embeddings)

        probs = expit(scores)

        result = pd.DataFrame({"id": df_test["id"]})
        for i, label in enumerate(LABEL_COLS):
            result[label] = probs[:, i]

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        result.to_csv(output_file, index=False)
        print(f"Saved predictions to: {output_file}")
        return result

    def predict_single(self, text: str):
        embedding = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        embedding = self.normalizer.transform(embedding)

        scores = np.zeros(len(LABEL_COLS), dtype=np.float32)
        for j, label in enumerate(LABEL_COLS):
            scores[j] = self.svm[label].decision_function(embedding)[0]

        probs = expit(scores)

        print("LABEL\t\tSCORE")
        print("-" * 25)
        for label, score in zip(LABEL_COLS, probs):
            print(f"{label:<15}\t{score:.4f}")


if __name__ == "__main__":
    not_retrain = False  # True -> load existing model if available
    if os.path.exists(MODEL_PATH) and not_retrain:
        svm, normalizer = Trainer.load_model(MODEL_PATH)
        print(f"Loaded model: {MODEL_PATH}")
    else:
        trainer = Trainer()
        trainer.train_all()
        svm, normalizer = Trainer.load_model(MODEL_PATH)
    inference = Inference(svm, normalizer)
    inference.predict_file(TEST_PATH, RES_PATH, TEST_EMB_PATH)
    print(f"saved {RES_PATH}")
