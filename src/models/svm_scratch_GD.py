import os
import numpy as np
import pandas as pd
import pickle

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import Normalizer
from scipy.special import expit

os.chdir('..')

EMB_PATH = "./data/embeddings/train.npy"
RAW_PATH = "./data/processed/train.csv"
TEST_PATH = "./data/processed/test.csv"
SUBMIT_PATH = "./data/output/submit_scratch_GD.csv"
MODEL_FILE = "./data/output/svm_model_GD.pkl"
MODEL_NAME = "sentence-transformers/all-distilroberta-v1"
LABEL_COLS = ["toxic", "severe_toxic", "obscene",
              "threat", "insult", "identity_hate"]


class SVM_classifier:
    def __init__(
        self,
        learning_rate=1e-3,
        no_of_epochs=10,
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

                    grad_w_hinge = - \
                        (X_mis.T @ (sw_mis * y_mis)) / max(sw_sum, 1e-8)
                    grad_b_hinge = - np.sum(sw_mis * y_mis) / max(sw_sum, 1e-8)

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
        print(f"Loading embeddings from {EMB_PATH}")
        emb_data = np.load(EMB_PATH, allow_pickle=True)
        ids = emb_data[:, 0].astype(str)
        X_full = emb_data[:, 1:].astype(np.float32)

        raw_data = pd.read_csv(RAW_PATH)
        print(f"Loaded raw train from {RAW_PATH}, shape = {raw_data.shape}")

        df_emb = pd.DataFrame({"id": ids, "emb_idx": np.arange(len(ids))})
        df_merged = raw_data[["id"] +
                             LABEL_COLS].merge(df_emb, on="id", how="inner")

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
                learning_rate=1e-4,
                no_of_epochs=20,
                lambda_parameter=1e-4,
                batch_size=512,
                shuffle=True,
                class_weight="balanced",
            )

            model.fit(X_norm, y)
            self.models[col] = model

        save_obj = {
            "models": self.models,
            "normalizer": self.normalizer,
            "label_cols": LABEL_COLS,
        }

        os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
        with open(MODEL_FILE, "wb") as f:
            pickle.dump(save_obj, f)

        print(f"\nAll models saved to: {MODEL_FILE}")


class Inference:
    def __init__(self):
        print(f"Loading model bundle from {MODEL_FILE}")
        with open(MODEL_FILE, "rb") as f:
            saved = pickle.load(f)

        self.models = saved["models"]
        self.normalizer = saved["normalizer"]
        self.label_cols = saved["label_cols"]

        self.encoder = SentenceTransformer(MODEL_NAME)
        print("Loaded SentenceTransformer and SVM models.\n")

    def predict_file_kaggle(self, test_path=TEST_PATH, out_path=SUBMIT_PATH):

        df_test = pd.read_csv(test_path)
        print(f"Loaded test file: {test_path}, shape = {df_test.shape}")

        texts = df_test["comment_text"].fillna("").tolist()

        print("Encoding SBERT embeddings for test...")
        emb = self.encoder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=True,
        )

        emb_norm = self.normalizer.transform(emb)

        preds = np.zeros((len(df_test), len(self.label_cols)),
                         dtype=np.float32)

        for j, col in enumerate(self.label_cols):
            print(f"Predicting (scores) for label: {col}")
            scores = self.models[col].decision_function(emb_norm)
            probs = expit(scores)
            preds[:, j] = probs

        out = pd.DataFrame({"id": df_test["id"]})
        for j, col in enumerate(self.label_cols):
            out[col] = preds[:, j]

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        out.to_csv(out_path, index=False)
        print(f"\nSaved KAGGLE-style predictions to: {out_path}")

        return out


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train_all()

    infer = Inference()
    infer.predict_file_kaggle(TEST_PATH, SUBMIT_PATH)
