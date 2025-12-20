import os
import numpy as np
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from scipy.special import expit
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)

TRAIN_EMB_PATH = "./data/embeddings/train.npy"
RAW_PATH = "./data/processed/train.csv"
TEST_PATH = "./data/processed/test.csv"
TEST_EMB_PATH = "./data/embeddings/test.npy"
RES_PATH = "./data/output/submit_SGD.csv"
MODEL_PATH = "./data/output/svm_sgd_model.pkl"
MODEL_NAME = "sentence-transformers/all-distilroberta-v1"
LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 128
LAMBDA_PARAM = 1e-4
N_EPOCHS = 30
BATCH_SIZE_SVM = 200
SHUFFLE = True


class PegasosSVM:
    def __init__(self, lambda_param=1e-4, n_epochs=20, batch_size=200, shuffle=True):
        self.lambda_param = float(lambda_param)
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.w = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).ravel()
        y = np.where(y <= 0, -1.0, 1.0).astype(np.float32)

        n_samples, n_features = X.shape
        self.w = np.zeros(n_features, dtype=np.float32)

        t = 0
        for _ in range(self.n_epochs):
            if self.shuffle:
                idx = np.random.permutation(n_samples)
                X_epoch = X[idx]
                y_epoch = y[idx]
            else:
                X_epoch = X
                y_epoch = y

            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                X_batch = X_epoch[start:end]
                y_batch = y_epoch[start:end]
                if X_batch.shape[0] == 0:
                    continue

                t += 1
                eta_t = 1.0 / (self.lambda_param * t)

                margins = y_batch * (X_batch @ self.w)
                idx_mis = margins < 1.0

                if np.any(idx_mis):
                    X_mis = X_batch[idx_mis]
                    y_mis = y_batch[idx_mis]
                    m = X_batch.shape[0]
                    grad_hinge = -(X_mis.T @ y_mis) / m
                else:
                    grad_hinge = 0.0

                self.w = (1.0 - eta_t * self.lambda_param) * self.w - eta_t * grad_hinge

                norm_w = np.linalg.norm(self.w)
                r = 1.0 / np.sqrt(self.lambda_param)
                if norm_w > r:
                    self.w = self.w * (r / norm_w)

        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float32)
        return X @ self.w

    def predict(self, X):
        scores = self.decision_function(X)
        return (scores >= 0).astype(int)


class Trainer:
    def __init__(self):
        self.normalizer = Normalizer(norm="l2")
        self.models = {}
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

    def balance_pos(self, X, y, target_pos_ratio=0.1):
        y = np.asarray(y).ravel()
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]

        if len(pos_idx) == 0:
            return X, y

        n_neg = len(neg_idx)
        desired_pos = max(len(pos_idx), int(target_pos_ratio * n_neg))

        repeat = max(1, int(np.ceil(desired_pos / len(pos_idx))))
        pos_rep_idx = np.tile(pos_idx, repeat)

        new_idx = np.concatenate([neg_idx, pos_rep_idx])
        np.random.shuffle(new_idx)

        return X[new_idx], y[new_idx]

    def train(self):
        X, y = self.load_data()
        self.preprocess(X, y)

        print("\n===== TRAINING PEGASOS SVM (FROM SCRATCH, BIAS IN W) =====")
        print(
            f"lambda={LAMBDA_PARAM}, epochs={N_EPOCHS}, batch_size={BATCH_SIZE_SVM}\n"
        )

        for j, col in enumerate(LABEL_COLS):
            print(f"--- Training label: {col} ---")
            y_col_train = self.y_train[:, j]

            true_pos_rate = float(y_col_train.mean())
            print(f"  true 1-rate (raw train) = {true_pos_rate:.4f}")

            target_pos_ratio = float(np.clip(true_pos_rate * 20.0, 0.05, 0.30))
            print(f"  using target_pos_ratio  = {target_pos_ratio:.3f}")

            X_bal, y_bal = self.balance_pos(
                self.X_train, y_col_train, target_pos_ratio=target_pos_ratio
            )

            svm = PegasosSVM(
                lambda_param=LAMBDA_PARAM,
                n_epochs=N_EPOCHS,
                batch_size=BATCH_SIZE_SVM,
                shuffle=SHUFFLE,
            )
            svm.fit(X_bal, y_bal)
            self.models[col] = svm

            scores_train = svm.decision_function(self.X_train)
            preds_train = (scores_train >= 0).astype(int)
            pred_pos_rate = float(preds_train.mean())
            acc_train = float((preds_train == y_col_train).mean())

            print(
                f"  pred 1-rate (raw train) = {pred_pos_rate:.4f}, train acc = {acc_train:.4f}"
            )
            print(
                f"  score range on train    = [{scores_train.min():.4f}, {scores_train.max():.4f}]\n"
            )

        model_dir = os.path.dirname(MODEL_PATH)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({"svm": self.models, "normalizer": self.normalizer}, f)

        print(f"All models saved to {MODEL_PATH}")

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

            id2idx = {emb_id: i for i, emb_id in enumerate(emb_ids)}
            idx = np.empty(len(test_ids), dtype=np.int64)
            for i, _id in enumerate(test_ids):
                if _id not in id2idx:
                    raise KeyError(f"ID {_id} not found in embedding file")
                idx[i] = id2idx[_id]

            embeddings = X_full[idx]
            print(f"Loaded embeddings from: {test_emb}")
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

        scores = np.zeros((embeddings.shape[0], len(LABEL_COLS)), dtype=np.float32)
        for j, label in enumerate(LABEL_COLS):
            scores[:, j] = self.svm[label].decision_function(embeddings)

        probs = expit(scores)
        result = pd.DataFrame({"id": df_test["id"]})
        for i, label in enumerate(LABEL_COLS):
            result[label] = probs[:, i]

        out_dir = os.path.dirname(output_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        result.to_csv(output_file, index=False)
        print(f"Saved predictions to: {output_file}")
        return result

    def predict_single(self, text: str):
        embedding = self.model.encode(
            [text], convert_to_numpy=True, normalize_embeddings=True
        )
        embedding = self.normalizer.transform(embedding)

        scores = np.zeros((len(LABEL_COLS),), dtype=float)
        for j, label in enumerate(LABEL_COLS):
            scores[j] = float(self.svm[label].decision_function(embedding)[0])

        probs = expit(scores)
        bins = (scores >= 0).astype(int)
        return {"scores": scores, "probs": probs, "bins": bins}


if __name__ == "__main__":
    not_retrain = False
    if os.path.exists(MODEL_PATH) and not_retrain:
        svm, normalizer = Trainer.load_model(MODEL_PATH)
        print(f"Loaded model: {MODEL_PATH}")
    else:
        trainer = Trainer()
        trainer.train()
        svm, normalizer = Trainer.load_model(MODEL_PATH)
    # Uncomment for predicting file
    # inference = Inference(svm, normalizer)
    # inference.predict_file(TEST_PATH, RES_PATH, TEST_EMB_PATH)
    # print(f"saved {RES_PATH}")
