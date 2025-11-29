import os
import numpy as np
import pandas as pd
import pickle

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from scipy.special import expit

os.chdir('.')

EMB_PATH = '../data/embeddings/train.npy'
RAW_PATH = '../data/processed/train.csv'

TEST_EMB_PATH = '../data/embeddings/test.npy'
TEST_CSV_PATH = '../data/processed/test.csv'

SUBMIT_PATH = '../data/output/submit_SGD_21.csv'
MODEL_FILE = '../data/output/svm_scratch_sgd_model.pkl'

MODEL_NAME = 'sentence-transformers/all-distilroberta-v1'

LABEL_COLS = ["toxic", "severe_toxic", "obscene",
              "threat", "insult", "identity_hate"]

TEST_SIZE = 0.2
RANDOM_STATE = 42

BATCH_SIZE_EMB = 8

LAMBDA_PARAM = 1e-4
N_EPOCHS = 20
BATCH_SIZE_SVM = 200
SHUFFLE = True


class PegasosSVM:

    def __init__(self,
                 lambda_param=1e-4,
                 n_epochs=20,
                 batch_size=200,
                 shuffle=True):
        self.lambda_param = lambda_param
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.w = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).ravel()
        y = np.where(y <= 0, -1, 1).astype(np.float32)

        n_samples, n_features = X.shape
        self.w = np.zeros(n_features, dtype=np.float32)

        t = 0

        for epoch in range(self.n_epochs):
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
                    grad_hinge = - (X_mis.T @ y_mis) / m
                else:
                    grad_hinge = 0.0

                self.w = (1 - eta_t * self.lambda_param) * \
                    self.w - eta_t * grad_hinge

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
        self.normalizer = Normalizer(norm='l2')
        self.models = {}
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None

    def load_data(self):
        emb_data = np.load(EMB_PATH, allow_pickle=True)
        raw_data = pd.read_csv(RAW_PATH)

        ids = emb_data[:, 0].astype(str)
        X_full = emb_data[:, 1:].astype(np.float32)

        df_emb = pd.DataFrame({'id': ids, 'emb_idx': np.arange(len(ids))})
        df_merged = raw_data[['id'] +
                             LABEL_COLS].merge(df_emb, on='id', how='inner')

        matched_indices = df_merged['emb_idx'].values
        X = X_full[matched_indices]
        y = df_merged[LABEL_COLS].values.astype(int)

        print(f"Matched samples: {len(df_merged)}")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")

        return X, y

    def preprocess(self, X, y):
        X_norm = self.normalizer.fit_transform(X)

        ones_col = np.ones((X_norm.shape[0], 1), dtype=np.float32)
        X_aug = np.hstack([X_norm, ones_col])   # (N, d+1)

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_aug, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )

        print(f"Train: {self.X_train.shape[0]}")
        print(f"Val  : {self.X_val.shape[0]}")

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

    def train_all(self):
        X, y = self.load_data()
        self.preprocess(X, y)

        print("\n===== TRAINING PEGASOS SVM (FROM SCRATCH, BIAS IN W) =====")
        print(
            f"lambda={LAMBDA_PARAM}, epochs={N_EPOCHS}, batch_size={BATCH_SIZE_SVM}\n")

        for j, col in enumerate(LABEL_COLS):
            print(f"--- Training label: {col} ---")
            y_col_train = self.y_train[:, j]

            true_pos_rate = y_col_train.mean()
            print(f"  true 1-rate (raw train) = {true_pos_rate:.4f}")

            target_pos_ratio = np.clip(true_pos_rate * 20.0, 0.05, 0.30)
            print(f"  using target_pos_ratio  = {target_pos_ratio:.3f}")

            X_bal, y_bal = self.balance_pos(self.X_train, y_col_train,
                                            target_pos_ratio=target_pos_ratio)

            svm = PegasosSVM(
                lambda_param=LAMBDA_PARAM,
                n_epochs=N_EPOCHS,
                batch_size=BATCH_SIZE_SVM,
                shuffle=SHUFFLE
            )
            svm.fit(X_bal, y_bal)
            self.models[col] = svm

            scores_train = svm.decision_function(self.X_train)
            preds_train = (scores_train >= 0).astype(int)
            pred_pos_rate = preds_train.mean()
            acc_train = (preds_train == y_col_train).mean()

            print(f"  pred 1-rate (raw train) = {pred_pos_rate:.4f}, "
                  f"train acc = {acc_train:.4f}")
            print(f"  score range on train    = [{scores_train.min():.4f}, "
                  f"{scores_train.max():.4f}]\n")

        os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
        with open(MODEL_FILE, "wb") as f:
            pickle.dump({
                "models": self.models,
                "normalizer": self.normalizer,
                "label_cols": LABEL_COLS
            }, f)

        print(f"All models saved to {MODEL_FILE}")


class Inference:
    def __init__(self):
        print(f"Loading model bundle from {MODEL_FILE}")
        with open(MODEL_FILE, "rb") as f:
            saved = pickle.load(f)

        self.models = saved["models"]
        self.normalizer = saved["normalizer"]
        self.label_cols = saved["label_cols"]

        self.encoder = SentenceTransformer(MODEL_NAME)
        print("Loaded SVM models & normalizer.\n")

    def predict_file(self,
                     test_emb_path=TEST_EMB_PATH,
                     test_csv_path=TEST_CSV_PATH,
                     out_path=SUBMIT_PATH):

        emb_test = np.load(test_emb_path, allow_pickle=True)
        ids_emb = emb_test[:, 0].astype(str)
        X_full = emb_test[:, 1:].astype(np.float32)

        df_test = pd.read_csv(test_csv_path)
        print(f"Loaded test csv: {test_csv_path}, shape = {df_test.shape}")
        print(
            f"Loaded test embedding: {test_emb_path}, shape = {emb_test.shape}")

        df_emb = pd.DataFrame(
            {'id': ids_emb, 'emb_idx': np.arange(len(ids_emb))})
        df_merged = df_test[['id']].merge(df_emb, on='id', how='inner')

        matched_indices = df_merged['emb_idx'].values
        X_raw = X_full[matched_indices]

        X_norm = self.normalizer.transform(X_raw)
        ones_col = np.ones((X_norm.shape[0], 1), dtype=np.float32)
        X_test = np.hstack([X_norm, ones_col])

        print(f"After merge: X_test shape = {X_test.shape}")

        n_samples = X_test.shape[0]
        n_labels = len(self.label_cols)
        scores = np.zeros((n_samples, n_labels), dtype=np.float32)

        for j, col in enumerate(self.label_cols):
            print(f"Predicting scores for label: {col}")
            model = self.models[col]
            scores[:, j] = model.decision_function(X_test)

        probs = expit(scores)

        out = pd.DataFrame({"id": df_merged["id"]})
        for j, col in enumerate(self.label_cols):
            out[col] = probs[:, j]

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        out.to_csv(out_path, index=False)
        print(f"\nSaved Kaggle-style predictions to: {out_path}")

        return out


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train_all()

    infer = Inference()
    infer.predict_file(TEST_EMB_PATH, TEST_CSV_PATH, SUBMIT_PATH)
