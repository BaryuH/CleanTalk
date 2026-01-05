import os
import numpy as np
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
from scipy.special import expit
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

GRID_LOG_DIR = "./data/output"
GRID_LOG_PATH = os.path.join(
    GRID_LOG_DIR, f"linearsvc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
)


class Trainer:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.normalizer = Normalizer(norm="l2")
        self.svm = None
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None

        os.makedirs(GRID_LOG_DIR, exist_ok=True)
        self.log_fp = open(GRID_LOG_PATH, "w", encoding="utf-8")

    def _log(self, msg: str):
        print(msg)
        self.log_fp.write(msg + "\n")
        self.log_fp.flush()

    def close(self):
        try:
            self.log_fp.close()
        except Exception:
            pass

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

        self._log(f"Matched: {len(df_merged)}")
        self._log(f"X shape: {X.shape}")
        self._log(f"y shape: {y.shape}")
        return X, y

    def preprocess(self, X, y):
        X = self.normalizer.fit_transform(X)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        self._log(f"Train: {self.X_train.shape[0]}")
        self._log(f"Val:   {self.X_val.shape[0]}")

    def _eval_val_accuracy(self, clf, X_val, y_val):
        preds = clf.predict(X_val)
        return float((preds == y_val).mean())

    def grid_search_linearsvc(self, C_list, max_iter_list, class_weight_list):
        if C_PARAM not in C_list:
            C_list = list(C_list) + [C_PARAM]
        if MAX_ITER not in max_iter_list:
            max_iter_list = list(max_iter_list) + [MAX_ITER]
        if CLASS_WEIGHT not in class_weight_list:
            class_weight_list = list(class_weight_list) + [CLASS_WEIGHT]

        C_list = sorted(set(C_list))
        max_iter_list = sorted(set(max_iter_list))
        class_weight_list = list(dict.fromkeys(class_weight_list))

        self._log("\n===== GRID SEARCH LinearSVC =====")
        self._log(f"Grid C          : {C_list}")
        self._log(f"Grid max_iter   : {max_iter_list}")
        self._log(f"Grid class_weight: {class_weight_list}\n")
        self._log(f"Log file: {GRID_LOG_PATH}")

        best_val = -1.0
        best_params = None
        best_model = None

        for C in C_list:
            for it in max_iter_list:
                for cw in class_weight_list:
                    base = LinearSVC(C=C, max_iter=it, class_weight=cw)
                    clf = MultiOutputClassifier(base, n_jobs=-1)
                    clf.fit(self.X_train, self.y_train)

                    val_acc = self._eval_val_accuracy(clf, self.X_val, self.y_val)
                    self._log(
                        f"C={C:.6g}, max_iter={it}, class_weight={cw} => val_acc={val_acc:.4f}"
                    )

                    if val_acc > best_val:
                        best_val = val_acc
                        best_params = {
                            "C": float(C),
                            "max_iter": int(it),
                            "class_weight": cw,
                        }
                        best_model = clf

        self._log("\n>>> BEST PARAMS")
        self._log(
            f"C={best_params['C']}, max_iter={best_params['max_iter']}, "
            f"class_weight={best_params['class_weight']} | best_val_acc={best_val:.4f}"
        )
        return best_model, best_params, best_val

    def train(self):
        self._log("\n===== TRAINING =====")
        C_grid = [0.05, 0.1, 0.2, 0.25, 0.4]
        max_iter_grid = [1000]
        class_weight_grid = ["balanced"]

        best_model, best_params, best_val = self.grid_search_linearsvc(
            C_grid, max_iter_grid, class_weight_grid
        )

        self.svm = best_model
        model_dir = os.path.dirname(MODEL_PATH)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(
                {
                    "svm": self.svm,
                    "normalizer": self.normalizer,
                    "grid_log": GRID_LOG_PATH,
                    "best_params": best_params,
                    "best_val_acc": float(best_val),
                },
                f,
            )
        self._log(f"\nModel saved: {MODEL_PATH}")
        self._log(f"Grid log saved: {GRID_LOG_PATH}")
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
    not_retrain = False
    if os.path.exists(MODEL_PATH) and not_retrain:
        svm, normalizer = Trainer.load_model(MODEL_PATH)
        print(f"Loaded model: {MODEL_PATH}")
    else:
        trainer = Trainer()
        try:
            X, y = trainer.load_data()
            trainer.preprocess(X, y)
            svm, normalizer = trainer.train()
        finally:
            trainer.close()

    # Uncomment for predicting file
    # inference = Inference(svm, normalizer)
    # inference.predict_file(TEST_PATH, RES_PATH, TEST_EMB_PATH)
    # print(f"saved {RES_PATH}")
