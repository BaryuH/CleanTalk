import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
os.chdir('.')
EMB_PATH = './data/embeddings/train.npy'
RAW_PATH = './data/raw/train.csv'
TEST_PATH = './data/raw/test2.csv'
RES_PATH = './data/raw/submit.csv'
MODEL_NAME = 'sentence-transformers/all-distilroberta-v1'

LABEL_COLS = ["toxic", "severe_toxic", "obscene",
              "threat", "insult", "identity_hate"]
TEST_SIZE = 0.2
RANDOM_STATE = 42
C_PARAM = 0.05
MAX_ITER = 8000
CLASS_WEIGHT = "balanced"
BATCH_SIZE = 32


class Trainer:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.normalizer = Normalizer(norm='l2')
        self.svm = None
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
        print(f"Matched: {len(df_merged)}")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        return X, y

    def preprocess(self, X, y):
        X = self.normalizer.fit_transform(X)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        print(f"Train: {self.X_train.shape[0]}")
        print(f"Val:   {self.X_val.shape[0]}")

    def train(self):
        print(f"\nTraining")
        print(f"C={C_PARAM}, max_iter={MAX_ITER}, class_weight={CLASS_WEIGHT}")
        self.svm = MultiOutputClassifier(
            LinearSVC(C=C_PARAM, max_iter=MAX_ITER, class_weight=CLASS_WEIGHT), n_jobs=-1)
        self.svm.fit(self.X_train, self.y_train)

    def run(self):
        X, y = self.load_data()
        self.preprocess(X, y)
        self.train()
        return self.svm, self.normalizer


class Inference:
    def __init__(self, svm, normalizer):
        self.model = SentenceTransformer(MODEL_NAME)
        self.svm = svm
        self.normalizer = normalizer

    def predict_file(self, test_file, output_file):
        df_test = pd.read_csv(test_file)
        print(f"{len(df_test)} row")
        texts = df_test['comment_text'].fillna("").tolist()
        embeddings = self.model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        embeddings = self.normalizer.transform(embeddings)
        res = self.svm.predict(embeddings)
        result = pd.DataFrame({'id': df_test['id']})
        for i, label in enumerate(LABEL_COLS):
            result[label] = res[:, i]
        result.to_csv(output_file, index=False)
        return result


if __name__ == "__main__":
    trainer = Trainer()
    svm, normalizer = trainer.run()
    inference = Inference(svm, normalizer)
    result = inference.predict_file(TEST_PATH, RES_PATH)
    print(f"\nsaved {RES_PATH}")
