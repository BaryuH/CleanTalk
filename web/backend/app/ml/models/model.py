import pickle
from sentence_transformers import SentenceTransformer
from app.core.config import settings
from app.ml.preprocess.preprocessing import preprocess
from app.ml.services.scoring import compute_points, classify, LABELS


class CleanTalk1:
    def __init__(self) -> None:
        self.encoder = SentenceTransformer(
            settings.EMBEDDING_MODEL_NAME,
            cache_folder=str(settings.EMBEDDING_CACHE_DIR),
        )
        with open(settings.TOXICITY_MODEL_PATH, "rb") as f:
            obj = pickle.load(f)

        self.svm_model = obj["svm"]
        self.normalizer = obj.get("normalizer", None)

    def predict(self, text: str) -> dict:
        clean = preprocess(text)
        emb = self.encoder.encode(
            [clean], convert_to_numpy=True, normalize_embeddings=True
        )

        if self.normalizer is not None:
            emb = self.normalizer.transform(emb)
        y_pred = self.svm_model.predict(emb)[0]
        y_pred_int = [int(x) for x in y_pred]

        points = compute_points(y_pred_int)
        final_label = classify(points, y_pred_int)

        return {
            "clean_text": clean,
            "labels": {LABELS[i]: y_pred_int[i] for i in range(len(LABELS))},
            "points": points,
            "final_label": final_label,
        }
