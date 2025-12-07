import joblib
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
        model_obj = joblib.load(settings.TOXICITY_MODEL_PATH)
        self.svm_model = model_obj["svm"]

    def predict(self, text: str) -> dict:
        clean = preprocess(text)
        emb = self.encoder.encode([clean])
        y_pred = self.svm_model.predict(emb)[0]
        points = compute_points(y_pred)
        final_label = classify(points, y_pred)
        return {
            "clean_text": clean,
            "labels": {LABELS[i]: int(y_pred[i]) for i in range(len(LABELS))},
            "points": points,
            "final_label": final_label,
        }
