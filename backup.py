# backup.py
from transformers import pipeline
from logging_config import get_json_logger

logger = get_json_logger("backup")

class ZeroShotBackup:
    def __init__(self, model_name="facebook/bart-large-mnli", device=-1):
        # device=-1 uses CPU; set device=0 for GPU
        logger.info({"step":"loading_zero_shot", "model":model_name, "device":device})
        self.pipeline = pipeline("zero-shot-classification", model=model_name, device=device)

    def predict(self, text: str, candidate_labels=None):
        if candidate_labels is None:
            candidate_labels = ["positive", "negative"]
        out = self.pipeline(text, candidate_labels)
        # out contains 'labels' and 'scores' lists
        label = out["labels"][0].title()
        score = float(out["scores"][0])
        result = {"label": label, "confidence": score, "raw": out}
        logger.info({"event":"zero_shot_prediction", "text": text if len(text)<200 else text[:200]+"...", "result": result})
        return result
