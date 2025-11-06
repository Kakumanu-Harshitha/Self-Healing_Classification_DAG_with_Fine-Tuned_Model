# inference.py
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from logging_config import get_json_logger

logger = get_json_logger("inference")

class Classifier:
    def __init__(self, model_dir="./models/distilbert_lora_imdb", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info({"step":"loading_classifier", "model_dir": model_dir, "device": self.device})
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.model.eval()
        # label mapping for imdb
        self.label_map = {0: "Negative", 1: "Positive"}

    def predict(self, text: str, return_probs=False):
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model(**enc)
            logits = out.logits
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
            pred_idx = int(probs.argmax())
            conf = float(probs[pred_idx])
            result = {"label": self.label_map[pred_idx], "confidence": conf, "probs": probs.tolist()}
            if return_probs:
                result["logits"] = logits.cpu().numpy().tolist()
            logger.info({"event":"prediction","text": text if len(text)<200 else text[:200]+"...","result":result})
            return result
