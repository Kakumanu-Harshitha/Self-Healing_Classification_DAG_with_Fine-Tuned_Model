# nodes.py
from dataclasses import dataclass
from typing import Any, Dict, Callable, Optional
from logging_config import get_json_logger

logger = get_json_logger("nodes")

@dataclass
class InferenceNode:
    classifier: Any
    def run(self, text: str) -> Dict:
        res = self.classifier.predict(text)
        logger.info({"node":"InferenceNode","predicted_label":res["label"], "confidence": res["confidence"]})
        return res

@dataclass
class ConfidenceCheckNode:
    threshold: float = 0.75
    def run(self, inference_result: Dict) -> Dict:
        conf = inference_result["confidence"]
        decision = "ACCEPT" if conf >= self.threshold else "FALLBACK"
        logger.info({"node":"ConfidenceCheckNode","confidence":conf,"decision":decision})
        return {"decision":decision, "inference": inference_result}

@dataclass
class FallbackNode:
    fallback_strategy: str = "ask_user"  # or "backup_model"
    backup_model: Optional[Any] = None
    def run(self, text: str, inference_result: Dict, cli_fn: Callable):
        if self.fallback_strategy == "ask_user":
            # Friendly clarification question
            q = f"I am unsure — model predicted '{inference_result['label']}' with confidence {inference_result['confidence']:.2f}. " \
                f"Can you confirm the correct label? (negative/positive): "
            answer = cli_fn(q)
            answer_clean = answer.strip().lower()
            final = "Positive" if answer_clean.startswith("p") else "Negative"
            logger.info({"node":"FallbackNode","strategy":"ask_user","user_answer":answer_clean,"final_label":final})
            return {"final_label": final, "source":"user_clarification"}
        elif self.fallback_strategy == "backup_model" and self.backup_model is not None:
            alt = self.backup_model.predict(text)
            logger.info({"node":"FallbackNode","strategy":"backup_model","alt":alt})
            return {"final_label": alt["label"], "source":"backup_model", "alt_conf": alt["confidence"]}
        else:
            logger.info({"node":"FallbackNode","strategy":"none","final": inference_result["label"]})
            return {"final_label": inference_result["label"], "source":"no_action"}
