# cli_app.py
import os
from inference import Classifier
from backup import ZeroShotBackup
from nodes import InferenceNode, ConfidenceCheckNode, FallbackNode
from logging_config import get_json_logger
import argparse

logger = get_json_logger("cli_app")

def cli_input(prompt):
    return input(prompt).strip()

def main(args):
    model_dir = args.model_dir
    threshold = args.threshold
    strategy = args.strategy

    logger.info({"step":"start_cli","model_dir":model_dir,"threshold":threshold,"strategy":strategy})
    classifier = Classifier(model_dir=model_dir)
    backup_model = None
    if strategy == "backup_model":
        # prefer GPU if available
        device = 0 if classifier.device.startswith("cuda") else -1
        backup_model = ZeroShotBackup(device=device)

    inf_node = InferenceNode(classifier)
    cc_node = ConfidenceCheckNode(threshold=threshold)
    fb_node = FallbackNode(fallback_strategy=strategy, backup_model=backup_model)

    print("Self-healing classifier CLI. Type 'exit' to quit.")
    while True:
        text = input("\nInput: ").strip()
        if text.lower() in ("exit","quit"):
            print("Goodbye!")
            break

        inf_res = inf_node.run(text)
        print(f"[InferenceNode] Predicted label: {inf_res['label']} | Confidence: {inf_res['confidence']*100:.1f}%")
        cc_res = cc_node.run(inf_res)
        if cc_res["decision"] == "FALLBACK":
            print("[ConfidenceCheckNode] Confidence too low. Triggering fallback...")
            fb_out = fb_node.run(text, inf_res, cli_input)
            final_label = fb_out["final_label"]
            print(f"Final Label: {final_label} (resolved via {fb_out.get('source')})")
        else:
            print("[ConfidenceCheckNode] Confidence OK. Accepted.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default="./models/distilbert_lora_imdb", help="path to fine-tuned model")
    p.add_argument("--threshold", type=float, default=0.75, help="confidence threshold")
    p.add_argument("--strategy", choices=["ask_user","backup_model"], default="ask_user", help="fallback strategy")
    args = p.parse_args()
    main(args)
