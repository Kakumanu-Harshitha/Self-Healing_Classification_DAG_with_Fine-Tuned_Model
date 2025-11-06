# train_lora.py
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
from peft import LoraConfig, get_peft_model
from logging_config import get_json_logger
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

logger = get_json_logger("train")

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "./models/distilbert_lora_imdb"
NUM_LABELS = 2

def tokenize_fn(batch, tokenizer):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    acc = accuracy_score(labels, preds)
    p, r, f, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f}

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info({"step":"load_dataset"})
    ds = load_dataset("imdb")

    logger.info({"step":"tokenizer_load", "model": MODEL_NAME})
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    ds = ds.map(lambda x: tokenize_fn(x, tokenizer), batched=True, remove_columns=["text"])
    ds = ds.rename_column("label", "labels")
    ds.set_format(type="torch")

    logger.info({"step":"load_model"})
    base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

    # Configure LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_lin","v_lin","k_lin","out_lin"] , # generic; peft will match if exists for model
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS"
    )
    model = get_peft_model(base_model, lora_config)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",             # <-- updated argument name
    save_strategy="epoch",
    num_train_epochs=3,
    learning_rate=2e-5,
    logging_steps=200,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    bf16=torch.cuda.is_available(),    # better precision flag for newer GPUs (use bf16 instead of fp16)
    )

    # For quicker dev runs you can select subsets:
    train_dataset = ds["train"]  # .select(range(20000))
    eval_dataset = ds["test"]    # .select(range(5000))

    logger.info({"step":"trainer_setup"})
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    logger.info({"step":"train_start"})
    trainer.train()
    logger.info({"step":"train_finished"})

    logger.info({"step":"saving_model", "path": OUTPUT_DIR})
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
