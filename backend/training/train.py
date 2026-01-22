import os
from dataclasses import dataclass
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report

from datasets import DatasetDict
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from training.dataset import load_jigsaw, split_dataset, dataframe_to_hf_dataset, get_label_columns


@dataclass
class TrainingConfig:
    model_name: str = "distilbert-base-uncased"
    max_length: int = 128
    train_batch_size: int = 16
    eval_batch_size: int = 32
    lr: float = 2e-5
    num_epochs: int = 3
    output_dir: str = "models/safeprompt_distilbert"
    data_path: str = "data/train.csv"


def preprocess_function(examples, tokenizer, label_columns: List[str]):
    encodings = tokenizer(
        examples["comment_text"],
        truncation=True,
        padding="max_length",
        max_length=cfg.max_length,
    )
    labels = []
    for i in range(len(examples["comment_text"])):
        labels.append([float(examples[col][i]) for col in label_columns])
    encodings["labels"] = labels
    return encodings


def compute_metrics_fn(label_columns: List[str]):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        # multi-label: apply sigmoid then threshold
        probs = 1 / (1 + np.exp(-logits))
        y_true = labels
        y_pred = (probs >= 0.5).astype(int)

        metrics: Dict[str, float] = {}

        # macro F1 across all labels
        metrics["macro_f1"] = f1_score(
            y_true.reshape(-1), y_pred.reshape(-1), average="macro", zero_division=0
        )

        # per-label F1
        for i, label_name in enumerate(label_columns):
            metrics[f"f1_{label_name}"] = f1_score(
                y_true[:, i], y_pred[:, i], zero_division=0
            )

        return metrics

    return compute_metrics


if __name__ == "__main__":
    cfg = TrainingConfig()
    label_columns = get_label_columns()

    os.makedirs(cfg.output_dir, exist_ok=True)

    print("Loading data...")
    df = load_jigsaw(cfg.data_path)
    train_df, val_df = split_dataset(df, val_frac=0.1)

    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")

    train_ds = dataframe_to_hf_dataset(train_df)
    val_ds = dataframe_to_hf_dataset(val_df)
    ds_dict = DatasetDict({"train": train_ds, "validation": val_ds})

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    config = AutoConfig.from_pretrained(
        cfg.model_name,
        num_labels=len(label_columns),
        problem_type="multi_label_classification",
        id2label={i: l for i, l in enumerate(label_columns)},
        label2id={l: i for i, l in enumerate(label_columns)},
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        config=config,
    )

    print("Tokenizing datasets...")

    def _preprocess(examples):
        return preprocess_function(examples, tokenizer, label_columns)

    tokenized_datasets = ds_dict.map(_preprocess, batched=True)

    training_args = TrainingArguments(
    output_dir=cfg.output_dir,
    learning_rate=cfg.lr,
    per_device_train_batch_size=cfg.train_batch_size,
    per_device_eval_batch_size=cfg.eval_batch_size,
    num_train_epochs=cfg.num_epochs,
    weight_decay=0.01,

    save_strategy="no",
    save_steps=0,
    save_total_limit=0,
    # Works across older transformer versions
    logging_steps=100,
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_fn(label_columns),
    )

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    print("Evaluating on validation set...")
    preds = trainer.predict(tokenized_datasets["validation"])
    logits = preds.predictions
    probs = 1 / (1 + np.exp(-logits))
    y_true = np.array(tokenized_datasets["validation"]["labels"])
    y_pred = (probs >= 0.5).astype(int)

    print(classification_report(y_true, y_pred, target_names=label_columns, zero_division=0))
