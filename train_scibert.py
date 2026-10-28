# ==============================================================================
# RUN IN GOOGLE COLAB (GPU RECOMMENDED) - DO NOT RUN ON LOCAL CPU
# Fine-tunes SciBERT (allenai/scibert_scivocab_uncased) on arXiv papers.
# Uses Early Stopping on the chronological validation set.
#
# Colab Quickstart:
#   1. Connect to T4 GPU runtime (Runtime -> Change runtime type -> T4 GPU)
#   2. !pip install transformers datasets torch scikit-learn pandas
#   3. python train_scibert.py --mode small    # ~3 minutes on Colab GPU
#   4. python train_scibert.py --mode full     # ~15 minutes on Colab GPU
# ==============================================================================

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "f1_macro": macro_f1}


def main():
    ap = argparse.ArgumentParser(description="Fine-tune SciBERT on arXiv categories.")
    ap.add_argument("--mode", default="small", choices=["small", "full"], help="small (fast demo) or full (full dataset)")
    ap.add_argument("--epochs", type=int, default=4, help="Maximum epochs")
    ap.add_argument("--batch-size", type=int, default=16, help="Batch size per device")
    ap.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    args = ap.parse_args()

    data_dir = Path("data")
    papers_path = data_dir / "papers.pkl"
    splits_path = data_dir / "splits.json"

    if not papers_path.exists() or not splits_path.exists():
        raise FileNotFoundError("Run prepare.py first to create data/papers.pkl and data/splits.json.")

    df = pd.read_pickle(papers_path)
    splits = json.loads(splits_path.read_text())

    train_idx = splits["train"]
    val_idx = splits["val"]
    test_idx = splits["test"]

    if args.mode == "small":
        train_idx = train_idx[:min(2000, len(train_idx))]
        val_idx = val_idx[:min(300, len(val_idx))]
        test_idx = test_idx[:min(300, len(test_idx))]
        print(f"[Small Mode] Training on {len(train_idx)}, Val on {len(val_idx)}, Test on {len(test_idx)}")
    else:
        print(f"[Full Mode] Training on {len(train_idx)}, Val on {len(val_idx)}, Test on {len(test_idx)}")

    # Encode labels
    classes = sorted(list(df.iloc[train_idx].primary_category.unique()))
    c2i = {c: i for i, c in enumerate(classes)}

    train_df = df.iloc[train_idx][df.iloc[train_idx].primary_category.isin(c2i)].copy()
    val_df = df.iloc[val_idx][df.iloc[val_idx].primary_category.isin(c2i)].copy()
    test_df = df.iloc[test_idx][df.iloc[test_idx].primary_category.isin(c2i)].copy()

    train_df["label"] = train_df.primary_category.map(c2i)
    val_df["label"] = val_df.primary_category.map(c2i)
    test_df["label"] = test_df.primary_category.map(c2i)

    train_df["text"] = train_df.title + ". " + train_df.abstract
    val_df["text"] = val_df.title + ". " + val_df.abstract
    test_df["text"] = test_df.title + ". " + test_df.abstract

    model_name = "allenai/scibert_scivocab_uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=256, padding="max_length")

    ds_train = Dataset.from_pandas(train_df[["text", "label"]].reset_index(drop=True)).map(tokenize_fn, batched=True)
    ds_val = Dataset.from_pandas(val_df[["text", "label"]].reset_index(drop=True)).map(tokenize_fn, batched=True)
    ds_test = Dataset.from_pandas(test_df[["text", "label"]].reset_index(drop=True)).map(tokenize_fn, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(classes))

    training_args = TrainingArguments(
        output_dir="out_scibert",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to=[],
        logging_steps=50,
        save_total_limit=1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    print("Beginning fine-tuning...")
    trainer.train()

    print("Evaluating on date-boundary test holdout...")
    eval_res = trainer.evaluate(ds_test)
    test_acc = round(float(eval_res["eval_accuracy"]), 4)
    test_f1 = round(float(eval_res["eval_f1_macro"]), 4)

    out_metrics = {
        "SciBERT (Fine-Tuned)": {
            "accuracy": test_acc,
            "f1_macro": test_f1,
            "note": "Evaluated on date-boundary holdout with early stopping in Colab GPU"
        }
    }
    out_file = data_dir / "scibert_metrics.json"
    out_file.write_text(json.dumps(out_metrics, indent=2))
    print(f"Results saved to {out_file}: Accuracy = {test_acc * 100:.2f}%, Macro-F1 = {test_f1:.4f}")


if __name__ == "__main__":
    main()
