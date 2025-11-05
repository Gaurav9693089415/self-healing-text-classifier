"""
train.py
--------
Runs LoRA fine-tuning for DistilBERT on the cleaned IMDB dataset.

Features:
- Tokenization + train/validation split
- LoRA adapter injection
- GPU-optimized training (FP16 + checkpointing)
- Saves trained model + tokenizer
"""


# ------------------------
# ALL IMPORTS FIRST (flake8 E402)
# ------------------------
import argparse
import os
import sys
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType

# ------------------------
# SYSTEM PATH FIX AFTER IMPORTS
# ------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from utils.config import (
    MODEL_NAME,
    MAX_LENGTH,
    TEST_SPLIT,
    DEFAULT_BATCH_SIZE,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_TARGET_MODULES,
    RANDOM_SEED,
)
from utils.logger import log_event, log_system_event


def load_data(csv_path: str, sample_size=None):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[ERROR] Dataset not found: {csv_path}")

    print(f"[INFO] Loading dataset → {csv_path}")
    df = pd.read_csv(csv_path)

    if "text" not in df.columns or "label" not in df.columns:
        raise KeyError("CSV must contain 'text' and 'label' columns")

    df = df.dropna(subset=["text", "label"])

    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"[INFO] Rows: {len(df)}")
    print(f"[INFO] Label counts: {df['label'].value_counts().to_dict()}")
    return df


def tokenize_dataset(df, tokenizer):
    tokenized = tokenizer(
        df["text"].astype(str).tolist(),
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )
    tokenized["labels"] = df["label"].astype(int).tolist()
    dataset = Dataset.from_dict(tokenized)
    return dataset.train_test_split(test_size=TEST_SPLIT, seed=RANDOM_SEED)


def train_model(csv_path, sample_size, output_dir, epochs, batch_size):

    print("\n==============================")
    print("     LORA FINE-TUNING")
    print("==============================\n")

    log_system_event(f"Training started with dataset: {csv_path}")

    try:
        df = load_data(csv_path, sample_size)

        print(f"\n[2/6] Loading model → {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

        print("\n[3/6] Tokenizing...")
        dataset = tokenize_dataset(df, tokenizer)

        print("\n[4/6] Applying LoRA...")
        lora_cfg = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES,
            task_type=TaskType.SEQ_CLS,
            bias="none",
        )
        model = get_peft_model(base_model, lora_cfg)

        if torch.cuda.is_available():
            model.to("cuda")
            print("[INFO] Using CUDA")
        else:
            print("[WARNING] Using CPU")

        print("\n[5/6] Configuring Trainer...")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        training_args = TrainingArguments(
            output_dir=os.path.join(output_dir, "checkpoints"),
            eval_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            load_best_model_at_end=True,
            save_total_limit=2,
            metric_for_best_model="eval_loss",
            logging_steps=20,
            logging_dir=os.path.join(output_dir, "logs"),
            report_to="none",
            seed=RANDOM_SEED,
            fp16=True,
            gradient_checkpointing=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
        )

        print("\n[6/6] Training...\n")
        trainer.train()

        print("\n[INFO] Evaluating...")
        res = trainer.evaluate()
        print(f"[INFO] Eval Loss: {res['eval_loss']:.4f}")

        os.makedirs(output_dir, exist_ok=True)
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        log_event(
            event="TrainingCompleted",
            text="train.csv",
            prediction="N/A",
            confidence=1.0,
            source="training",
            note=f"eval_loss={res['eval_loss']:.4f}",
        )

    except Exception as exc:
        print(f"[ERROR] Training failed: {exc}")
        log_system_event(f"Training failed: {exc}", level="ERROR")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="./data/train.csv")
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./models/lora_finetuned")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    train_model(
        csv_path=args.csv,
        sample_size=args.sample_size,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
