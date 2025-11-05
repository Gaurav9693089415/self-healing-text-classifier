"""
load_backup.py
--------------
Utility for loading the backup zero-shot classifier (BART-MNLI).

Used when:
- Primary model confidence is low
- `--use_backup` flag is enabled via CLI

Responsibilities:
- Load HuggingFace zero-shot model
- Return ready-to-use inference pipeline
"""

from transformers import pipeline
from utils.config import BACKUP_MODEL_NAME


def load_backup_model(model_name: str = BACKUP_MODEL_NAME):
    """
    Load a zero-shot classification pipeline.

    Args:
        model_name (str): HF model identifier for zero-shot model.

    Returns:
        transformers.Pipeline: Zero-shot classifier.
    """
    print(f"[INFO] Loading backup zero-shot model: {model_name} ...")
    classifier = pipeline(
        "zero-shot-classification",
        model=model_name,
    )
    print("[INFO] Backup model loaded successfully.\n")
    return classifier
