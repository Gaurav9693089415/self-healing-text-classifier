"""
config.py
---------
Central configuration module for the Self-Healing Text Classification System.

Responsibilities:
- Store all global constants and settings.
- Avoid hard-coding values inside individual nodes.
- Centralize model names, thresholds, log paths, and dataset settings.
"""

import os

# ==================== MODEL CONFIGURATION ====================

# Base model used for LoRA fine-tuning
MODEL_NAME = "distilbert-base-uncased"

# Directory where LoRA fine-tuned model is saved
LORA_OUTPUT_DIR = "./models/lora_finetuned"

# Backup zero-shot classifier model
BACKUP_MODEL_NAME = "facebook/bart-large-mnli"


# ==================== CLASSIFICATION SETTINGS ====================

# Label mapping for binary sentiment classification
LABELS = ["negative", "positive"]

# Confidence threshold for fallback trigger (0.0 to 1.0)
# Lower = more fallbacks, Higher = fewer fallbacks
CONFIDENCE_THRESHOLD = 0.70


# ==================== TOKENIZATION SETTINGS ====================

# Maximum sequence length for tokenization
MAX_LENGTH = 256


# ==================== TRAINING CONFIGURATION ====================

# Train-test split ratio
TEST_SPLIT = 0.1

# Default training hyperparameters
DEFAULT_EPOCHS = 2
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 2e-5

# LoRA Configuration
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["q_lin", "v_lin"]


# ==================== LOGGING CONFIGURATION ====================

# Logging paths
LOG_DIR = "./logs"
LOG_CSV = os.path.join(LOG_DIR, "classification.csv")
LOG_JSONL = os.path.join(LOG_DIR, "classification.jsonl")

# Maximum text length to log (to avoid huge log files)
LOG_TEXT_MAX_LENGTH = 200


# ==================== SYSTEM SETTINGS ====================

# Device preference (will fallback to CPU if CUDA unavailable)
DEVICE = "cuda"  # or "cpu" to force CPU usage

# Random seed for reproducibility
RANDOM_SEED = 42


# ==================== VALIDATION ====================


def validate_config():
    """
    Validate configuration settings at startup.
    Catches common configuration errors early.
    """
    assert (
        0.0 <= CONFIDENCE_THRESHOLD <= 1.0
    ), f"CONFIDENCE_THRESHOLD must be between 0 and 1, got {CONFIDENCE_THRESHOLD}"

    assert 0.0 < TEST_SPLIT < 1.0, f"TEST_SPLIT must be between 0 and 1, got {TEST_SPLIT}"

    assert MAX_LENGTH > 0, f"MAX_LENGTHmust be positive, got {MAX_LENGTH}"

    assert len(LABELS) >= 2, f"Must have at least 2 labels, got {len(LABELS)}"

    print("[CONFIG] ✓ Configuration validated successfully")


# Auto-validate on import
if __name__ != "__main__":
    try:
        validate_config()
    except AssertionError as e:
        print(f"[CONFIG] ✗ Configuration Error: {e}")
        raise
