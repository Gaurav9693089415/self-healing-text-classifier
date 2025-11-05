"""
inference.py
------------
Primary inference node for the Self-Healing Classification DAG.

Responsibilities:
- Load LoRA-fine-tuned DistilBERT model + tokenizer
- Generate prediction & confidence score
- Stream output with color coding (Positive=Green, Negative=Red)
- Log inference events for analysis
"""

import torch
import time
from colorama import Fore, Style
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils.config import LABELS, MAX_LENGTH
from utils.logger import log_event


def stream_line(text, delay=0.012):
    """Stream plain text (NO ANSI escape codes)."""
    for ch in text:
        print(ch, end="", flush=True)
        time.sleep(delay)
    print()


class InferenceNode:
    """Primary inference node using the fine-tuned model."""

    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def __call__(self, state: dict) -> dict:
        text = state["text"]

        # Tokenization
        inputs = self.tokenizer(
            text, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LENGTH
        ).to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            confidence = float(probs.max().cpu().item())
            label_idx = int(probs.argmax().cpu().item())
            prediction = LABELS[label_idx]

        # Color-coded label (printed normally, not streamed)
        color = Fore.GREEN if prediction == "positive" else Fore.RED
        print(
            f"{Fore.CYAN}[InferenceNode]{Style.RESET_ALL} "
            f"Predicted: {color}{prediction.capitalize()}{Style.RESET_ALL} "
            f"| Confidence: {confidence:.0%}"
        )

        # Log event
        log_event(event="Inference", text=text, prediction=prediction, confidence=confidence, source="primary")

        # Update shared state
        state.update({"prediction": prediction, "confidence": confidence, "source": "primary"})
        return state
