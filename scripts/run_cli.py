"""
run_cli.py
-----------
Simple CLI interface for running the Self-Healing Classification workflow.

Handles:
- User text input
- Running the LangGraph DAG
- Displaying predictions, fallback steps, and final output
"""


# ------------------------
# ALL IMPORTS FIRST (flake8 E402)
# ------------------------
import argparse
import os
import sys
import time
import transformers
from colorama import init, Fore, Style

# ------------------------
# NON-IMPORT CODE AFTER IMPORTS
# ------------------------

# Silence HF warnings
transformers.logging.set_verbosity_error()

# Fix path so "utils", "nodes", "graphs" can be imported
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

# Now import project modules (allowed)
from utils.config import BACKUP_MODEL_NAME
from scripts.load_backup import load_backup_model
from graphs.classification_dag import build_classification_graph

# Initialize colorama
init(autoreset=True)


def stream_line(text, delay=0.012):
    """Smooth streaming print for better UX."""
    for ch in text:
        print(ch, end="", flush=True)
        time.sleep(delay)
    print()


def run_cli(model_path, use_backup):

    backup_classifier = load_backup_model(BACKUP_MODEL_NAME) if use_backup else None
    app = build_classification_graph(model_path, use_backup, backup_classifier)

    print(Fore.YELLOW + "=== Self-Healing Classification CLI ===" + Style.RESET_ALL)
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input(Fore.GREEN + "Input:" + Style.RESET_ALL + " ").strip()

        if user_input.lower() in ("exit", "quit"):
            print(Fore.CYAN + "Exiting... Goodbye!" + Style.RESET_ALL)
            break

        print()
        state = {"text": user_input}
        app.invoke(state)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./models/lora_finetuned")
    parser.add_argument("--use_backup", action="store_true")

    args = parser.parse_args()
    run_cli(args.model_path, args.use_backup)
