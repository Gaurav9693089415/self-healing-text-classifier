"""
finalize.py
-----------
Responsible for printing the final classification result.

Handles:
- User corrections
- Backup model results
- High-confidence predictions
- Uncertain user responses
"""

from colorama import Fore, Style
from utils.logger import log_event
from utils.config import CONFIDENCE_THRESHOLD


class FinalizeNode:
    """Final step — produce human-readable output and log the decision."""

    def __call__(self, state: dict) -> dict:

        text = state["text"]
        prediction = state["prediction"]
        confidence = state["confidence"]
        source = state["source"]

        # Select appropriate message
        if source == "corrected":
            note = "(Corrected via user clarification)"
        elif source == "backup":
            note = "(Final label chosen via backup model)"
        elif source == "uncertain":
            note = "(Model prediction retained — user unsure)"
        elif source == "primary" and confidence >= CONFIDENCE_THRESHOLD:
            note = "(High-confidence model prediction)"
        else:
            note = "(Model prediction retained)"

        # Choose color
        color = Fore.GREEN if prediction == "positive" else Fore.RED

        print(
            f"\n{Fore.CYAN}Final Label:{Style.RESET_ALL} " f"{color}{prediction.capitalize()}{Style.RESET_ALL} {note}\n"
        )

        # Write logs
        log_event(
            event="FinalDecision", text=text, prediction=prediction, confidence=confidence, source=source, note=note
        )

        return state
