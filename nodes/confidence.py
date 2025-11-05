"""
confidence.py
--------------
Checks whether the model prediction is confident enough.
"""

from colorama import Fore, Style
from utils.config import CONFIDENCE_THRESHOLD
from utils.logger import log_event


class ConfidenceCheckNode:
    """Confidence validation node."""

    def __init__(self, threshold=CONFIDENCE_THRESHOLD):
        self.threshold = threshold

    def __call__(self, state: dict) -> dict:
        confidence = state["confidence"]
        text = state["text"]
        prediction = state["prediction"]

        if confidence < self.threshold:
            print(
                Fore.YELLOW + "[ConfidenceCheckNode]" + Style.RESET_ALL + " Confidence too low → triggering fallback..."
            )

            log_event(
                event="ConfidenceCheck",
                text=text,
                prediction=prediction,
                confidence=confidence,
                source="primary",
                note="LOW_CONFIDENCE",
            )

        return state
