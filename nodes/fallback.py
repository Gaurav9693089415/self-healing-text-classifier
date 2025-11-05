"""
fallback.py
------------
Handles fallback logic:
- Backup model prediction (optional)
- User clarification
- Handles uncertainty ("not sure"), negation, strong yes/no,
  sentiment keywords, and natural phrasing.
"""

from colorama import Fore, Style
from utils.config import LABELS, CONFIDENCE_THRESHOLD
from utils.logger import log_event


class FallbackNode:
    """Fallback logic combining backup classifier and user clarification."""

    def __init__(self, use_backup=False, backup_classifier=None, threshold=CONFIDENCE_THRESHOLD):

        self.use_backup = use_backup
        self.backup_classifier = backup_classifier
        self.threshold = threshold

        # Uncertain FIRST
        self.uncertain_markers = [
            "not sure",
            "dont know",
            "don't know",
            "idk",
            "maybe",
            "perhaps",
            "unsure",
            "no idea",
            "confused",
            "not certain",
            "cant tell",
            "can't tell",
        ]

        # Strong YES markers → accept opposite label
        self.yes_markers = [
            "yes",
            "yeah",
            "yep",
            "sure",
            "absolutely",
            "definitely",
            "for sure",
            "of course",
            "certainly",
            "surely",
        ]

        # Strong NO markers → keep original prediction
        self.no_markers = [
            "no",
            "nope",
            "nah",
            "not really",
            "not at all",
            "definitely not",
            "absolutely not",
            "never",
            "no chance",
            "no way",
            "unlikely",
        ]

    # ---------------------- BACKUP MODEL ---------------------- #
    def _backup_stage(self, text: str):
        out = self.backup_classifier(text, candidate_labels=LABELS, multi_label=False)
        return out["labels"][0], float(out["scores"][0])

    # ---------------------- USER CLARIFICATION ---------------------- #
    def _user_stage(self, text: str, predicted: str, confidence: float):
        opposite = "negative" if predicted == "positive" else "positive"

        print(
            f"{Fore.BLUE}[FallbackNode]{Style.RESET_ALL} "
            f"Could you clarify your intent? Was this a {opposite} review?"
        )

        user_raw = input("\nUser: ").strip()
        user = user_raw.lower()

        # 1. Uncertainty FIRST
        if any(mark in user for mark in self.uncertain_markers):
            final, status = predicted, "uncertain"

        # 2. Negation patterns
        elif "not negative" in user:
            final, status = "positive", "corrected"
        elif "not positive" in user:
            final, status = "negative", "corrected"

        # 3. Explicit sentiment keywords
        elif "negative" in user:
            final, status = "negative", "corrected"
        elif "positive" in user:
            final, status = "positive", "corrected"

        # 4. YES markers → choose opposite
        elif any(x in user for x in self.yes_markers):
            final, status = opposite, "corrected"

        # 5. NO markers → keep predicted label
        elif any(mark in user for mark in self.no_markers):
            final, status = predicted, "corrected"

        # 6. Unknown → treat as uncertain
        else:
            final, status = predicted, "uncertain"

        # Log event
        log_event(
            event="FallbackUser",
            text=text,
            prediction=final,
            confidence=confidence,
            source=status,
            note=f"user_input={user_raw}",
        )

        return final, confidence, status

    # ---------------------- MAIN CALL ---------------------- #
    def __call__(self, state: dict) -> dict:

        text = state["text"]
        predicted = state["prediction"]
        base_conf = state["confidence"]

        # Stage 1 — Backup model
        if self.use_backup and self.backup_classifier is not None:

            backup_label, backup_conf = self._backup_stage(text)

            print(
                f"{Fore.MAGENTA}[BackupModel]{Style.RESET_ALL} "
                f"Prediction: {backup_label.capitalize()} | Confidence: {backup_conf:.0%}"
            )

            log_event(
                event="BackupModel",
                text=text,
                prediction=backup_label,
                confidence=backup_conf,
                source="backup",
                note="backup_stage",
            )

            # If backup model is confident → accept it
            if backup_conf >= self.threshold:
                state.update({"prediction": backup_label, "confidence": backup_conf, "source": "backup"})
                return state

        # Stage 2 — Human clarification
        final_label, final_conf, status = self._user_stage(text, predicted, base_conf)

        state.update({"prediction": final_label, "confidence": final_conf, "source": status})

        return state
