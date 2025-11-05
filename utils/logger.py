# =============================================================================
# logger.py
# ---------
# Structured logging utility for the Self-Healing Text Classification System.
#
# Supports:
# - CSV logs (easy to inspect or use in Pandas/Excel)
# - JSONL logs (industry-standard for structured event tracing)
#
# Each event records:
# timestamp, event type, text, prediction, confidence, source, and notes.
# =============================================================================

import os
import csv
import json
from datetime import datetime
from typing import Optional

from utils.config import LOG_DIR, LOG_CSV, LOG_JSONL, LOG_TEXT_MAX_LENGTH

# Ensure logs directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# CSV Header definition
CSV_HEADER = ["timestamp", "event", "text", "prediction", "confidence", "source", "note"]


def now() -> str:
    """Return current timestamp in ISO format."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def truncate_text(text: str, max_length: int = LOG_TEXT_MAX_LENGTH) -> str:
    """
    Truncate excessive text to prevent log bloat.

    Args:
        text: Input text
        max_length: Max allowed text length
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def log_event(
    event: str,
    text: str,
    prediction: str = "",
    confidence: Optional[float] = None,
    source: str = "",
    note: str = "",
) -> None:
    """
    Log a classification event in both CSV and JSONL formats.

    Args:
        event: Type of event (Inference, FallbackUser, etc.)
        text: Input text
        prediction: Output label
        confidence: Model confidence
        source: Origin ('primary', 'corrected', etc.)
        note: Extra metadata
    """

    truncated_text = truncate_text(text)

    formatted_confidence = float(f"{confidence:.4f}") if confidence is not None else None

    row = {
        "timestamp": now(),
        "event": event,
        "text": truncated_text,
        "prediction": prediction,
        "confidence": formatted_confidence,
        "source": source,
        "note": note,
    }

    try:
        # ---- Write CSV ----
        write_header = not os.path.exists(LOG_CSV)
        with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        # ---- Write JSONL ----
        with open(LOG_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    except IOError as e:
        print("[LOGGER] ✗ Failed to write log:", e)
    except Exception as e:
        print("[LOGGER] ✗ Unexpected logging error:", e)


def clear_logs() -> None:
    """Delete all log files."""
    try:
        if os.path.exists(LOG_CSV):
            os.remove(LOG_CSV)
            print(f"[LOGGER] Cleared {LOG_CSV}")

        if os.path.exists(LOG_JSONL):
            os.remove(LOG_JSONL)
            print(f"[LOGGER] Cleared {LOG_JSONL}")

    except Exception as e:
        print("[LOGGER] ✗ Failed to clear logs:", e)


def log_system_event(message: str, level: str = "INFO") -> None:
    """
    Log system-level events (startup, failures, etc.)
    """
    log_event(
        event="System",
        text=message,
        prediction="N/A",
        confidence=None,
        source="system",
        note=f"level={level}",
    )


# ==================== TEST MODE ====================
if __name__ == "__main__":
    print("Testing logging system...\n")

    clear_logs()

    log_event(
        event="Inference",
        text="This is a test sentence.",
        prediction="positive",
        confidence=0.8542,
        source="primary",
    )

    log_event(
        event="ConfidenceCheck",
        text="Low confidence example",
        prediction="negative",
        confidence=0.4231,
        source="primary",
        note="LOW_CONFIDENCE",
    )

    log_event(
        event="FallbackUser",
        text="User clarification example",
        prediction="positive",
        confidence=0.4500,
        source="corrected",
        note="user_input=yes",
    )

    log_system_event("System initialized", level="INFO")

    print("\n✓ Logs written to:")
    print("  -", LOG_CSV)
    print("  -", LOG_JSONL)
