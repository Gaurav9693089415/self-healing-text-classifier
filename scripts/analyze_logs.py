"""
analyze_logs.py
---------------
Generates visualizations and statistics from classification logs.

This script reads logs recorded during inference and fallback stages
and produces:

1. Confidence Histogram
2. Confidence Curve (over time)
3. Fallback Frequency Chart

Responsibilities:
- Parse CSV log file
- Compute useful statistics
- Generate publication-ready matplotlib plots
"""

import os
import csv
from collections import Counter
import matplotlib.pyplot as plt
import statistics

from utils.config import LOG_CSV, LOG_DIR


# ---------------------- READ LOGS ---------------------- #


def read_logs():
    """Load CSV logs into a list of dictionaries."""
    if not os.path.exists(LOG_CSV):
        print(f"[WARN] No log file found at: {LOG_CSV}")
        return []

    try:
        with open(LOG_CSV, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception as err:
        print(f"[ERROR] Failed to parse CSV logs: {err}")
        return []


# ---------------------- COMPUTE STATS ---------------------- #


def compute_stats(rows):
    """
    Compute summary statistics:
    - Mean / median confidence
    - Event type frequencies
    """

    confidences = []
    events = Counter()

    for r in rows:
        events[r["event"]] += 1

        try:
            conf = float(r["confidence"])
            confidences.append(conf)
        except (ValueError, TypeError):
            continue

    stats = {
        "total_events": len(rows),
        "event_breakdown": dict(events),
        "mean_conf": statistics.mean(confidences) if confidences else None,
        "median_conf": statistics.median(confidences) if confidences else None,
    }

    return stats, confidences, events


# ---------------------- PLOT: CONFIDENCE HISTOGRAM ---------------------- #


def plot_confidence_hist(confidences):
    """Save histogram of model confidence scores."""
    if not confidences:
        print("[WARN] No confidences to plot.")
        return

    path = os.path.join(LOG_DIR, "confidence_histogram.png")

    plt.figure(figsize=(7, 4))
    plt.hist(confidences, bins=20, edgecolor="black")
    plt.title("Confidence Histogram")
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    print(f"[INFO] Saved histogram → {path}")


# ---------------------- PLOT: CONFIDENCE CURVE ---------------------- #


def plot_confidence_curve(confidences):
    """Plot confidence values in chronological order."""
    if not confidences:
        print("[WARN] No confidences to plot.")
        return

    path = os.path.join(LOG_DIR, "confidence_curve.png")

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(confidences) + 1), confidences, marker="o")
    plt.title("Confidence Curve")
    plt.xlabel("Prediction #")
    plt.ylabel("Confidence")
    plt.ylim(0, 1)
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    print(f"[INFO] Saved curve → {path}")


# ---------------------- PLOT: FALLBACK STATS ---------------------- #


def plot_fallback_stats(events):
    """Plot bar chart of fallback-related events."""
    path = os.path.join(LOG_DIR, "fallback_stats.png")

    normal = events.get("Inference", 0)

    fallback = events.get("ConfidenceCheck", 0) + events.get("FallbackUser", 0) + events.get("BackupModel", 0)

    plt.figure(figsize=(6, 4))
    plt.bar(["Normal Predictions", "Fallback Triggered"], [normal, fallback])
    plt.title("Fallback Frequency")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    print(f"[INFO] Saved fallback stats → {path}")


# ---------------------- MAIN EXECUTION ---------------------- #


if __name__ == "__main__":
    print("[INFO] Reading logs...\n")

    rows = read_logs()
    if not rows:
        exit()

    print("[INFO] Computing statistics...\n")
    stats, confidences, events = compute_stats(rows)

    print("------ Log Statistics ------")
    for key, val in stats.items():
        print(f"{key}: {val}")
    print("----------------------------\n")

    plot_confidence_hist(confidences)
    plot_confidence_curve(confidences)
    plot_fallback_stats(events)

    print("\nAnalysis complete! Check the logs/ folder for generated charts.\n")
