

---

# Self-Healing Text Classification System

A confidence-aware sentiment classification pipeline built using **LangGraph**, a **LoRA fine-tuned DistilBERT model**, and a structured fallback mechanism to prevent incorrect predictions.

The system automatically detects low-confidence predictions and “self-heals” by either:

* Escalating to a backup zero-shot classifier
* Asking the user for clarification
* Logging and finalizing the corrected result

---

## Features

* LoRA-fine-tuned DistilBERT sentiment classifier
* Self-healing DAG implemented using LangGraph
* Confidence-based routing and decision-making
* Fallback strategies: user clarification, optional backup model (BART-MNLI zero-shot)
* Interactive CLI with clean, color-coded output
* Structured logging (CSV + JSONL)
* Log analysis utilities: histograms, confidence curves, fallback statistics

---

## System Architecture

### DAG Overview

The classification pipeline is implemented as a directed acyclic graph (DAG) with confidence-based routing:

* High-confidence predictions go directly to `FinalizeNode`
* Low-confidence predictions are routed to `FallbackNode`
* Fallback resolves using user confirmation or backup model

### DAG Diagram

<img src="./dag_diagram.png" width="480" height="300" />


## Project Structure

```
self_healing_cls/
│── data/
│── models/
│── nodes/
│   ├── inference.py
│   ├── confidence.py
│   ├── fallback.py
│   └── finalize.py
│── graphs/
│   └── classification_dag.py
│── scripts/
│   ├── train.py
│   ├── run_cli.py
│   ├── analyze_logs.py
│   └── load_backup.py
│── utils/
│   ├── config.py
│   └── logger.py
│── logs/
│── dag_diagram.png
│── README.md
│── requirements.txt
```

---

## Installation

### 1. Create Virtual Environment

```bash
python -m venv myenv
myenv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Fine-Tuning the Model (LoRA)

The dataset should follow the format:

```
text,label
I loved this movie!,1
This was bad.,0
```

### Train on the full dataset

```bash
python scripts/train.py --csv data/train.csv
```

### Train on a smaller sample (faster)

```bash
python scripts/train.py --csv data/train.csv --sample_size 2000
```

The fine-tuned model is saved to:

```
models/lora_finetuned/
```

---

## Running the Self-Healing CLI

### 1. Standard Mode (user clarification only)

```bash
python scripts/run_cli.py --model_path models/lora_finetuned
```

### 2. With Backup Zero-Shot Model

```bash
python scripts/run_cli.py --model_path models/lora_finetuned --use_backup
```

### Example CLI Session

```
Input: The movie was painfully slow and boring.

[InferenceNode] Predicted: Positive | Confidence: 54%
[ConfidenceCheckNode] Confidence too low → triggering fallback...
[FallbackNode] Could you clarify your intent? Was this a negative review?

User: Yes, it was definitely negative.

Final Label: Negative (Corrected via user clarification)
```

---

## Log Analysis

Generate statistical charts from past runs:

```bash
python scripts/analyze_logs.py
```

The following files are saved in `logs/`:

* confidence_histogram.png
* confidence_curve.png
* fallback_stats.png

---

## Evaluation Mapping

| ATG Requirement                     | Status            |
| ----------------------------------- | ----------------- |
| Fine-tuned transformer model        | Completed         |
| Confidence-based fallback mechanism | Implemented       |
| Interactive CLI                     | Implemented       |
| LangGraph-based DAG                 | Fully implemented |
| Structured logging                  | CSV + JSONL       |
| Documentation                       | Complete          |
| Demo video                          | To be added       |

---

## Demo Video

Add the Google Drive or YouTube link here after recording the demonstration.

---

## Summary

This project implements a complete self-healing sentiment classification pipeline using:

* Transformer fine-tuning with LoRA
* Confidence-aware inference
* Human-in-the-loop fallback strategies
* Structured logs and performance analytics
* Clean, modular LangGraph workflow

The system is designed to be robust, interpretable, and suitable for production-grade applications.

---

