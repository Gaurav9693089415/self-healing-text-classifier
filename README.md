


---

# Self-Healing Text Classification System

A confidence-aware sentiment classification pipeline built using a **LoRA fine-tuned DistilBERT model** and a **LangGraph-based DAG workflow**.
The system prevents incorrect predictions by intelligently handling low-confidence outputs through:

* Backup zero-shot model escalation
* User clarification
* Structured logging and final decision routing

---

## Demo Output (Screenshot Example)

Below is a real sample of the CLI output showing inference, fallback, and final decisions.

<p align="left">
  <img src="dag_diagram.png" alt="Self-Healing DAG" width="500"/>
</p>

---

## Features

* LoRA fine-tuned DistilBERT sentiment model
* LangGraph DAG with deterministic, confidence-based routing
* Human-in-the-loop fallback mechanism
* Optional backup zero-shot classifier (BART-MNLI)
* Interactive CLI with clean colored output
* Structured CSV + JSONL logs
* Log analysis with histograms, trends, fallback stats
* Code formatted with `black` and linted with `flake8`

---

## System Architecture

### Workflow Overview

The self-healing classifier is modeled as a **LangGraph Directed Acyclic Graph (DAG)**:

```
InferenceNode → ConfidenceCheckNode → (FallbackNode) → FinalizeNode
```

### Why LangGraph?

* Enforces modular, transparent data flow
* Guarantees deterministic routing based on confidence thresholds
* Makes human-in-the-loop fallback easy to integrate
* Helps structure complex classification pipelines into clean nodes

---

## Human-in-the-Loop Logic

If model confidence is below the threshold:

* The system asks the user to clarify the sentiment
* If the user is uncertain, the model retains the original prediction but marks it accordingly
* If a backup model is enabled, it is used for secondary classification

This ensures correctness-first behavior.

---

## DAG Diagram

<p align="center">
  <img src="./dag_diagram.png" alt="DAG Diagram" width="700"/>
</p>

---

## Project Structure

```
self_healing_cls/
├── data/
├── models/
├── nodes/
│   ├── inference.py
│   ├── confidence.py
│   ├── fallback.py
│   └── finalize.py
├── graphs/
│   └── classification_dag.py
├── scripts/
│   ├── train.py
│   ├── run_cli.py
│   ├── analyze_logs.py
│   └── load_backup.py
├── utils/
│   ├── config.py
│   └── logger.py
├── logs/
├── dag_diagram.png
├── requirements.txt
└── README.md
```

---

## Installation & Setup

### 1. Clone the repository

```
git clone https://github.com/<your-username>/self-healing-text-classifier.git
cd self-healing-text-classifier
```

### 2. Create and activate a virtual environment

```
python -m venv myenv
myenv\Scripts\activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

## Fine-Tuning the Model (LoRA)

Dataset format:

```
text,label
I loved this movie!,1
This was bad.,0
```

### Train on the full dataset

```
python scripts/train.py --csv data/train.csv
```

### Train using a faster sample

```
python scripts/train.py --csv data/train.csv --sample_size 2000
```

Fine-tuned model is saved in:

```
models/lora_finetuned/
```

---

## Running the Self-Healing CLI

### Standard Mode (user clarification fallback)

```
python scripts/run_cli.py --model_path models/lora_finetuned
```

### With backup zero-shot classifier

```
python scripts/run_cli.py --model_path models/lora_finetuned --use_backup
```

### Example CLI Session

```
Input: The movie was painfully slow and boring

[InferenceNode] Predicted: Negative | Confidence: 52%
[ConfidenceCheckNode] Confidence too low → triggering fallback...
[FallbackNode] Could you clarify? Was this a positive review?

User: Yes, it was positive.

Final Label: Positive (Corrected via user clarification)
```

---

## Log Analysis

Generate charts:

```
python scripts/analyze_logs.py
```

Saved to `/logs/`:

* confidence_histogram.png
* confidence_curve.png
* fallback_stats.png

---

## Evaluation Mapping

| Requirement                     | Status      |
| ------------------------------- | ----------- |
| Fine-tuned transformer model    | Completed   |
| Self-healing fallback mechanism | Implemented |
| Interactive CLI                 | Implemented |
| LangGraph DAG workflow          | Implemented |
| Structured logging              | CSV + JSONL |
| README documentation            | Completed   |
| Demo video                      | To be added |

---

## Summary

This project demonstrates a structured **self-healing text classification system** using:

* LoRA transformer fine-tuning
* Confidence-based routing
* Human-in-the-loop fallback logic
* Modular LangGraph DAG workflows
* Reproducible logging + analytical visualization

Designed for practical, reliable sentiment classification under real-world uncertainty.

---

## Author

**Gaurav Kumar**
GitHub: [https://github.com/Gaurav9693089415](https://github.com/Gaurav9693089415)

---


