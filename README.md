
---

# Self-Healing Text Classification System

A confidence-aware sentiment classification pipeline built using a **LoRA fine-tuned DistilBERT model** and a **LangGraph-based DAG workflow**.
The system automatically detects low-confidence predictions and corrects them using human-in-the-loop clarification or an optional backup zero-shot classifier.

---

## Overview

This project implements a reliable and extensible sentiment classification system where decisions are made based on **model confidence**, not blind automation.

The workflow ensures:

* High-confidence predictions are accepted immediately
* Low-confidence predictions trigger fallback
* The fallback may involve:

  * User clarification
  * Backup model consultation
* All events are logged for auditability

This design makes the system aligned with safety-first classification workflows.

---

## System Architecture

### DAG Workflow

```
User Input
     │
     ▼
InferenceNode (LoRA model)
     │
     ▼
ConfidenceCheckNode
     ├── High Confidence → FinalizeNode
     └── Low Confidence  → FallbackNode → FinalizeNode
```

### Node Responsibilities

| Node                    | Role                                                         |
| ----------------------- | ------------------------------------------------------------ |
| **InferenceNode**       | Generates prediction & confidence using the fine-tuned model |
| **ConfidenceCheckNode** | Routes the state based on confidence threshold               |
| **FallbackNode**        | Consults backup model or asks the user                       |
| **FinalizeNode**        | Produces final label and logs the outcome                    |

---

## DAG Diagram

<p align="center">
  <img src="dag_diagram.png" width="700" />
</p>

---

## Features

* LoRA-tuned DistilBERT sentiment classifier
* Modular LangGraph DAG workflow
* Confidence-based routing
* Human-in-the-loop fallback mechanism
* Optional zero-shot backup model (BART-MNLI)
* Structured logging (CSV + JSONL)
* Visualization utilities (confidence histogram, curve, fallback stats)
* Clean, color-coded CLI interface

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

## Installation

### Create environment

```
python -m venv myenv
myenv\Scripts\activate
```

### Install dependencies

```
pip install -r requirements.txt
```

---

## Fine-Tuning (LoRA)

Dataset format:

```
text,label
"I loved this movie!",1
"It was boring.",0
```

### Train on full dataset

```
python scripts/train.py --csv data/train.csv
```

### Fast training (sample)

```
python scripts/train.py --csv data/train.csv --sample_size 2000
```

Model is saved to:

```
models/lora_finetuned/
```

---

## Running the Self-Healing CLI

### Standard Mode

```
python scripts/run_cli.py --model_path models/lora_finetuned
```

### With backup model

```
python scripts/run_cli.py --model_path models/lora_finetuned --use_backup
```

### Example Output

```
Input: The movie was painfully slow and boring.

[InferenceNode] Predicted: Positive | Confidence: 54%
[ConfidenceCheckNode] Confidence too low → triggering fallback...
[FallbackNode] Could you clarify? Was this a negative review?

User: Yes, it was definitely negative.

Final Label: Negative (Corrected via user clarification)
```

---

## Log Analysis

Generate analytics:

```
python scripts/analyze_logs.py
```

Outputs saved in `logs/`:

* confidence_histogram.png
* confidence_curve.png
* fallback_stats.png

---

## Evaluation Mapping

| Requirement               | Status      |
| ------------------------- | ----------- |
| Transformer fine-tuning   | Completed   |
| Confidence-based fallback | Implemented |
| Interactive CLI           | Implemented |
| LangGraph DAG             | Complete    |
| Structured logs           | CSV + JSONL |
| Documentation             | Completed   |
| Demo video                | To be added |

---

## Author

**Gaurav Kumar**
GitHub: [https://github.com/Gaurav9693089415](https://github.com/Gaurav9693089415)

---
