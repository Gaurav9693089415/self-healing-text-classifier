
# Self-Healing Text Classification System

A confidence-aware sentiment classification pipeline built using a LoRA fine-tuned DistilBERT model and a LangGraph-based DAG.  
The system automatically detects low-confidence predictions and “self-heals” by:

• Escalating to a backup zero-shot classifier  
• Asking the user for clarification  
• Logging and finalizing the corrected label  

---

## Features

• LoRA-fine-tuned DistilBERT classifier  
• Self-healing DAG implemented using LangGraph  
• Confidence-based routing and decision-making  
• User clarification fallback and optional backup model  
• Clean, interactive CLI with colored output  
• Structured logs (CSV + JSONL)  
• Log analysis utilities: confidence histograms, curves, fallback statistics  

---

## System Architecture

### How LangGraph DAG Works in This Project

The entire classification pipeline is implemented as a LangGraph workflow, where every processing step is modeled as a node:

• **InferenceNode** – runs the trained model  
• **ConfidenceCheckNode** – compares confidence to a threshold  
• **FallbackNode** – performs user clarification or backup model inference  
• **FinalizeNode** – produces and logs the final label  

Using LangGraph ensures:

1. Clear and deterministic routing based on confidence  
2. Modularity (each step is isolated, testable, and extensible)  
3. Traceability (events logged at each stage)  
4. Correctness-first design: low-confidence predictions never pass unchecked  

This makes the system robust, predictable, and aligned with human-in-the-loop AI principles.

### Human-in-the-Loop Rationale

If model confidence is low, relying on automation is unsafe.  
The system therefore asks the user:

“Was this a positive review?”

This prevents incorrect outputs, strengthens reliability, and demonstrates responsible fallback design.  
If clarification is still uncertain, the model’s original prediction is retained but marked as “user unsure”.

---

## DAG Diagram

<img src="./dag_diagram.png" width="480" height="350" />

---

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

```

python -m venv myenv
myenv\Scripts\activate

```

### 2. Install Dependencies

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

### Train on a smaller sample

```

python scripts/train.py --csv data/train.csv --sample_size 2000

```

Model output directory:

```

models/lora_finetuned/

```

---

## Running the Self-Healing CLI

### Standard Mode (User Clarification Fallback)

```

python scripts/run_cli.py --model_path models/lora_finetuned

```

### With Backup Zero-Shot Model

```

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

Run:

```

python scripts/analyze_logs.py

```

Generates:

• confidence_histogram.png  
• confidence_curve.png  
• fallback_stats.png  

Saved inside `logs/`.

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

(Add Google Drive / YouTube link here.)

---

## Summary

This project implements a robust self-healing sentiment classification pipeline using:

• LoRA-based transformer fine-tuning  
• Confidence-aware inference and routing  
• Human-in-the-loop fallback strategy  
• Structured logs and analysis  
• Clean, modular LangGraph DAG design  


```

