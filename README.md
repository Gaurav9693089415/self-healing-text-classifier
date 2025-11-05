

---

# **Self-Healing Text Classification System**

A confidence-aware sentiment classification pipeline built using a **LoRA fine-tuned DistilBERT model** and a **LangGraph-based DAG**.
The system identifies low-confidence predictions and automatically “self-heals” by:

• Escalating to a backup zero-shot classifier
• Requesting clarification from the user
• Logging and finalizing the corrected label

---

# **Features**

• LoRA-fine-tuned DistilBERT classifier
• LangGraph-based self-healing DAG
• Confidence-based decision routing
• Human-in-the-loop fallback mechanism
• Optional backup model (zero-shot BART-MNLI)
• Color-coded interactive CLI
• Structured logs in CSV and JSONL
• Analytics utilities for log visualization

---

# **System Architecture**

## **LangGraph DAG in This Project**

The workflow is modeled as a **deterministic DAG**, where each processing step is represented as a node:

• **InferenceNode** – runs the fine-tuned classifier
• **ConfidenceCheckNode** – evaluates prediction confidence
• **FallbackNode** – performs user clarification or backup inference
• **FinalizeNode** – produces and logs the final decision

### Why LangGraph?

• Clear and testable modular components
• Explicit control flow based on confidence thresholds
• Built-in structure for human-in-the-loop workflows
• Ensures correctness-first behavior for production scenarios

## **Human-in-the-Loop Justification**

When the model confidence is low, automated classification is avoided.
Instead, the system asks the user a clarifying question such as:

“Was this a positive review?”

This guarantees:

• Reduced risk of incorrect outputs
• Transparent recovery strategy
• Responsible and interpretable AI behavior

---

# **DAG Diagram**

<img src="./dag_diagram.png" width="480" height="350" />

---

# **Project Structure**

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

# **Installation**

## 1. Create Virtual Environment

```
python -m venv myenv
myenv\Scripts\activate
```

## 2. Install Dependencies

```
pip install -r requirements.txt
```

---

# **Fine-Tuning the Model (LoRA)**

Dataset format:

```
text,label
I loved this movie!,1
This was bad.,0
```

## Train on the full dataset

```
python scripts/train.py --csv data/train.csv
```

## Train on a smaller sample

```
python scripts/train.py --csv data/train.csv --sample_size 2000
```

Model is saved to:

```
models/lora_finetuned/
```

---

# **Running the Self-Healing CLI**

## Standard Mode (User Clarification Fallback)

```
python scripts/run_cli.py --model_path models/lora_finetuned
```

## With Backup Zero-Shot Model

```
python scripts/run_cli.py --model_path models/lora_finetuned --use_backup
```

## Example CLI Session

```
Input: The movie was painfully slow and boring.

[InferenceNode] Predicted: Negative | Confidence: 98%
Final Label: Negative (High-confidence model prediction)


Input: I feel mixed about the movie, not sure how I feel.

[InferenceNode] Predicted: Negative | Confidence: 52%
[ConfidenceCheckNode] Confidence too low → triggering fallback...
[FallbackNode] Could you clarify your intent? Was this a positive review?

User: yes, it was definitely positive.

Final Label: Positive (Corrected via user clarification)

Input: exit
Exiting... Goodbye!
```

---

# **Log Analysis**

Run:

```
python scripts/analyze_logs.py
```

Generates:

• confidence_histogram.png
• confidence_curve.png
• fallback_stats.png

All saved inside the `logs/` directory.

---

# **Evaluation Mapping**

| Requirement                         | Status            |
| ----------------------------------- | ----------------- |
| Fine-tuned transformer model        | Completed         |
| Confidence-based fallback mechanism | Implemented       |
| Interactive CLI                     | Implemented       |
| LangGraph-based DAG                 | Fully implemented |
| Structured logging                  | CSV + JSONL       |
| Documentation                       | Complete          |
| Demo video                          | To be added       |

---

# **Demo Video**

Add Google Drive or YouTube link here after recording.

---

# **Summary**

This system implements a complete, production-style self-healing sentiment classifier using:

• Transformer fine-tuning with LoRA
• Confidence-based decision routing
• Human-in-the-loop fallback strategies
• Structured logging and analytics
• Clean and modular LangGraph DAG design

The result is a robust, interpretable classification pipeline suitable for real-world applications.

```

