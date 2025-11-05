

---

# **Self-Healing Text Classification System**

A production-ready sentiment classification pipeline built using a **LoRA fine-tuned DistilBERT model** and a **LangGraph-based Directed Acyclic Graph (DAG)**.
The system intelligently handles low-confidence predictions through human-in-the-loop clarification and an optional backup zero-shot classifier, ensuring correctness over blind automation.

---

## **Demo Output (Screenshot)**

The following screenshot demonstrates the actual CLI execution of the Self-Healing Classification System, showcasing the complete workflow including inference, confidence checking, fallback, and final decision.

<p align="left">
  <img src="demo_output.png" alt="Self-Healing Classification CLI Output" width="700" height="500"/>
</p>

---
---

## **Note on Output Differences ( Assignment Example Vs. Screenshot)**

The assignment example:

```
Input: The movie was painfully slow and boring.
[InferenceNode] Predicted label: Positive | Confidence: 54%
[ConfidenceCheckNode] Confidence too low. Triggering fallback...
[FallbackNode] Could you clarify your intent? Was this a negative review?
User: Yes, it was definitely negative.
Final Label: Negative (Corrected via user clarification)
```

is an illustrative sample to demonstrate fallback behavior.

In this implementation(in my case ), the **fine-tuned DistilBERT model achieves significantly higher confidence** (often 85–99%) on IMDB sentiment classification.
This occurs because:

* IMDB sentiment is a well-understood and relatively simple dataset
* DistilBERT adapts effectively when fine-tuned with LoRA
* The model learns strong signal patterns from the dataset

As a result, the system may frequently produce high-confidence predictions that bypass fallback.


---
## **Overview**

Modern text classifiers may produce uncertain predictions when input text is ambiguous.
This project implements a **self-healing classification workflow** that:

* Performs sentiment classification using a fine-tuned transformer model
* Evaluates confidence for every prediction
* Triggers fallback logic when confidence is below a threshold
* Uses user clarification or a backup model to improve reliability
* Logs all events for transparency and later analysis

The architecture prioritizes **safety and correctness**, aligning with practical human-in-the-loop AI requirements.

---

## **Why LangGraph**

LangGraph provides a modular, node-based workflow where every decision step is explicit and testable.
Using LangGraph ensures:

* Deterministic routing based on confidence levels
* Clean separation of responsibilities among nodes
* Better debugging, observability, and traceability
* Easy extension (e.g., new fallback strategies)

This makes LangGraph well-suited for classification systems requiring controlled logic and recovery paths.

---

## **Human-in-the-Loop Rationale**

Incorrect predictions can be risky when confidence is low.
To prevent misclassification, the system asks the user:

“Was this a positive review?”

If the user expresses uncertainty, the system retains the original prediction but marks it appropriately.
This makes the final decision more reliable and transparent.

---

## **System Architecture**

```
User Input (CLI)
     │
     ▼
InferenceNode (LoRA DistilBERT)
     │
     ▼
ConfidenceCheckNode (Threshold: 70%)
     ├── High Confidence → FinalizeNode
     └── Low Confidence  → FallbackNode → FinalizeNode
```

---

## **DAG Diagram**

<p align="left">
  <img src="dag_diagram.png" width="700" height="300" />
</p>

---

## **Project Structure**

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
├── demo_output.png
├── requirements.txt
└── README.md
```

---

## **Installation**

### Create virtual environment

```
python -m venv myenv
myenv\Scripts\activate
```

### Install dependencies

```
pip install -r requirements.txt
```

---

## **Fine-Tuning the Model (LoRA)**

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

### Fast experiment (sample only)

```
python scripts/train.py --csv data/train.csv --sample_size 2000
```

Model artifacts are saved in:

```
models/lora_finetuned/
```

---

## **Running the Self-Healing CLI**

### Standard mode (user fallback)

```
python scripts/run_cli.py --model_path models/lora_finetuned
```

### With backup zero-shot model

```
python scripts/run_cli.py --model_path models/lora_finetuned --use_backup
```

---

## **CLI Flow Examples**

### High confidence

```
Input: This movie was fantastic.

[InferenceNode] Predicted: Positive | Confidence: 94%
Final Label: Positive (High-confidence prediction)
```

### Low confidence → user correction

```
Input: The movie was painfully slow and boring.

[InferenceNode] Predicted: Positive | Confidence: 54%
[ConfidenceCheckNode] Confidence too low → triggering fallback...
[FallbackNode] Was this a negative review?

User: Yes
Final Label: Negative (Corrected via user clarification)
```

### User uncertain

```
User: Not sure
Final Label: Negative (Model prediction retained — user unsure)
```

### With backup model

```
[BackupModel] Prediction: Negative | Confidence: 82%
```

---

## **Log Analysis**

Generate analytics:

```
python scripts/analyze_logs.py
```

Charts generated:

* confidence_histogram.png
* confidence_curve.png
* fallback_stats.png

Saved in the `logs/` directory.

---

## **Evaluation Mapping (ATG Requirements)**

| Requirement                   | Status      |
| ----------------------------- | ----------- |
| Fine-tuned transformer model  | Completed   |
| Confidence-based fallback     | Implemented |
| Human-in-the-loop interaction | Implemented |
| LangGraph DAG workflow        | Completed   |
| Interactive CLI               | Completed   |
| Structured logs               | CSV + JSONL |
| Backup model (optional)       | Implemented |
| Visualization utilities       | Implemented |
| Documentation                 | Completed   |

---

## **Author**

**Gaurav Kumar**
GitHub: [https://github.com/Gaurav9693089415](https://github.com/Gaurav9693089415)

---

