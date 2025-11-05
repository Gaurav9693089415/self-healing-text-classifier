
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

## **Why the Screenshot Output Differs from the Assignment Example**

The assignment includes a sample interaction such as:

```

Input: The movie was painfully slow and boring.
[InferenceNode] Predicted label: Positive | Confidence: 54%
[ConfidenceCheckNode] Confidence too low. Triggering fallback...
[FallbackNode] Could you clarify your intent? Was this a negative review?
User: Yes, it was definitely negative.
Final Label: Negative (Corrected via user clarification)

```


In this project(in my case ,for screenshot), the fine-tuned DistilBERT model performs significantly better on the IMDB dataset and often produces **very high confidence scores (85–99%)**, even for moderately complex sentences. This is expected because:

- IMDB sentiment classification is a relatively simple dataset  
- DistilBERT adapts well during LoRA fine-tuning  
- The model learns sentiment cues very effectively  
- Clear positive/negative reviews are classified with high certainty  

As a result:

- High-confidence predictions may skip fallback (correct behavior)  
- Ambiguous sentences still trigger fallback (as shown in the screenshot)  

This difference **does not indicate any issue** with the system — it simply reflects that the trained model is stronger than the hypothetical example in the assignment.

To intentionally demonstrate fallback during evaluation, you may either:

- Temporarily increase the threshold (e.g., 0.70 → 0.99), or  
- Provide more ambiguous input sentences.

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
  <img src="dag_diagram.png" width="700" height="400" />
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

Input: I feel mixed about the movie, not sure how I feel.

[InferenceNode] Predicted: Negative | Confidence: 52%
[ConfidenceCheckNode] Confidence too low → triggering fallback...
[BackupModel] Prediction: Negative | Confidence: 77%

Final Label: Negative (Final label chosen via backup model)
```

---

## **CLI Flow Examples(Actual CLI Output (Experimental Results))**

### High confidence

```

Input: This movie was fantastic.

[InferenceNode] Predicted: Positive | Confidence: 98%

Final Label: Positive (High-confidence model prediction)
```

### Low confidence → user correction

```

Input: The movie was okay, not too bad but not great either

[InferenceNode] Predicted: Positive | Confidence: 66%
[ConfidenceCheckNode] Confidence too low → triggering fallback...
[FallbackNode] Could you clarify your intent? Was this a negative review?

User: yes,it was definitely negative

Final Label: Negative (Corrected via user clarification)
```

### User uncertain

```

User: Not sure
Final Label: Positive (Model prediction retained — user unsure)

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

### **Charts generated:**

* confidence_histogram.png  
* confidence_curve.png  
* fallback_stats.png  

Saved in the `logs/` directory.

### **Chart Descriptions:**

 1. confidence_histogram.png
 Shows the distribution of model confidence scores across all predictions.
 This helps evaluate whether the classifier is generally confident, uncertain, or overly biased toward certain score ranges.


  confidence_curve.png
  Plots confidence values in chronological order based on the sequence of inputs.
  Useful for identifying performance trends such as fluctuations in confidence, stability across sessions, or points where fallbacks were triggered.


  fallback_stats.png
  A comparison of how many predictions were finalized directly vs. how many required fallback.
  This chart gives a clear picture of how frequently the self-healing mechanism activates and whether the confidence threshold is tuned appropriately.
---

## Evaluation Mapping 

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
GitHub: https://github.com/Gaurav9693089415

---
