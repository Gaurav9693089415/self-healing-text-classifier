# Self-Healing Text Classification System

A production-ready sentiment classification pipeline built with **LoRA fine-tuned DistilBERT** and **LangGraph workflow orchestration**. The system implements intelligent fallback mechanisms to handle uncertain predictions through human-in-the-loop interaction and optional backup model consultation.


---

##  Project Overview

This system goes beyond traditional classification by implementing a **self-healing architecture** that:

-  Performs high-accuracy sentiment analysis using a fine-tuned transformer model
-  Automatically detects low-confidence predictions and triggers fallback strategies
-  Engages users for clarification when model uncertainty is detected
-  Optionally consults a backup zero-shot classifier (BART-MNLI) for validation
-  Maintains comprehensive structured logs (CSV + JSONL) for analysis

**Key Innovation**: Rather than blindly accepting all predictions, the system prioritizes **correctness over automation** by intelligently seeking human guidance when confidence is low.

---

##  System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    User Input (CLI)                       │
└─────────────────────┬────────────────────────────────────┘
                      │
                      ▼
          ┌───────────────────────┐
          │   InferenceNode       │ ← LoRA Fine-tuned DistilBERT
          │   Primary Prediction  │    + Confidence Score
          └───────────┬───────────┘
                      │
                      ▼
          ┌───────────────────────┐
          │ ConfidenceCheckNode   │
          │ Threshold: 70%        │
          └───────────┬───────────┘
                      │
           ┌──────────┴──────────┐
           │                     │
     High Confidence       Low Confidence
           │                     │
           ▼                     ▼
    ┌─────────────┐      ┌──────────────────┐
    │ FinalizeNode│      │   FallbackNode   │
    │  Accept     │      │ ┌──────────────┐ │
    └─────────────┘      │ │ Backup Model │ │ (Optional)
                         │ └──────────────┘ │
                         │ ┌──────────────┐ │
                         │ │ User Prompt  │ │ (Required)
                         │ └──────────────┘ │
                         └────────┬─────────┘
                                  │
                                  ▼
                           ┌─────────────┐
                           │ FinalizeNode│
                           │  Log Result │
                           └─────────────┘
```

### Workflow Components

1. **InferenceNode**: Loads LoRA-tuned model and generates predictions with confidence scores
2. **ConfidenceCheckNode**: Evaluates confidence against threshold (default: 70%)
3. **FallbackNode**: Handles uncertain predictions through:
   - Optional backup zero-shot model consultation
   - User clarification with intelligent response parsing
4. **FinalizeNode**: Produces final classification with appropriate context labels

---

##  Features

### Core Capabilities
- **LoRA Fine-Tuning**: Efficient parameter-efficient fine-tuning of DistilBERT
- **LangGraph DAG**: Deterministic, confidence-based routing through modular nodes
- **Human-in-the-Loop**: Interactive fallback mechanism for low-confidence predictions
- **Backup Model**: Optional BART-MNLI zero-shot classifier for secondary validation
- **Smart Response Parsing**: Handles uncertainty, negation, sentiment keywords, and natural language

### User Experience
- **Interactive CLI**: Clean, color-coded terminal interface
- **Streaming Output**: Real-time prediction feedback
- **Contextual Labels**: Clear indication of prediction source (primary/corrected/uncertain)

### Logging & Analysis
- **Dual Format Logs**: CSV (analysis-ready) + JSONL (structured events)
- **Visualization Tools**: Confidence histograms, trend curves, fallback statistics
- **Event Tracking**: Complete audit trail from inference to final decision

---

##  Project Structure

```
SELF_HEALING_CLS/
│
├── data/
│   ├── imdb_raw.csv              # Original IMDB dataset
│   └── train.csv                 # Preprocessed training data
│
├── docs/
│   └── classification_dag.py     # DAG visualization
│
├── graphs/
│   └── classification_dag.py     # LangGraph workflow definition
│
├── logs/
│   ├── classification.csv        # Event logs (CSV)
│   ├── classification.jsonl      # Event logs (JSONL)
│   ├── confidence_histogram.png  # Confidence distribution
│   ├── confidence_curve.png      # Confidence over time
│   └── fallback_stats.png        # Fallback frequency chart
│
├── models/
│   └── lora_finetuned/          # Fine-tuned model artifacts
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       ├── tokenizer_config.json
│       └── ...
│
├── nodes/
│   ├── confidence.py             # Confidence threshold checking
│   ├── fallback.py               # Fallback logic (backup + user)
│   ├── finalize.py               # Final output formatting
│   └── inference.py              # Primary model inference
│
├── scripts/
│   ├── analyze_logs.py           # Generate visualization charts
│   ├── load_backup.py            # Backup model loader utility
│   ├── prepare_dataset.py        # Data cleaning & preprocessing
│   ├── run_cli.py                # Main CLI entry point
│   └── train.py                  # LoRA fine-tuning script
│
├── utils/
│   ├── config.py                 # Global configuration constants
│   └── logger.py                 # Structured logging utilities
│
├── .flake8                       # Linting configuration
├── .gitignore                    # Git ignore rules
├── dag_diagram.png               # System architecture diagram
├── README.md                     # This file
└── requirements.txt              # Python dependencies
```

---

##  Installation & Setup

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for training)
- 4GB+ available disk space

### Step 1: Clone the Repository

```bash
git clone https://github.com/Gaurav9693089415/self-healing-text-classifier.git
cd self-healing-text-classifier
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv myenv
myenv\Scripts\activate
```

**Linux/Mac:**
```bash
python -m venv myenv
source myenv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Core Dependencies:**
- `transformers` - HuggingFace transformer models
- `torch` - PyTorch deep learning framework
- `peft` - Parameter-efficient fine-tuning (LoRA)
- `langgraph` - Workflow orchestration
- `datasets` - Dataset management
- `colorama` - Terminal color output
- `matplotlib` - Visualization

---

## 🎓 Fine-Tuning the Model

### Dataset Format

The training script expects CSV format with two columns:

```csv
text,label
"I loved this movie! Best film of the year.",1
"Boring and predictable. Complete waste of time.",0
```

- **text**: Review text (string)
- **label**: Sentiment (0=negative, 1=positive)

### Preprocessing the Dataset

If you have raw IMDB data, first run the preprocessing script:

```bash
python scripts/prepare_dataset.py
```

This will:
- Clean HTML tags (e.g., `<br />`)
- Normalize whitespace
- Remove empty entries
- Save to `data/train.csv`

### Training Commands

**Full Dataset Training (Recommended):**
```bash
python scripts/train.py --csv data/train.csv --epochs 2 --batch_size 16
```

**Quick Test (2000 samples):**
```bash
python scripts/train.py --csv data/train.csv --sample_size 2000 --epochs 1
```

**Custom Configuration:**
```bash
python scripts/train.py \
  --csv data/train.csv \
  --sample_size 5000 \
  --output_dir ./models/custom_model \
  --epochs 3 \
  --batch_size 8
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--csv` | `./data/train.csv` | Path to training CSV |
| `--sample_size` | `None` | Number of samples (None = full dataset) |
| `--output_dir` | `./models/lora_finetuned` | Model save directory |
| `--epochs` | `2` | Training epochs |
| `--batch_size` | `16` | Batch size per device |

### LoRA Configuration

Configured in `utils/config.py`:

```python
LORA_R = 8                                # LoRA rank
LORA_ALPHA = 16                          # LoRA alpha
LORA_DROPOUT = 0.1                       # Dropout rate
LORA_TARGET_MODULES = ["q_lin", "v_lin"] # Target layers
```

### Training Output

The script will save:
- `adapter_model.safetensors` - LoRA weights
- `adapter_config.json` - LoRA configuration
- `tokenizer_config.json` - Tokenizer settings
- Training logs in `models/lora_finetuned/logs/`

Expected training time:
- Full IMDB dataset (~50k samples): ~2-3 hours on GPU
- Sample (2000): ~10-15 minutes on GPU

---

##  Running the Self-Healing CLI

### Basic Usage (User Clarification Only)

```bash
python scripts/run_cli.py --model_path models/lora_finetuned
```

### With Backup Model

```bash
python scripts/run_cli.py --model_path models/lora_finetuned --use_backup
```

### CLI Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model_path` | Path to fine-tuned model | `./models/lora_finetuned` |
| `--use_backup` | Enable backup zero-shot model | `False` |

---

##  CLI Flow Explanations

### Example Session 1: High Confidence (No Fallback)

```
=== Self-Healing Classification CLI ===
Type 'exit' to quit.

Input: This movie was absolutely fantastic! Best film I've seen this year.

[InferenceNode] Predicted: Positive | Confidence: 94%

Final Label: Positive (High-confidence model prediction)
```

**Flow:**
1. User provides input
2. InferenceNode predicts with 94% confidence
3. ConfidenceCheckNode accepts (≥70% threshold)
4. FinalizeNode outputs result
5. No fallback triggered 

---

### Example Session 2: Low Confidence (User Correction)

```
Input: The movie was painfully slow and boring.

[InferenceNode] Predicted: Positive | Confidence: 52%
[ConfidenceCheckNode] Confidence too low → triggering fallback...
[FallbackNode] Could you clarify your intent? Was this a negative review?

User: Yes, it was definitely negative.

Final Label: Negative (Corrected via user clarification)
```

**Flow:**
1. Model predicts "Positive" but only 52% confident
2. ConfidenceCheckNode triggers fallback (<70%)
3. FallbackNode asks for clarification
4. User confirms opposite label
5. System corrects to "Negative" 

---

### Example Session 3: User Uncertainty

```
Input: It had some good moments but overall disappointed.

[InferenceNode] Predicted: Negative | Confidence: 58%
[ConfidenceCheckNode] Confidence too low → triggering fallback...
[FallbackNode] Could you clarify your intent? Was this a positive review?

User: I'm not sure, it's mixed.

Final Label: Negative (Model prediction retained — user unsure)
```

**Flow:**
1. Model predicts "Negative" with 58% confidence
2. Fallback triggered
3. User expresses uncertainty
4. System retains original prediction but marks as uncertain 

---

### Example Session 4: Backup Model Consultation

```
Input: The cinematography was decent but the plot fell flat.

[InferenceNode] Predicted: Positive | Confidence: 55%
[ConfidenceCheckNode] Confidence too low → triggering fallback...
[BackupModel] Prediction: Negative | Confidence: 82%
[FallbackNode] Could you clarify your intent? Was this a negative review?

User: Yes, negative overall.

Final Label: Negative (Corrected via user clarification)
```

**Flow:**
1. Primary model uncertain (55%)
2. Backup zero-shot model consulted (82% for "Negative")
3. User still prompted for final confirmation
4. User agrees with backup model assessment 

---

### User Response Patterns

The FallbackNode intelligently parses various user responses:

| User Input | Interpretation | Action |
|------------|---------------|--------|
| `"yes"`, `"yeah"`, `"definitely"` | Strong affirmative | Accept opposite label |
| `"no"`, `"nope"`, `"not really"` | Strong negative | Keep original prediction |
| `"not sure"`, `"idk"`, `"maybe"` | Uncertainty | Mark as uncertain, keep original |
| `"not negative"` | Negation pattern | Override to positive |
| `"positive"`, `"negative"` | Explicit sentiment | Use specified label |

**Handling Uncertainty:**
The system treats user uncertainty respectfully - if you're genuinely unsure, it won't force a correction but will log the uncertainty for analysis.

---

##  Log Analysis

### Generating Visualizations

```bash
python scripts/analyze_logs.py
```

### Generated Charts

1. **Confidence Histogram** (`logs/confidence_histogram.png`)
   - Distribution of model confidence scores
   - Helps identify threshold tuning opportunities

2. **Confidence Curve** (`logs/confidence_curve.png`)
   - Confidence over time (chronological)
   - Shows model performance trends

3. **Fallback Statistics** (`logs/fallback_stats.png`)
   - Bar chart: Normal predictions vs. Fallback triggers
   - Quantifies system intervention frequency



---

## ⚙️ Configuration

### Adjusting Confidence Threshold

Edit `utils/config.py`:

```python
# Lower threshold = more fallbacks (safer but slower)
# Higher threshold = fewer fallbacks (faster but riskier)
CONFIDENCE_THRESHOLD = 0.70  # Default: 70%
```

**Recommended ranges:**
- **Conservative**: 0.75-0.85 (prioritizes accuracy)
- **Balanced**: 0.65-0.75 (default)
- **Aggressive**: 0.50-0.65 (minimizes user interruption)

### Other Key Settings

```python
# Model Configuration
MODEL_NAME = "distilbert-base-uncased"
BACKUP_MODEL_NAME = "facebook/bart-large-mnli"

# Classification Labels
LABELS = ["negative", "positive"]

# Tokenization
MAX_LENGTH = 256

# Training
TEST_SPLIT = 0.1
DEFAULT_BATCH_SIZE = 16
LORA_R = 8
LORA_ALPHA = 16
```

---

##  Code Quality

### Linting

```bash
flake8 .
```

Configuration in `.flake8`:
- Max line length: 120 characters
- Ignores: E402 (import ordering for path fixes), W503 (line breaks)

### Code Formatting

The codebase follows PEP 8 standards with:
- Descriptive variable names
- Comprehensive docstrings
- Type hints where applicable
- Inline comments for complex logic

---

## 📊 Evaluation Summary

| Requirement | Implementation | Status |
|------------|----------------|--------|
| Fine-tuned transformer model | LoRA-tuned DistilBERT | ✅ Complete |
| Self-healing fallback | Confidence check + user clarification | ✅ Complete |
| Interactive CLI | Color-coded, streaming output | ✅ Complete |
| LangGraph DAG workflow | 4-node pipeline with conditional routing | ✅ Complete |
| Structured logging | CSV + JSONL with event tracking | ✅ Complete |
| Documentation | Comprehensive README with examples | ✅ Complete |
| Backup model (bonus) | BART-MNLI zero-shot classifier | ✅ Complete |
| Log visualization (bonus) | Histograms, curves, stats | ✅ Complete |

---


## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

##  Author

**Gaurav Kumar**

- GitHub: [@Gaurav9693089415](https://github.com/Gaurav9693089415)
- Project Link: [Self-Healing Text Classifier](https://github.com/Gaurav9693089415/self-healing-text-classifier)

---



