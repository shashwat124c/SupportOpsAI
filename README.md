# SupportOpsAI

An instruction-tuned customer support ticket intelligence system built using **QLoRA fine-tuning** of an open-source Large Language Model.

SupportOpsAI takes a customer support ticket containing a subject and body and predicts:

- **Support Queue**
- **Ticket Priority**

The project focuses on building and evaluating a complete parameter-efficient LLM fine-tuning workflow using Hugging Face Transformers, PEFT, TRL, and bitsandbytes.

---

## Overview

The goal of SupportOpsAI is to explore how a small instruction-following LLM can be adapted to a domain-specific customer support classification task using **LoRA/QLoRA**, rather than updating the entire model.

The system follows this pipeline:

```text
Customer Support Ticket
        │
        ▼
Data Cleaning & Deduplication
        │
        ▼
English-only Dataset
        │
        ▼
Stratified Train / Validation / Test Split
        │
        ▼
Instruction Formatting
        │
        ▼
Qwen2.5-1.5B-Instruct
        │
        ▼
4-bit QLoRA Fine-tuning
        │
        ▼
Structured JSON Prediction
        │
        ▼
Evaluation & Error Analysis
```

---

## Key Features

- **Instruction Fine-Tuning:** Adapted an open-source instruction-following LLM for domain-specific classification.
- **Parameter-Efficient Training:** LoRA / QLoRA with 4-bit NF4 quantization and double quantization via `bitsandbytes`.
- **Modern Hugging Face Stack:** Built with `transformers`, `peft`, and `trl`.
- **Rigorous Data Preprocessing:** Duplicate removal prior to a stratified train/validation/test split.
- **Structured Outputs:** Enforces and evaluates valid JSON generation for programmatic consumption.
- **Comprehensive Evaluation:**
  - Baseline vs. fine-tuned model comparison
  - Accuracy and Macro-F1 metrics
  - Joint queue + priority accuracy
  - JSON validity rate
  - Class-level classification reports and confusion matrices
  - Prediction distribution and error analysis

---

## Model

### Base Model
- **`Qwen/Qwen2.5-1.5B-Instruct`**

The base model is adapted using QLoRA while keeping the majority of the original parameters frozen.

### Why QLoRA?
Full fine-tuning updates all model parameters and requires substantial GPU memory. QLoRA instead:
1. Loads the base model using low-bit (4-bit) quantization.
2. Freezes the quantized base model weights.
3. Injects small trainable LoRA adapter matrices into attention projections.
4. Updates only the adapter parameters during training.

For this experiment, **only ~0.28% of the total model parameters were trainable**.

### QLoRA Configuration

| Parameter | Value |
| :--- | :--- |
| **Quantization** | 4-bit |
| **Quantization Type** | NF4 |
| **Double Quantization** | Enabled |
| **Compute dtype** | FP16 |
| **LoRA Rank ($r$)** | 16 |
| **LoRA Alpha ($\alpha$)** | 32 |
| **LoRA Dropout** | 0.05 |
| **Target Modules** | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| **Epochs** | 3 |
| **Learning Rate** | 2e-4 |
| **Effective Batch Size** | 16 |
| **Maximum Sequence Length** | 512 |
| **Gradient Checkpointing** | Enabled |
| **Trainable Parameters** | ~0.28% |

---

## Dataset

The project uses the **`Tobi-Bueck/customer-support-tickets`** dataset, containing synthetic customer support tickets across multiple support categories and priority tiers.

### Preprocessing Pipeline
- Filtered to **English tickets only**
- Concatenated `subject` + `body` as model input
- Missing subjects retained
- Duplicate subject/body pairs removed before splitting
- Stratified train/validation/test split (`random_state=42`)

### Split Summary (23,747 Unique Tickets)

| Split | Count | Percentage |
| :--- | :--- | :--- |
| **Train** | 18,997 | 80% |
| **Validation** | 2,375 | 10% |
| **Test** | 2,375 | 10% |

> *The test set was held out and kept completely isolated from training and validation.*

### Target Classes

#### Support Queue (10 Classes)
- Technical Support
- Product Support
- Customer Service
- IT Support
- Billing and Payments
- Returns and Exchanges
- Service Outages and Maintenance
- Sales and Pre-Sales
- Human Resources
- General Inquiry

#### Priority (3 Classes)
- High
- Medium
- Low

---

## Output Format

The fine-tuned model is instruction-tuned to generate a strict, machine-readable JSON object:

```json
{
  "queue": "IT Support",
  "priority": "medium"
}
```

This ensures downstream services can ingest and route predictions deterministically without heuristic regex parsing.

---

## Evaluation & Results

Both the base zero-shot baseline and the QLoRA-adapted model were evaluated against the identical 2,375-sample test set.

### Evaluation Metrics
- **Queue Accuracy & Macro-F1**
- **Priority Accuracy & Macro-F1**
- **Joint Accuracy** (both queue and priority correct simultaneously)
- **JSON Validity Rate**

### Benchmark Comparison

| Metric | Base Model | QLoRA Fine-Tuned | Delta |
| :--- | :---: | :---: | :---: |
| **Queue Accuracy** | 31.66% | **36.97%** | +5.31% |
| **Queue Macro-F1** | **20.40%** | 19.41% | -0.99% |
| **Priority Accuracy** | 37.81% | **47.54%** | +9.73% |
| **Priority Macro-F1** | 23.75% | **35.27%** | +11.52% |
| **Joint Accuracy** | 15.24% | **21.05%** | +5.81% |
| **JSON Validity** | 95.87% | **100.00%** | +4.13% |

### Key Takeaways
- **Priority Classification:** Achieved the largest leap, improving Accuracy by **+9.73 percentage points** and Macro-F1 by **+11.52 percentage points**.
- **Structured Parsing:** JSON validity reached **100.00%**, eliminating schema and formatting failures.
- **Joint Performance:** Joint accuracy improved by **+5.81 percentage points**.

---

## Error Analysis

Evaluating class-level metrics revealed significant insights beyond aggregate accuracy:

1. **Majority Class Bias:** The fine-tuned model developed a strong bias toward the *Technical Support* queue, leading to high false-positive rates for that category and suppressed recall on minority classes.
2. **Semantic Ambiguity:** Overlapping domains (e.g., *IT Support* vs. *Technical Support*, or *Customer Service* vs. *General Inquiry*) proved difficult to separate purely through token-level classification.
3. **Imbalanced Representation:** Natural distribution skews in synthetic datasets hinder minority class generalization unless explicitly counteracted with reweighting or targeted sampling.

Detailed reports, confusion matrices, and misclassification logs are saved under the `results/` directory.

---

## Repository Structure

```text
SupportOpsAI/
│
├── data/
│   ├── kaggle_raw/
│   ├── processed/
│   │   ├── english_clean.csv
│   │   ├── train.csv
│   │   ├── validation.csv
│   │   └── test.csv
│   └── seed_texts.jsonl
│
├── models/
│   ├── supportopsai-qlora/
│   │   ├── checkpoint-1188/
│   │   └── checkpoint-2376/
│   │
│   └── supportopsai-qlora-final/
│       ├── adapter_model.safetensors
│       ├── adapter_config.json
│       ├── tokenizer.json
│       └── ...
│
├── notebooks/
│   ├── startup.ipynb
│   └── 03_qlora.ipynb
│
├── results/
│   ├── baseline_results.json
│   ├── baseline_vs_qlora.csv
│   ├── qlora_final_metrics.json
│   ├── qlora_test_predictions.csv
│   ├── queue_classification_report.csv
│   ├── queue_confusion_matrix.csv
│   ├── priority_classification_report.csv
│   ├── priority_confusion_matrix.csv
│   └── ...
│
├── src/
├── api/
└── frontend/
```

---

## Training Artifacts

- **Final Trained Adapter:** `models/supportopsai-qlora-final/`
- **Intermediate Checkpoints:** `models/supportopsai-qlora/`

The repository stores the LoRA adapter weights (`adapter_model.safetensors`), configurations, and tokenizer artifacts, eliminating the need to store redundant full-model base weights.

---

## Reproducing the Experiment

The primary training and evaluation pipeline is contained in:

```text
notebooks/03_qlora.ipynb
```

The workflow covers:
1. Dataset loading and inspection
2. Cleaning, deduplication, and stratified train/val/test splitting
3. Instruction formatting and prompt construction
4. Base-model zero-shot benchmarking
5. BitsAndBytes 4-bit loading and PEFT/QLoRA configuration
6. SFT training via TRL
7. Adapter export and serialization
8. Programmatic evaluation, JSON parsing verification, and error analysis

*Designed to execute efficiently on a single consumer or cloud GPU (e.g., NVIDIA T4 / V100 / A10G).*

---

## Technologies Used

- **Frameworks & Modeling:** Python, PyTorch, Hugging Face (`transformers`, `peft`, `trl`, `accelerate`), `bitsandbytes`
- **Evaluation & Data Processing:** Pandas, NumPy, scikit-learn
- **Data Serialization:** JSON, JSONL, CSV, SafeTensors

---

## Limitations

This project is an exploratory LLM fine-tuning experiment and is **not intended for production routing out of the box**:
- The underlying corpus is synthetically generated.
- Queue distributions are imbalanced, resulting in low recall on long-tail categories.
- High semantic overlap exists between categories.
- Real-world production deployment requires human-in-the-loop review, domain-specific golden validation sets, and latency optimizations.

---

## Future Improvements

- [ ] Implement class-weighted cross-entropy loss or focal loss to combat class imbalance.
- [ ] Add hard-negative augmentation and minority class upsampling.
- [ ] Apply targeted prompt/completion masking to ensure loss is computed strictly on response tokens.
- [ ] Expand prediction scope to ticket intent and sentiment.
- [ ] Implement inference acceleration (vLLM, TensorRT-LLM) and endpoint containerization.

---

## Dataset License

This project utilizes the `Tobi-Bueck/customer-support-tickets` dataset, licensed under **CC-BY-NC-4.0**. Please consult the original license terms before using derived artifacts for any commercial applications.

---

## Project Status

**Completed Experimental Implementation.**  
Focus: *QLoRA fine-tuning → Structured JSON prediction → Quantitative evaluation → Error analysis.*
