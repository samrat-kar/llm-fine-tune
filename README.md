# LLM Fine-Tuning: Text-to-SQL with QLoRA

Fine-tuning `Qwen/Qwen2.5-1.5B-Instruct` for natural language → SQL generation
using **QLoRA** (4-bit quantization + LoRA) on the `b-mc2/sql-create-context` dataset.

## Project Overview

| Item | Detail |
|------|--------|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Task | Text-to-SQL generation |
| Dataset | `b-mc2/sql-create-context` (~82K rows) |
| Fine-tuning method | QLoRA (4-bit NF4, LoRA r=16) |
| Framework | HuggingFace PEFT + TRL |
| Experiment tracking | Weights & Biases |
| Published model | `<your-hf-username>/qwen2.5-1.5b-sql-qlora` |

## Workflow

```
Notebook 1  →  Dataset preparation & formatting
Notebook 2  →  Baseline evaluation (before fine-tuning)
Notebook 3  →  QLoRA fine-tuning (logs to W&B)
Notebook 4  →  Post fine-tuning evaluation + catastrophic forgetting check
Notebook 5  →  Merge adapter & publish to Hugging Face Hub
```

## Quick Start

### 1. Environment setup

```bash
# Python 3.10+ recommended
pip install -r requirements.txt
```

> **GPU requirement**: A single GPU with ≥16 GB VRAM (or Google Colab T4 free tier)
> is sufficient for QLoRA with Qwen2.5-1.5B.

### 2. Configure

Edit `src/config.py`:
- `HF_USERNAME` — your Hugging Face username (for publishing)
- `WANDB_PROJECT` — your W&B project name

### 3. Run the notebooks in order

The notebooks are written in Jupytext percent-script format (`.py` files with
`# %%` cell markers). Open them in VS Code (with the Jupyter extension) or
convert to `.ipynb` with:

```bash
jupytext --to notebook notebooks/01_dataset_preparation.py
```

Or run them directly:

```bash
cd notebooks
python 01_dataset_preparation.py
python 02_baseline_evaluation.py
python 03_fine_tuning.py
python 04_evaluation_comparison.py
python 05_publish_to_hub.py
```

### 4. Google Colab

Upload the whole repo to your Google Drive, mount it in Colab, and run each
notebook. The scripts work unmodified on Colab's T4 GPU.

```python
# In Colab — mount drive and set working dir
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/llm-fine-tune
!pip install -r requirements.txt -q
```

## Repository Structure

```
llm-fine-tune/
├── requirements.txt          # All dependencies
├── README.md                 # This file
├── src/
│   ├── config.py             # All hyperparameters (single source of truth)
│   ├── dataset_utils.py      # Dataset loading, formatting, filtering
│   └── evaluation_utils.py   # ROUGE-L, exact match, MMLU evaluation
├── notebooks/
│   ├── 01_dataset_preparation.py
│   ├── 02_baseline_evaluation.py
│   ├── 03_fine_tuning.py
│   ├── 04_evaluation_comparison.py
│   └── 05_publish_to_hub.py
├── data/
│   └── processed/            # Saved HuggingFace dataset (auto-created)
├── outputs/
│   ├── qlora-sql/            # Training checkpoints
│   └── merged_model/         # Merged model weights (for publishing)
└── results/
    ├── baseline_results.json
    ├── finetuned_results.json
    ├── finetuned_mmlu_results.json
    ├── final_summary.csv
    └── *.png                 # Plots
```

## Training Configuration

```python
# LoRA
lora_r          = 16
lora_alpha      = 32
lora_dropout    = 0.05
target_modules  = ["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"]

# Training
learning_rate   = 2e-4
lr_scheduler    = "cosine"
warmup_ratio    = 0.05
epochs          = 3
batch_size      = 2          # per device
grad_accum      = 8          # effective batch = 16
max_seq_length  = 512
optimizer       = "paged_adamw_32bit"

# Quantization
bits            = 4
quant_type      = "nf4"
compute_dtype   = "float16"
double_quant    = True
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **ROUGE-L** | Longest common subsequence overlap between predicted and reference SQL |
| **Exact Match** | Normalised string equality (lower-case, whitespace-collapsed) |
| **MMLU Accuracy** | Multiple-choice accuracy on 3 MMLU subjects (catastrophic forgetting check) |

## Results

Evaluated on 500 held-out validation samples. Training on NVIDIA H100 80GB (~87 min).

| Metric | Baseline | Fine-tuned | Δ |
|--------|----------|------------|---|
| ROUGE-L | 0.8784 | 0.9856 | **+0.1072** |
| Exact Match | 0.0000 | 0.7540 | **+0.7540** |
| MMLU Overall (FT only) | N/A¹ | 0.4800 | — |

¹ MMLU baseline was not run; fine-tuned score of 0.48 is well above random (0.25),
confirming general capability is retained.

**Training loss**: 2.26 → 0.41 (train), 0.53 → 0.43 (val) over 1,800 steps / 3 epochs.

W&B project: https://wandb.ai/samratkar77/ReadyTensor-FineTune/runs/esrwl3zs

## Model Card

| Field | Value |
|-------|-------|
| **Model name** | `samratkar77/qwen2.5-1.5b-sql-qlora` |
| **Base model** | `Qwen/Qwen2.5-1.5B-Instruct` |
| **Task** | Text-to-SQL generation |
| **Language** | English |
| **License** | Apache 2.0 |
| **Fine-tuning method** | QLoRA — 4-bit NF4 quantization + LoRA (r=16, α=32) |
| **Training data** | `b-mc2/sql-create-context` (~78K train rows after filtering) |
| **Trainable parameters** | ~8.4M / 1.54B (≈0.55%) |

### Intended Use

Generate SQL queries from a natural-language question and a set of `CREATE TABLE`
statements. Suitable for prototyping text-to-SQL features, database assistants, and
educational projects.

### Out-of-Scope Use

- Complex multi-table joins not well-represented in the training data
- Non-English queries
- Dialect-specific SQL (T-SQL, PL/pgSQL); results may be unreliable
- Production database access without human review

### Limitations and Biases

- Training data covers primarily single-table or simple join scenarios
- May hallucinate column or table names not present in the given schema
- Performance degrades for aggregate functions with complex `HAVING` clauses

### Quick Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "samratkar77/qwen2.5-1.5b-sql-qlora"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="auto"
)

system_prompt = (
    "You are an expert SQL assistant. "
    "Given a natural language question and the relevant database schema, "
    "write a single correct SQL query that answers the question. "
    "Return only the SQL query with no explanation."
)
question = "How many employees are in the sales department?"
context  = "CREATE TABLE employees (id INT, name TEXT, department TEXT, salary REAL);"

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user",   "content": f"Given the following SQL tables:\n\n{context}\n\nWrite a SQL query to answer: {question}"},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
print(tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
# Expected: SELECT COUNT(*) FROM employees WHERE department = 'sales';
```

## Links

- Published model: `https://huggingface.co/samratkar77/qwen2.5-1.5b-sql-qlora`
- Dataset: `https://huggingface.co/datasets/b-mc2/sql-create-context`
- Base model: `https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct`
- W&B project: *(add link after training)*

## Reproducibility

All random seeds are fixed (`seed=42`). The full pipeline is deterministic given
the same GPU hardware and software versions. To reproduce from scratch:

```bash
pip install -r requirements.txt
python notebooks/01_dataset_preparation.py
python notebooks/02_baseline_evaluation.py
python notebooks/03_fine_tuning.py
python notebooks/04_evaluation_comparison.py
```

Expected runtime on a single T4 GPU:
- Notebooks 1-2: ~15 minutes
- Notebook 3 (training): ~2-3 hours
- Notebook 4: ~20 minutes
