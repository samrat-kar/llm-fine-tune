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
batch_size      = 4          # per device
grad_accum      = 4          # effective batch = 16
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

*(Fill in after running Notebook 4)*

| Metric | Baseline | Fine-tuned | Δ |
|--------|----------|------------|---|
| ROUGE-L | — | — | — |
| Exact Match | — | — | — |
| MMLU Overall | — | — | — |

W&B project: `<link from Notebook 3>`

## Links

- Published model: `https://huggingface.co/<your-hf-username>/qwen2.5-1.5b-sql-qlora`
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
