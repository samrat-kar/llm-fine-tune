# %% [markdown]
# # Notebook 2 — Baseline Evaluation
#
# **Goal**: Evaluate the *unmodified* `Qwen/Qwen2.5-1.5B-Instruct` model on
# the SQL-generation task to establish a baseline before fine-tuning.
#
# **Metrics**:
# - **ROUGE-L** — soft overlap between predicted and reference SQL
# - **Exact Match** — normalised string equality
#
# The same evaluation sample (first 500 rows of the validation split) is used
# here and in Notebook 4 so results are directly comparable.

# %% [markdown]
# ## Setup

# %%
import sys, os
sys.path.insert(0, os.path.abspath(".."))

import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.config import (
    BASE_MODEL_ID,
    EVAL_SAMPLE_SIZE,
    LOAD_IN_4BIT,
    BNB_4BIT_COMPUTE_DTYPE,
    BNB_4BIT_QUANT_TYPE,
    BNB_USE_DOUBLE_QUANT,
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
)
from src.evaluation_utils import evaluate_model, save_results

os.makedirs(RESULTS_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# %% [markdown]
# ## 1. Load tokenizer and base model (4-bit for memory efficiency)

# %%
print(f"Loading tokenizer: {BASE_MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# %%
bnb_config = BitsAndBytesConfig(
    load_in_4bit=LOAD_IN_4BIT,
    bnb_4bit_compute_dtype=getattr(torch, BNB_4BIT_COMPUTE_DTYPE),
    bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
    bnb_4bit_use_double_quant=BNB_USE_DOUBLE_QUANT,
)

print(f"Loading model: {BASE_MODEL_ID}")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
print("Model loaded.")
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params/1e9:.2f}B")

# %% [markdown]
# ## 2. Load evaluation sample
#
# We use the **original** (unformatted) validation rows so we can pass
# `question` and `context` to the generation function separately.

# %%
from datasets import load_dataset

raw_val = load_dataset("b-mc2/sql-create-context", split="train").train_test_split(
    test_size=0.05, seed=42
)["test"]

n_eval = min(EVAL_SAMPLE_SIZE, len(raw_val))
eval_rows = [raw_val[i] for i in range(n_eval)]
print(f"Evaluation sample size: {n_eval}")
print(f"\nSample row:")
print(f"  question : {eval_rows[0]['question']}")
print(f"  answer   : {eval_rows[0]['answer']}")

# %% [markdown]
# ## 3. Run baseline evaluation

# %%
# NOTE: This will take several minutes on a T4 GPU.
baseline_results = evaluate_model(
    model=model,
    tokenizer=tokenizer,
    eval_rows=eval_rows,
    device=device,
    label="baseline",
)

# %%
save_results(baseline_results, "baseline_results.json")

# %% [markdown]
# ## 4. Inspect predictions

# %%
print("\n=== Prediction samples (baseline) ===")
for i in range(5):
    print(f"\n--- Sample {i+1} ---")
    print(f"Question  : {eval_rows[i]['question']}")
    print(f"Reference : {eval_rows[i]['answer']}")
    print(f"Predicted : {baseline_results['predictions'][i]}")

# %% [markdown]
# ## 5. Error analysis — length comparison

# %%
import numpy as np

pred_lens = [len(p.split()) for p in baseline_results["predictions"]]
ref_lens  = [len(r.split()) for r in baseline_results["references"]]

fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(ref_lens, pred_lens, alpha=0.3, s=10, color="steelblue")
lim = max(max(ref_lens), max(pred_lens))
ax.plot([0, lim], [0, lim], "r--", linewidth=1, label="perfect match")
ax.set_xlabel("Reference length (words)")
ax.set_ylabel("Predicted length (words)")
ax.set_title("Baseline: reference vs predicted SQL length")
ax.legend()
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/baseline_length_scatter.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Baseline MMLU check
#
# Evaluates general capability on 3 MMLU subjects so notebook 4 can show
# the full catastrophic-forgetting comparison table.

# %%
from src.evaluation_utils import load_mmlu_subset, evaluate_mmlu

print("Running MMLU baseline (150 questions across 3 subjects)...")
mmlu_rows = load_mmlu_subset()
baseline_mmlu = evaluate_mmlu(model, tokenizer, mmlu_rows, device=device)
save_results(baseline_mmlu, "baseline_mmlu_results.json")

# %% [markdown]
# ## Summary
#
# | Metric | Baseline Score |
# |--------|---------------|
# | ROUGE-L | (see output above) |
# | Exact Match | (see output above) |
#
# These numbers are the **reference point**. After fine-tuning (Notebook 3)
# we expect both metrics to increase substantially.
#
# Proceed to **Notebook 3** to run QLoRA fine-tuning.
