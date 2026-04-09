# %% [markdown]
# # Notebook 4 — Post-Fine-Tuning Evaluation & Comparison
#
# **Goal**: Evaluate the fine-tuned model on the same validation subset used
# in Notebook 2, compare before/after metrics, and run a catastrophic-
# forgetting check using an MMLU subset.
#
# **Metrics**:
# - ROUGE-L and Exact Match (SQL task)
# - MMLU accuracy (general capability retention)

# %% [markdown]
# ## Setup

# %%
import sys, os
sys.path.insert(0, os.path.abspath(".."))

import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from src.config import (
    BASE_MODEL_ID, ADAPTER_DIR,
    EVAL_SAMPLE_SIZE, RESULTS_DIR,
    LOAD_IN_4BIT, BNB_4BIT_COMPUTE_DTYPE, BNB_4BIT_QUANT_TYPE, BNB_USE_DOUBLE_QUANT,
)
from src.evaluation_utils import (
    evaluate_model, save_results, print_comparison,
    load_mmlu_subset, evaluate_mmlu,
)

os.makedirs(RESULTS_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# %% [markdown]
# ## 1. Load evaluation sample (same as Notebook 2)

# %%
raw_val = load_dataset("b-mc2/sql-create-context", split="train").train_test_split(
    test_size=0.05, seed=42
)["test"]

n_eval = min(EVAL_SAMPLE_SIZE, len(raw_val))
eval_rows = [raw_val[i] for i in range(n_eval)]
print(f"Evaluation sample: {n_eval} rows")

# %% [markdown]
# ## 2. Load fine-tuned model

# %%
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=LOAD_IN_4BIT,
    bnb_4bit_compute_dtype=getattr(torch, BNB_4BIT_COMPUTE_DTYPE),
    bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
    bnb_4bit_use_double_quant=BNB_USE_DOUBLE_QUANT,
)

print(f"Loading base model: {BASE_MODEL_ID}")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

print(f"Loading LoRA adapter from: {ADAPTER_DIR}")
ft_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
ft_model.eval()
print("Fine-tuned model loaded.")

# %% [markdown]
# ## 3. Evaluate fine-tuned model

# %%
ft_results = evaluate_model(
    model=ft_model,
    tokenizer=tokenizer,
    eval_rows=eval_rows,
    device=device,
    label="fine-tuned",
)
save_results(ft_results, "finetuned_results.json")

# %% [markdown]
# ## 4. Load baseline results and compare

# %%
baseline_path = os.path.join(RESULTS_DIR, "baseline_results.json")
with open(baseline_path) as f:
    baseline_results = json.load(f)

print_comparison(baseline_results, ft_results)

# %% [markdown]
# ## 5. Visualise the improvement

# %%
metrics = ["rouge_l_mean", "exact_match"]
labels  = ["ROUGE-L", "Exact Match"]

baseline_vals = [baseline_results[m] for m in metrics]
ft_vals       = [ft_results[m]       for m in metrics]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
bars_b = ax.bar(x - width/2, baseline_vals, width, label="Baseline", color="steelblue")
bars_f = ax.bar(x + width/2, ft_vals,       width, label="Fine-tuned", color="coral")

ax.set_ylabel("Score")
ax.set_title("Baseline vs Fine-tuned: SQL Generation Metrics")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1)
ax.legend()

# Annotate bars
for bar in bars_b:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
for bar in bars_f:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/baseline_vs_finetuned.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Qualitative comparison — sample predictions

# %%
# We need baseline predictions too; load from file or re-run notebook 2
baseline_pred_path = os.path.join(RESULTS_DIR, "baseline_predictions.json")

# If predictions were saved separately, load them; otherwise show fine-tuned samples only
print("\n=== Fine-tuned model predictions ===")
for i in range(8):
    print(f"\n[{i+1}] Question : {eval_rows[i]['question']}")
    print(f"     Reference: {eval_rows[i]['answer']}")
    print(f"     Predicted: {ft_results['predictions'][i]}")

# %% [markdown]
# ## 7. Catastrophic forgetting check — MMLU

# %%
# This evaluates general language understanding on 3 MMLU subjects
# (high_school_mathematics, computer_security, moral_scenarios)
# to verify fine-tuning did not harm general capability.

mmlu_rows = load_mmlu_subset()
ft_mmlu = evaluate_mmlu(ft_model, tokenizer, mmlu_rows, device=device)
save_results(ft_mmlu, "finetuned_mmlu_results.json")

# %%
# If you also ran MMLU on the baseline (Notebook 2), load and compare
baseline_mmlu_path = os.path.join(RESULTS_DIR, "baseline_mmlu_results.json")
if os.path.exists(baseline_mmlu_path):
    with open(baseline_mmlu_path) as f:
        baseline_mmlu = json.load(f)

    print("\n=== MMLU Accuracy (Catastrophic Forgetting Check) ===")
    print(f"{'Subject':<35} {'Baseline':>10} {'Fine-tuned':>12} {'Delta':>8}")
    print("-" * 70)

    subjects = list(ft_mmlu["per_subject"].keys())
    for subj in subjects:
        b_acc = baseline_mmlu["per_subject"].get(subj, 0)
        f_acc = ft_mmlu["per_subject"].get(subj, 0)
        delta = f_acc - b_acc
        sign = "+" if delta >= 0 else ""
        print(f"{subj:<35} {b_acc:>10.4f} {f_acc:>12.4f} {sign+f'{delta:.4f}':>8}")

    print("-" * 70)
    b_ov = baseline_mmlu["overall_accuracy"]
    f_ov = ft_mmlu["overall_accuracy"]
    delta_ov = f_ov - b_ov
    sign = "+" if delta_ov >= 0 else ""
    print(f"{'Overall':<35} {b_ov:>10.4f} {f_ov:>12.4f} {sign+f'{delta_ov:.4f}':>8}")
else:
    print("Baseline MMLU not found. Run MMLU section in Notebook 2 first.")
    print(f"\nFine-tuned MMLU accuracy: {ft_mmlu['overall_accuracy']:.4f}")

# %% [markdown]
# ## 8. MMLU visualisation

# %%
if os.path.exists(baseline_mmlu_path):
    with open(baseline_mmlu_path) as f:
        baseline_mmlu = json.load(f)

    subjects = list(ft_mmlu["per_subject"].keys())
    b_vals = [baseline_mmlu["per_subject"].get(s, 0) for s in subjects]
    f_vals = [ft_mmlu["per_subject"].get(s, 0) for s in subjects]

    # Add overall
    subjects.append("Overall")
    b_vals.append(baseline_mmlu["overall_accuracy"])
    f_vals.append(ft_mmlu["overall_accuracy"])

    x = np.arange(len(subjects))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.2, b_vals, 0.4, label="Baseline",    color="steelblue", alpha=0.85)
    ax.bar(x + 0.2, f_vals, 0.4, label="Fine-tuned",  color="coral",     alpha=0.85)
    ax.axhline(0.25, color="grey", linestyle=":", linewidth=1, label="Random (25%)")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", " ") for s in subjects], rotation=15, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("MMLU: Catastrophic Forgetting Check")
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/mmlu_comparison.png", dpi=120, bbox_inches="tight")
    plt.show()

# %% [markdown]
# ## 9. Summary table

# %%
summary = {
    "Metric": ["ROUGE-L", "Exact Match", "MMLU Overall"],
    "Baseline":   [
        f"{baseline_results['rouge_l_mean']:.4f}",
        f"{baseline_results['exact_match']:.4f}",
        f"{baseline_mmlu['overall_accuracy']:.4f}" if os.path.exists(baseline_mmlu_path) else "N/A",
    ],
    "Fine-tuned": [
        f"{ft_results['rouge_l_mean']:.4f}",
        f"{ft_results['exact_match']:.4f}",
        f"{ft_mmlu['overall_accuracy']:.4f}",
    ],
}
summary_df = pd.DataFrame(summary)
print("\n=== Final Results Summary ===")
print(summary_df.to_string(index=False))
summary_df.to_csv(f"{RESULTS_DIR}/final_summary.csv", index=False)
print(f"\nSaved to {RESULTS_DIR}/final_summary.csv")

# %% [markdown]
# ## Next Steps
#
# 1. Review results — if ROUGE-L improved significantly, proceed to publishing
# 2. Run **Notebook 5 (optional)**: merge adapter into base model
# 3. Publish to Hugging Face Hub (see `notebooks/05_publish_to_hub.py`)
