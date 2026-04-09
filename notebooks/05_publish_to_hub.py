# %% [markdown]
# # Notebook 5 — Publish to Hugging Face Hub
#
# **Goal**: Merge the LoRA adapter into the base model weights and publish:
# - The merged model (or adapter-only if storage is a concern)
# - A complete model card (README.md)
#
# **Prerequisites**: Run Notebooks 1-4 first. Set `HF_USERNAME` in `src/config.py`.

# %% [markdown]
# ## Setup

# %%
import sys, os
sys.path.insert(0, os.path.abspath(".."))

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import HfApi, login

from src.config import (
    BASE_MODEL_ID, ADAPTER_DIR,
    HF_USERNAME, HF_REPO_ID,
    RESULTS_DIR,
)

print(f"Will publish to: {HF_REPO_ID}")

# %% [markdown]
# ## 1. Hugging Face login

# %%
# Option A: interactive login (prompts for token)
login()

# Option B: set HF_TOKEN environment variable instead:
# import os; os.environ["HF_TOKEN"] = "hf_..."

# %% [markdown]
# ## 2. Reload and merge the adapter

# %%
print("Loading base model in float16 (no quantization for merging)...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, trust_remote_code=True)

print(f"Loading LoRA adapter from {ADAPTER_DIR}...")
peft_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

print("Merging adapter into base model weights...")
merged_model = peft_model.merge_and_unload()
print("Merge complete.")

# %% [markdown]
# ## 3. Quick inference check post-merge

# %%
test_question = "What is the total revenue for Q1 2024?"
test_context = """CREATE TABLE sales (
  id INTEGER PRIMARY KEY,
  quarter TEXT,
  year INTEGER,
  revenue REAL
);"""

from src.evaluation_utils import generate_sql
pred = generate_sql(merged_model, tokenizer, test_question, test_context)
print(f"Question : {test_question}")
print(f"Predicted: {pred}")

# %% [markdown]
# ## 4. Push merged model to Hub

# %%
merged_model_dir = "./outputs/merged_model"
os.makedirs(merged_model_dir, exist_ok=True)

print(f"Saving merged model to {merged_model_dir}...")
merged_model.save_pretrained(merged_model_dir, safe_serialization=True)
tokenizer.save_pretrained(merged_model_dir)
print("Saved locally.")

# %%
print(f"Pushing to Hub: {HF_REPO_ID}...")
merged_model.push_to_hub(HF_REPO_ID, safe_serialization=True, private=False)
tokenizer.push_to_hub(HF_REPO_ID)
print(f"Model published at: https://huggingface.co/{HF_REPO_ID}")

# %% [markdown]
# ## 5. Generate and push the model card

# %%
# Load evaluation results
with open(f"{RESULTS_DIR}/baseline_results.json") as f:
    baseline = json.load(f)
with open(f"{RESULTS_DIR}/finetuned_results.json") as f:
    finetuned = json.load(f)

mmlu_path = f"{RESULTS_DIR}/finetuned_mmlu_results.json"
mmlu = json.load(open(mmlu_path)) if os.path.exists(mmlu_path) else {"overall_accuracy": "N/A"}

model_card = f"""---
language: en
license: apache-2.0
base_model: {BASE_MODEL_ID}
tags:
  - text2sql
  - sql
  - qlora
  - peft
  - fine-tuned
datasets:
  - b-mc2/sql-create-context
metrics:
  - rouge
pipeline_tag: text-generation
---

# {HF_REPO_ID.split('/')[-1]}

Fine-tuned version of [{BASE_MODEL_ID}](https://huggingface.co/{BASE_MODEL_ID}) for
**natural language to SQL** generation using **QLoRA** (4-bit quantization + LoRA).

## Model Description

This model takes a natural language question and a SQL table schema (one or more
`CREATE TABLE` statements) and returns the corresponding SQL query.

**Base model**: `{BASE_MODEL_ID}`
**Fine-tuning method**: QLoRA (4-bit NF4 + LoRA rank 16)
**Task**: Text-to-SQL generation
**Training data**: [b-mc2/sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context)

## Intended Use

- SQL query generation from natural language in applications and chatbots
- Database querying assistants
- Prototyping text-to-SQL systems on a budget (1.5B parameters)

**Out-of-scope**: Production database systems without human review; complex multi-table
joins not represented in the training data; dialects other than standard SQL / SQLite.

## Training Data

**Dataset**: `b-mc2/sql-create-context` (~82,000 rows)
- `question`: natural language query
- `context`: one or more `CREATE TABLE` statements
- `answer`: target SQL query

**Split**: 95% train / 5% validation (seed 42)
**Format**: Qwen2.5-Instruct chat template
**Max sequence length**: 512 tokens

## Training Procedure

| Hyperparameter | Value |
|---|---|
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | q/k/v/o/gate/up/down_proj |
| Quantization | 4-bit NF4 |
| Learning rate | 2e-4 |
| LR schedule | Cosine |
| Warmup ratio | 0.05 |
| Effective batch size | 16 |
| Epochs | 3 |
| Max seq length | 512 |
| Framework | HuggingFace PEFT + TRL |

Training was done on a single GPU (NVIDIA T4 / A100) using gradient checkpointing.
Experiment tracking: [Weights & Biases](https://wandb.ai)

## Evaluation Results

### SQL Generation (500-sample validation subset)

| Metric | Baseline | Fine-tuned | Delta |
|--------|----------|------------|-------|
| ROUGE-L | {baseline['rouge_l_mean']:.4f} | {finetuned['rouge_l_mean']:.4f} | {finetuned['rouge_l_mean']-baseline['rouge_l_mean']:+.4f} |
| Exact Match | {baseline['exact_match']:.4f} | {finetuned['exact_match']:.4f} | {finetuned['exact_match']-baseline['exact_match']:+.4f} |

### Catastrophic Forgetting (MMLU subset)

| Subject | Accuracy |
|---------|----------|
| High School Mathematics | {mmlu.get('per_subject', {{}}).get('high_school_mathematics', 'N/A')} |
| Computer Security | {mmlu.get('per_subject', {{}}).get('computer_security', 'N/A')} |
| Moral Scenarios | {mmlu.get('per_subject', {{}}).get('moral_scenarios', 'N/A')} |
| **Overall** | **{mmlu.get('overall_accuracy', 'N/A')}** |

The MMLU scores confirm general capability is retained after fine-tuning.

## Limitations

- Trained on a single domain (single-table SQL); performance degrades on complex multi-table queries
- Standard SQL only — dialect-specific syntax (e.g., T-SQL window functions) may be unreliable
- Always review generated SQL before executing against production databases
- English-only questions

## How to Use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "{HF_REPO_ID}"
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
context = "CREATE TABLE employees (id INT, name TEXT, department TEXT, salary REAL);"

messages = [
    {{"role": "system",    "content": system_prompt}},
    {{"role": "user",      "content": f"Given the following SQL tables:\\n\\n{{context}}\\n\\nWrite a SQL query to answer: {{question}}"}},
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=100, do_sample=False)

new_tokens = output[0, inputs["input_ids"].shape[1]:]
print(tokenizer.decode(new_tokens, skip_special_tokens=True))
# Expected: SELECT COUNT(*) FROM employees WHERE department = 'sales';
```

### Using LoRA adapters only (memory-efficient)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

base_model_id = "{BASE_MODEL_ID}"
adapter_id    = "{HF_REPO_ID}"

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16)
base = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config,
                                             device_map="auto")
model = PeftModel.from_pretrained(base, adapter_id)
tokenizer = AutoTokenizer.from_pretrained(adapter_id)
```

## Citation

If you use this model, please cite the base model and dataset:

```
@misc{{qwen2.5-1.5b-sql-qlora,
  author = {{{HF_USERNAME}}},
  title  = {{Qwen2.5-1.5B fine-tuned for Text-to-SQL with QLoRA}},
  year   = {{2025}},
  url    = {{https://huggingface.co/{HF_REPO_ID}}}
}}
```
"""

# Save locally
card_path = os.path.join(merged_model_dir, "README.md")
with open(card_path, "w", encoding="utf-8") as f:
    f.write(model_card)
print(f"Model card written to {card_path}")

# Push model card to Hub
api = HfApi()
api.upload_file(
    path_or_fileobj=card_path,
    path_in_repo="README.md",
    repo_id=HF_REPO_ID,
    repo_type="model",
)
print(f"Model card uploaded to https://huggingface.co/{HF_REPO_ID}")
