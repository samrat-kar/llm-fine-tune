# %% [markdown]
# # Notebook 1 — Dataset Preparation
#
# **Goal**: Load `b-mc2/sql-create-context` from Hugging Face, format it
# for instruction fine-tuning, create train / validation splits, and save the
# processed dataset to disk.
#
# **Dataset**: `b-mc2/sql-create-context`
# - ~82,000 rows of (question, SQL schema, SQL answer) triples
# - Natural-language question → correct SQL query given the table definitions
#
# **Output format** (chat template):
# ```
# <|system|>  You are an expert SQL assistant ...
# <|user|>    Given the following SQL tables: ... Write a SQL query to answer: ...
# <|assistant|> SELECT ...
# ```

# %% [markdown]
# ## Setup

# %%
import sys, os
sys.path.insert(0, os.path.abspath(".."))   # so src/ is importable

import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from transformers import AutoTokenizer

from src.config import (
    BASE_MODEL_ID,
    DATASET_NAME,
    DATASET_SPLIT_SEED,
    EVAL_SAMPLE_SIZE,
    MAX_SEQ_LENGTH,
    PROCESSED_DATA_DIR,
    VAL_SIZE,
)
from src.dataset_utils import (
    format_as_chat,
    load_raw_dataset,
    prepare_dataset,
    filter_long_sequences,
    build_prompt,
    SYSTEM_PROMPT,
)

random.seed(DATASET_SPLIT_SEED)
print("All imports OK")

# %% [markdown]
# ## 1. Load the raw dataset

# %%
raw_dataset = load_raw_dataset()
print(raw_dataset)

# %%
# Inspect a few samples
sample = raw_dataset["train"].select(range(3))
for i, row in enumerate(sample):
    print(f"\n--- Sample {i+1} ---")
    print(f"Question : {row['question']}")
    print(f"Context  :\n{row['context']}")
    print(f"Answer   : {row['answer']}")

# %% [markdown]
# ## 2. Dataset statistics

# %%
train_df = raw_dataset["train"].to_pandas()

# Question length distribution
train_df["q_len"] = train_df["question"].str.split().str.len()
train_df["a_len"] = train_df["answer"].str.split().str.len()
train_df["c_len"] = train_df["context"].str.split().str.len()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col, title in zip(axes,
                           ["q_len", "a_len", "c_len"],
                           ["Question length (words)",
                            "Answer length (words)",
                            "Context length (words)"]):
    train_df[col].clip(upper=train_df[col].quantile(0.99)).hist(
        ax=ax, bins=40, color="steelblue", edgecolor="white"
    )
    ax.set_title(title)
    ax.set_xlabel("words")
    ax.set_ylabel("count")

plt.suptitle("Dataset length distributions", y=1.02, fontsize=13)
plt.tight_layout()
plt.savefig("../results/dataset_length_distributions.png", dpi=120, bbox_inches="tight")
plt.show()
print(train_df[["q_len", "a_len", "c_len"]].describe().round(1))

# %% [markdown]
# ## 3. Load tokenizer and format dataset

# %%
print(f"Loading tokenizer: {BASE_MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"Vocabulary size : {tokenizer.vocab_size:,}")
print(f"Chat template   : {'yes' if tokenizer.chat_template else 'no'}")

# %%
# Show what a formatted example looks like
sample_row = raw_dataset["train"][0]
formatted = format_as_chat(sample_row, tokenizer)
print("=== Formatted example ===")
print(formatted["text"])

# %%
# Tokenise to check length
tokens = tokenizer(formatted["text"], return_length=True)
print(f"\nToken count: {tokens['length'][0]} / {MAX_SEQ_LENGTH}")

# %%
# Build & save the full processed dataset
os.makedirs("../results", exist_ok=True)
dataset = prepare_dataset(tokenizer=tokenizer, save=True)
print(dataset)

# %% [markdown]
# ## 4. Filter out sequences that are too long

# %%
dataset_filtered = filter_long_sequences(dataset, tokenizer)
print(dataset_filtered)

# Overwrite saved version with filtered one
dataset_filtered.save_to_disk(PROCESSED_DATA_DIR)
print(f"Saved filtered dataset to {PROCESSED_DATA_DIR}")

# %% [markdown]
# ## 5. Sanity check on the validation split

# %%
val_sample = dataset_filtered["validation"].select(range(5))
print("=== Validation samples ===")
for i, row in enumerate(val_sample):
    print(f"\n--- Val {i+1} ---")
    print(row["text"][:400], "...")

# %% [markdown]
# ## 6. Token-length distribution on processed dataset

# %%
def get_token_lengths(split_ds, sample_n=2000):
    n = min(sample_n, len(split_ds))
    subset = split_ds.select(range(n))
    lengths = tokenizer(list(subset["text"]), return_length=True, truncation=False)["length"]
    return lengths

train_lengths = get_token_lengths(dataset_filtered["train"])
val_lengths   = get_token_lengths(dataset_filtered["validation"])

fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(train_lengths, bins=50, alpha=0.6, label="train", color="steelblue")
ax.hist(val_lengths,   bins=50, alpha=0.6, label="validation", color="coral")
ax.axvline(MAX_SEQ_LENGTH, color="red", linestyle="--", label=f"max_seq={MAX_SEQ_LENGTH}")
ax.set_xlabel("Token count")
ax.set_ylabel("Count")
ax.set_title("Token length distribution (processed dataset)")
ax.legend()
plt.tight_layout()
plt.savefig("../results/token_length_distribution.png", dpi=120, bbox_inches="tight")
plt.show()

print(f"\nTrain   — mean: {sum(train_lengths)/len(train_lengths):.1f}, "
      f"max: {max(train_lengths)}, p95: {sorted(train_lengths)[int(0.95*len(train_lengths))]}")
print(f"Val     — mean: {sum(val_lengths)/len(val_lengths):.1f}, "
      f"max: {max(val_lengths)}, p95: {sorted(val_lengths)[int(0.95*len(val_lengths))]}")

# %% [markdown]
# ## Summary
#
# | Item | Value |
# |------|-------|
# | Dataset | `b-mc2/sql-create-context` |
# | Raw train rows | ~82,000 |
# | Val split (5 %) | ~4,100 |
# | Max sequence length | 512 tokens |
# | Format | Qwen2.5 chat template |
# | Saved to | `./data/processed/` |
#
# Proceed to **Notebook 2** for baseline evaluation.
