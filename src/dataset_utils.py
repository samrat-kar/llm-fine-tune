"""
Dataset utilities for the SQL fine-tuning project.

The raw dataset (b-mc2/sql-create-context) has three columns:
  question  – natural language query
  context   – one or more CREATE TABLE statements
  answer    – the target SQL query

We convert each row into a chat-style instruction-response pair that
matches Qwen2.5-Instruct's expected format.
"""

from __future__ import annotations

import os
from typing import Optional

from datasets import DatasetDict, load_dataset, load_from_disk

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import (
    DATASET_NAME,
    DATASET_SPLIT_SEED,
    EVAL_SAMPLE_SIZE,
    MAX_SEQ_LENGTH,
    PROCESSED_DATA_DIR,
    VAL_SIZE,
)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert SQL assistant. "
    "Given a natural language question and the relevant database schema, "
    "write a single correct SQL query that answers the question. "
    "Return only the SQL query with no explanation."
)


def build_prompt(question: str, context: str) -> str:
    """Return the user-turn text (no special tokens)."""
    return (
        f"Given the following SQL tables:\n\n"
        f"{context.strip()}\n\n"
        f"Write a SQL query to answer: {question.strip()}"
    )


def format_as_chat(row: dict, tokenizer=None) -> dict:
    """
    Convert a dataset row into the tokenizer's chat template format.

    If a tokenizer is provided, apply_chat_template is used so the output
    contains the model-specific special tokens. Otherwise a plain-text
    version is returned (useful for inspection).
    """
    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": build_prompt(row["question"], row["context"])},
        {"role": "assistant", "content": row["answer"].strip()},
    ]

    if tokenizer is not None:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    else:
        text = (
            f"<|system|>\n{SYSTEM_PROMPT}\n"
            f"<|user|>\n{build_prompt(row['question'], row['context'])}\n"
            f"<|assistant|>\n{row['answer'].strip()}"
        )

    return {"text": text}


# ---------------------------------------------------------------------------
# Dataset loading helpers
# ---------------------------------------------------------------------------

def load_raw_dataset() -> DatasetDict:
    """Download the raw dataset from the Hub."""
    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME)
    print(f"  train rows : {len(dataset['train']):,}")
    return dataset


def prepare_dataset(tokenizer=None, save: bool = True) -> DatasetDict:
    """
    Full pipeline:
      1. Load raw data
      2. Create a val split from train
      3. Format every row as chat text
      4. Optionally save to disk
    """
    raw = load_raw_dataset()

    # The dataset only has a 'train' split — carve out validation
    split = raw["train"].train_test_split(
        test_size=VAL_SIZE, seed=DATASET_SPLIT_SEED
    )
    dataset = DatasetDict({"train": split["train"], "validation": split["test"]})

    print(f"  train      : {len(dataset['train']):,}")
    print(f"  validation : {len(dataset['validation']):,}")

    # Apply formatting
    dataset = dataset.map(
        lambda row: format_as_chat(row, tokenizer),
        remove_columns=dataset["train"].column_names,
        desc="Formatting",
    )

    if save:
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        dataset.save_to_disk(PROCESSED_DATA_DIR)
        print(f"  saved to   : {PROCESSED_DATA_DIR}")

    return dataset


def load_processed_dataset() -> DatasetDict:
    """Load the preprocessed dataset from disk (must run prepare_dataset first)."""
    return load_from_disk(PROCESSED_DATA_DIR)


def get_eval_sample(dataset: DatasetDict, split: str = "validation") -> DatasetDict:
    """Return a fixed-size subsample for fast evaluation."""
    n = min(EVAL_SAMPLE_SIZE, len(dataset[split]))
    return dataset[split].select(range(n))


# ---------------------------------------------------------------------------
# Tokenisation filter (remove examples that exceed max length)
# ---------------------------------------------------------------------------

def filter_long_sequences(dataset: DatasetDict, tokenizer) -> DatasetDict:
    """Drop rows whose tokenised length exceeds MAX_SEQ_LENGTH."""

    def _is_short(row):
        ids = tokenizer(row["text"], return_length=True)["length"][0]
        return ids <= MAX_SEQ_LENGTH

    before = {k: len(v) for k, v in dataset.items()}
    dataset = dataset.filter(_is_short, desc="Filtering long sequences")
    after = {k: len(v) for k, v in dataset.items()}

    for split in before:
        dropped = before[split] - after[split]
        print(f"  [{split}] dropped {dropped} long rows ({dropped/before[split]:.1%})")

    return dataset
