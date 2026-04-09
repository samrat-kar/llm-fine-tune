"""
Evaluation utilities for the SQL fine-tuning project.

Metrics:
  - ROUGE-L  : soft string overlap (standard NLG metric)
  - Exact Match (EM): case-insensitive normalised SQL equality
  - Execution Accuracy: optional, requires sqlite3

General-capability check (catastrophic forgetting):
  - A small MMLU subset evaluated via multiple-choice accuracy
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
from typing import Optional

import numpy as np
import torch
from rouge_score import rouge_scorer
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import RESULTS_DIR
from src.dataset_utils import SYSTEM_PROMPT, build_prompt


# ---------------------------------------------------------------------------
# SQL normalisation
# ---------------------------------------------------------------------------

def normalise_sql(sql: str) -> str:
    """Lower-case, collapse whitespace, strip trailing semicolons."""
    sql = sql.strip().lower()
    sql = re.sub(r"\s+", " ", sql)
    sql = sql.rstrip(";").strip()
    return sql


def exact_match(prediction: str, reference: str) -> bool:
    return normalise_sql(prediction) == normalise_sql(reference)


# ---------------------------------------------------------------------------
# Generation helper
# ---------------------------------------------------------------------------

def generate_sql(
    model,
    tokenizer,
    question: str,
    context: str,
    max_new_tokens: int = 150,
    device: str = "cuda",
) -> str:
    """Generate a SQL query for a single (question, context) pair."""
    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": build_prompt(question, context)},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens (skip the prompt)
    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    tokenizer,
    eval_rows: list[dict],
    device: str = "cuda",
    label: str = "model",
) -> dict:
    """
    Run the model on eval_rows and compute ROUGE-L and Exact Match.

    eval_rows must contain 'question', 'context', 'answer' keys
    (the original, unformatted columns).

    Returns a dict with aggregate metrics and per-row predictions.
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    predictions, references = [], []
    em_scores = []

    for row in tqdm(eval_rows, desc=f"Evaluating [{label}]"):
        pred = generate_sql(
            model, tokenizer,
            question=row["question"],
            context=row["context"],
            device=device,
        )
        ref = row["answer"].strip()

        predictions.append(pred)
        references.append(ref)
        em_scores.append(int(exact_match(pred, ref)))

    rouge_scores = [
        scorer.score(ref, pred)["rougeL"].fmeasure
        for ref, pred in zip(references, predictions)
    ]

    results = {
        "label": label,
        "n_samples": len(eval_rows),
        "rouge_l_mean": float(np.mean(rouge_scores)),
        "rouge_l_std":  float(np.std(rouge_scores)),
        "exact_match":  float(np.mean(em_scores)),
        "predictions":  predictions,
        "references":   references,
    }

    print(f"\n[{label}] Results on {results['n_samples']} samples:")
    print(f"  ROUGE-L     : {results['rouge_l_mean']:.4f} ± {results['rouge_l_std']:.4f}")
    print(f"  Exact Match : {results['exact_match']:.4f}")

    return results


# ---------------------------------------------------------------------------
# MMLU subset — catastrophic forgetting check
# ---------------------------------------------------------------------------

MMLU_SUBJECTS = ["high_school_mathematics", "computer_security", "moral_scenarios"]
MMLU_N_PER_SUBJECT = 50   # rows per subject (keep eval fast)


def load_mmlu_subset() -> list[dict]:
    """Load a small MMLU subset for forgetting evaluation."""
    from datasets import load_dataset as _ld

    rows = []
    for subject in MMLU_SUBJECTS:
        ds = _ld("cais/mmlu", subject, split="test")
        n = min(MMLU_N_PER_SUBJECT, len(ds))
        for i in range(n):
            rows.append(
                {
                    "question": ds[i]["question"],
                    "choices":  ds[i]["choices"],
                    "answer":   ds[i]["answer"],   # 0-3 index
                    "subject":  subject,
                }
            )
    print(f"MMLU subset loaded: {len(rows)} rows across {len(MMLU_SUBJECTS)} subjects")
    return rows


def evaluate_mmlu(model, tokenizer, mmlu_rows: list[dict], device: str = "cuda") -> dict:
    """
    Evaluate accuracy on multiple-choice MMLU questions.
    Uses log-likelihood scoring (standard approach).
    """
    choices_letters = ["A", "B", "C", "D"]
    correct = 0
    subject_correct: dict[str, list] = {s: [] for s in MMLU_SUBJECTS}

    model.eval()
    for row in tqdm(mmlu_rows, desc="MMLU evaluation"):
        prompt = (
            f"Question: {row['question']}\n"
            + "\n".join(
                f"{letter}. {choice}"
                for letter, choice in zip(choices_letters, row["choices"])
            )
            + "\nAnswer:"
        )

        best_choice, best_ll = -1, float("-inf")
        for idx, letter in enumerate(choices_letters):
            full = prompt + f" {letter}"
            enc = tokenizer(full, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**enc, labels=enc["input_ids"])
            ll = -out.loss.item()
            if ll > best_ll:
                best_ll, best_choice = ll, idx

        is_correct = int(best_choice == row["answer"])
        correct += is_correct
        subject_correct[row["subject"]].append(is_correct)

    per_subject = {s: float(np.mean(v)) for s, v in subject_correct.items() if v}
    overall = correct / len(mmlu_rows)

    print(f"\nMMlu accuracy: {overall:.4f}")
    for subj, acc in per_subject.items():
        print(f"  {subj}: {acc:.4f}")

    return {"overall_accuracy": overall, "per_subject": per_subject}


# ---------------------------------------------------------------------------
# Save / load results
# ---------------------------------------------------------------------------

def save_results(results: dict, filename: str) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    # Drop predictions list for the JSON (can be large)
    to_save = {k: v for k, v in results.items() if k not in ("predictions", "references")}
    with open(path, "w") as f:
        json.dump(to_save, f, indent=2)
    print(f"Results saved to {path}")


def print_comparison(baseline: dict, finetuned: dict) -> None:
    """Print a side-by-side comparison table."""
    print("\n" + "=" * 55)
    print(f"{'Metric':<20} {'Baseline':>15} {'Fine-tuned':>15}")
    print("-" * 55)
    for key in ("rouge_l_mean", "exact_match"):
        b = baseline.get(key, 0)
        f = finetuned.get(key, 0)
        delta = f - b
        sign = "+" if delta >= 0 else ""
        print(f"{key:<20} {b:>15.4f} {f:>15.4f}  ({sign}{delta:.4f})")
    print("=" * 55)
