# %% [markdown]
# # Notebook 3 — QLoRA Fine-Tuning
#
# **Goal**: Fine-tune `Qwen/Qwen2.5-1.5B-Instruct` on the SQL-generation task
# using **QLoRA** (4-bit quantization + LoRA) with the Hugging Face PEFT +
# TRL stack.
#
# ## Configuration summary
# | Parameter | Value |
# |-----------|-------|
# | LoRA rank (r) | 16 |
# | LoRA alpha | 32 |
# | LoRA dropout | 0.05 |
# | Target modules | q/k/v/o/gate/up/down proj |
# | Quantization | 4-bit NF4 |
# | Learning rate | 2e-4 (cosine schedule) |
# | Effective batch size | 16 (4 × 4 grad accum) |
# | Epochs | 3 |
# | Max seq length | 512 |
# | Experiment tracking | Weights & Biases |

# %% [markdown]
# ## Setup

# %%
import sys, os
sys.path.insert(0, os.path.abspath(".."))

import torch
import wandb
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

from src.config import (
    # model
    BASE_MODEL_ID,
    # quantisation
    LOAD_IN_4BIT, BNB_4BIT_COMPUTE_DTYPE, BNB_4BIT_QUANT_TYPE, BNB_USE_DOUBLE_QUANT,
    # LoRA
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_BIAS, LORA_TARGET_MODULES,
    # training
    OUTPUT_DIR, NUM_TRAIN_EPOCHS,
    PER_DEVICE_TRAIN_BATCH_SIZE, PER_DEVICE_EVAL_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS, LEARNING_RATE, LR_SCHEDULER_TYPE,
    WARMUP_RATIO, WEIGHT_DECAY, MAX_GRAD_NORM,
    LOGGING_STEPS, EVAL_STEPS, SAVE_STEPS, FP16, BF16,
    MAX_SEQ_LENGTH,
    # W&B
    WANDB_PROJECT, WANDB_RUN_NAME,
    # paths
    PROCESSED_DATA_DIR, ADAPTER_DIR,
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {device}")
if device == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

# %% [markdown]
# ## 1. Initialise Weights & Biases

# %%
# If running on Colab: wandb.login() will prompt for your API key.
# Alternatively, set the WANDB_API_KEY environment variable.
wandb.init(
    project=WANDB_PROJECT,
    name=WANDB_RUN_NAME,
    config={
        "base_model":   BASE_MODEL_ID,
        "lora_r":       LORA_R,
        "lora_alpha":   LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "learning_rate": LEARNING_RATE,
        "epochs":       NUM_TRAIN_EPOCHS,
        "batch_size":   PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
        "max_seq_len":  MAX_SEQ_LENGTH,
        "quantization": "4-bit NF4",
    },
)
print(f"W&B run: {wandb.run.url}")

# %% [markdown]
# ## 2. Load tokenizer

# %%
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"   # required for SFTTrainer with left-pad models
print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size:,}")

# %% [markdown]
# ## 3. Load base model with 4-bit quantization

# %%
bnb_config = BitsAndBytesConfig(
    load_in_4bit=LOAD_IN_4BIT,
    bnb_4bit_compute_dtype=getattr(torch, BNB_4BIT_COMPUTE_DTYPE),
    bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
    bnb_4bit_use_double_quant=BNB_USE_DOUBLE_QUANT,
)

print(f"Loading base model: {BASE_MODEL_ID}")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Prepare for k-bit training (casts layer norms to float32, enables grad checkpointing)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
print("Model ready for k-bit training.")

# %% [markdown]
# ## 4. Add LoRA adapters

# %%
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias=LORA_BIAS,
    task_type="CAUSAL_LM",
    target_modules=LORA_TARGET_MODULES,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# %% [markdown]
# ## 5. Load processed dataset

# %%
dataset = load_from_disk(PROCESSED_DATA_DIR)
print(dataset)
print(f"\nSample formatted text:\n{dataset['train'][0]['text'][:500]}")

# %% [markdown]
# ## 6. Training arguments

# %%
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_TRAIN_EPOCHS,

    # Batch & grad accum
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

    # Optimiser
    learning_rate=LEARNING_RATE,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=WEIGHT_DECAY,
    max_grad_norm=MAX_GRAD_NORM,
    optim="paged_adamw_32bit",

    # Precision
    fp16=FP16,
    bf16=BF16,

    # Logging & saving
    logging_steps=LOGGING_STEPS,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    # W&B
    report_to="wandb",
    run_name=WANDB_RUN_NAME,

    # Misc
    dataloader_num_workers=0,
    seed=42,

    # SFT-specific
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    packing=True,
)

# %% [markdown]
# ## 7. Create SFTTrainer and train
#
# `SFTTrainer` from TRL handles:
# - Packing short sequences together to maximise GPU utilisation
# - Applying the dataset's `text` column as training targets
# - Gradient checkpointing for memory efficiency

# %%
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)

# %%
print("Starting training...")
train_result = trainer.train()

# %%
# Print final training stats
print("\nTraining complete!")
print(f"  Total steps     : {train_result.global_step}")
print(f"  Training loss   : {train_result.training_loss:.4f}")
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)

# %% [markdown]
# ## 8. Save the LoRA adapter

# %%
os.makedirs(ADAPTER_DIR, exist_ok=True)
trainer.model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
print(f"LoRA adapter saved to: {ADAPTER_DIR}")

# List saved files
for f in os.listdir(ADAPTER_DIR):
    size_kb = os.path.getsize(os.path.join(ADAPTER_DIR, f)) / 1024
    print(f"  {f:40s} {size_kb:8.1f} KB")

# %% [markdown]
# ## 9. Visualise training curves
#
# Training/eval loss is logged to W&B automatically. We also pull the
# local log file for a quick inline plot.

# %%
import json
import matplotlib.pyplot as plt

log_path = os.path.join(OUTPUT_DIR, "trainer_state.json")
if os.path.exists(log_path):
    with open(log_path) as f:
        state = json.load(f)

    log_history = state.get("log_history", [])
    train_steps = [e["step"] for e in log_history if "loss" in e]
    train_loss  = [e["loss"] for e in log_history if "loss" in e]
    eval_steps  = [e["step"] for e in log_history if "eval_loss" in e]
    eval_loss   = [e["eval_loss"] for e in log_history if "eval_loss" in e]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_steps, train_loss, label="train loss",      color="steelblue")
    ax.plot(eval_steps,  eval_loss,  label="validation loss", color="coral", marker="o")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training & validation loss")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"../results/training_loss_curve.png", dpi=120, bbox_inches="tight")
    plt.show()
else:
    print("trainer_state.json not found — check W&B for loss curves.")

# %%
wandb.finish()
print("W&B run finished. Check your project at https://wandb.ai")
print(f"\nAdapter saved at: {ADAPTER_DIR}")
print("Proceed to Notebook 4 for post-fine-tuning evaluation.")
