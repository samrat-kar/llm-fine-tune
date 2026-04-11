"""
Central configuration for the SQL fine-tuning project.
Edit values here — all notebooks import from this file.
"""

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATASET_NAME = "b-mc2/sql-create-context"
DATASET_SPLIT_SEED = 42
VAL_SIZE = 0.05          # 5 % of training data used for validation
EVAL_SAMPLE_SIZE = 500   # rows used for baseline & post-FT evaluation

# ---------------------------------------------------------------------------
# Sequence lengths
# ---------------------------------------------------------------------------
MAX_SEQ_LENGTH = 512

# ---------------------------------------------------------------------------
# QLoRA / quantization
# ---------------------------------------------------------------------------
LOAD_IN_4BIT = True
BNB_4BIT_COMPUTE_DTYPE = "bfloat16"
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_USE_DOUBLE_QUANT = True

# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_BIAS = "none"
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
OUTPUT_DIR = "./outputs/qlora-sql"
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 2
PER_DEVICE_EVAL_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8      # effective batch = 16
LEARNING_RATE = 2e-4
LR_SCHEDULER_TYPE = "cosine"
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
LOGGING_STEPS = 25
EVAL_STEPS = 100
SAVE_STEPS = 100
FP16 = False
BF16 = True

# ---------------------------------------------------------------------------
# Weights & Biases
# ---------------------------------------------------------------------------
WANDB_PROJECT = "ReadyTensor-FineTune"
WANDB_RUN_NAME = "qwen2.5-1.5b-sql-lora-r16"

# ---------------------------------------------------------------------------
# Hugging Face Hub
# ---------------------------------------------------------------------------
# Set your HF username before publishing
HF_USERNAME = "samrat-kar"
HF_REPO_ID = f"{HF_USERNAME}/qwen2.5-1.5b-sql-qlora"
ADAPTER_DIR = "./outputs/qlora-sql/final_adapter"

# ---------------------------------------------------------------------------
# Paths (local)
# ---------------------------------------------------------------------------
PROCESSED_DATA_DIR = "./data/processed"
RESULTS_DIR = "./results"
