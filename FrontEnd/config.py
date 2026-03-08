# Created By LORD

import torch
import random
import numpy as np

# -------------------------------------------------
# PATHS
# -------------------------------------------------

CSV_DATASET         = "dataset/combined_shuffled_Robust.csv"
CSV_DATASET_SARCASM = "dataset/Reddit_Sarcasm/train-balanced-sarcasm.csv"
CHECK_DIR           = "CheckPoints"
TEST_MODEL_PATH     = "SavedModel/best_model_epoch_16.pt"
VECTOR_DB_DIR       = "EmotionDB"

# -------------------------------------------------
# MODEL
# -------------------------------------------------

MODEL_NAME = "xlm-roberta-base"
MAX_LEN    = 256

# -------------------------------------------------
# TRAINING
# -------------------------------------------------

# RTX 3060 12GB VRAM breakdown (approximate):
#   xlm-roberta-base weights  ~1.1 GB
#   Optimizer states (AdamW)  ~2.2 GB  (2x model for fp32 master weights)
#   Activations + gradients   ~3.5 GB  (at batch size 32, seq len 256)
#   AMP overhead              ~0.5 GB
#   ─────────────────────────────────
#   Total at BATCH_SIZE=32    ~7.3 GB  ← safe headroom under 12 GB
#
# If you hit OOM, drop to BATCH_SIZE = 24 first, then 16.
# Use GRAD_ACCUM_STEPS to keep the effective batch size large:
#   effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS
#   32 * 2 = 64  (recommended baseline)

BATCH_SIZE        = 32
GRAD_ACCUM_STEPS  = 2     # effective batch size = 64

EPOCHS      = 40
LR          = 3e-5
MAX_SAMPLES = None
SEED        = 42

# -------------------------------------------------
# SARCASM LOSS WEIGHT
# -------------------------------------------------
# Counteracts class imbalance in the sarcasm dataset.
# Formula: (1 - positive_fraction) / positive_fraction
#
# The Reddit balanced sarcasm dataset is ~50/50, so 1.0 is correct.
# If you switch to an imbalanced dataset, recalculate:
#   e.g. 25% sarcastic  → SARCASM_POS_WEIGHT = 0.75 / 0.25 = 3.0
#   e.g. 10% sarcastic  → SARCASM_POS_WEIGHT = 0.90 / 0.10 = 9.0

SARCASM_POS_WEIGHT = 1.0

# -------------------------------------------------
# DEVICE
# -------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# RTX 3060 uses Ampere architecture — tf32 gives ~2x matmul throughput
# with no meaningful accuracy loss over fp32
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True  # auto-tune kernels for fixed input sizes

# -------------------------------------------------
# REPRODUCIBILITY
# -------------------------------------------------

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)
