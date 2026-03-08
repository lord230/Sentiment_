# Created By LORD

import torch
import random
import numpy as np

CSV_DATASET         = "dataset/combined_shuffled_Robust.csv"
CSV_DATASET_SARCASM = "dataset/Reddit_Sarcasm/train-balanced-sarcasm.csv"
CHECK_DIR           = "CheckPoints"
TEST_MODEL_PATH     = "SavedModel/best_model_epoch_16.pt"
VECTOR_DB_DIR       = "EmotionDB"



MODEL_NAME = "xlm-roberta-base"
MAX_LEN    = 256





BATCH_SIZE        = 32
GRAD_ACCUM_STEPS  = 2  

EPOCHS      = 40
LR          = 3e-5
MAX_SAMPLES = None
SEED        = 42


SARCASM_POS_WEIGHT = 1.0


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True 


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)
