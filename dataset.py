# Created By LORD

import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer

import config


# -------------------------------------------------
# SHARED TOKENIZER (loaded once, reused by both datasets)
# -------------------------------------------------

_tokenizer = None

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        # print(f"Loading tokenizer: {config.MODEL_NAME}")
        _tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    return _tokenizer


# -------------------------------------------------
# SENTIMENT DATASET
# -------------------------------------------------

class SentimentDataset(Dataset):
    """
    Dataset for sentiment classification.

    Expected CSV columns:
        Summary     — raw text
        Sentiment   — integer label: -1 (negative), 0 (neutral), 1 (positive)

    Labels are shifted internally to:
        0 = negative
        1 = neutral
        2 = positive
    """

    def __init__(
        self,
        csv_path,
        max_len=config.MAX_LEN,
        max_samples=config.MAX_SAMPLES,
    ):
        df = pd.read_csv(csv_path)

        # Drop rows with missing text or label
        df = df.dropna(subset=["Summary", "Sentiment"])

        if max_samples:
            df = df.sample(min(max_samples, len(df)), random_state=config.SEED)

        # Validate label range before shifting
        raw_labels = df["Sentiment"].astype(int)
        assert raw_labels.isin([-1, 0, 1]).all(), \
            "SentimentDataset: unexpected label values (expected -1, 0, 1)"

        self.texts  = df["Summary"].astype(str).tolist()
        self.labels = (raw_labels + 1).tolist()   # shift: -1/0/1 → 0/1/2
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = get_tokenizer()(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "sentiment":      torch.tensor(self.labels[idx], dtype=torch.long),
            "sarcasm":        torch.tensor(-1, dtype=torch.float),  # unavailable
        }


# -------------------------------------------------
# SARCASM DATASET
# -------------------------------------------------

class SarcasmDataset(Dataset):
    """
    Dataset for sarcasm detection.

    Expected CSV columns (Reddit balanced sarcasm):
        comment — raw text
        label   — integer: 0 (not sarcastic), 1 (sarcastic)
    """

    def __init__(
        self,
        csv_path,
        max_len=config.MAX_LEN,
        max_samples=config.MAX_SAMPLES,
    ):
        df = pd.read_csv(csv_path)

        # Drop rows with missing text or label
        df = df.dropna(subset=["comment", "label"])

        if max_samples:
            df = df.sample(min(max_samples, len(df)), random_state=config.SEED)

        # Validate label range
        labels = df["label"].astype(int)
        assert labels.isin([0, 1]).all(), \
            "SarcasmDataset: unexpected label values (expected 0 or 1)"

        self.texts   = df["comment"].astype(str).tolist()
        self.labels  = labels.tolist()
        self.max_len = max_len

        # Log class balance so you can set SARCASM_POS_WEIGHT correctly
        n_pos = sum(self.labels)
        n_neg = len(self.labels) - n_pos
        print(
            f"SarcasmDataset: {len(self.labels):,} samples | "
            f"positive (sarcastic): {n_pos:,} ({100*n_pos/len(self.labels):.1f}%) | "
            f"negative: {n_neg:,} ({100*n_neg/len(self.labels):.1f}%)"
        )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = get_tokenizer()(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "sentiment":      torch.tensor(-1, dtype=torch.long),          # unavailable
            "sarcasm":        torch.tensor(self.labels[idx], dtype=torch.float),
        }


# -------------------------------------------------
# QUICK SANITY CHECK  (python dataset.py)
# -------------------------------------------------

if __name__ == "__main__":

    import random
    import numpy as np
    
    print("=" * 50)
    print("Sentiment Dataset")
    print("=" * 50)
    sent_dataset = SentimentDataset(config.CSV_DATASET)
    print(f"Size: {len(sent_dataset):,}")
    x = random.randint(0, len(sent_dataset))
    item = sent_dataset[x]
    print(f"input_ids shape  : {item['input_ids'].shape}")
    print(f"attention_mask   : {item['attention_mask'].shape}")
    print(f"sentiment label  : {item['sentiment'].item()}")

    print()
    print("=" * 50)
    print("Sarcasm Dataset")
    print("=" * 50)
    sarc_dataset = SarcasmDataset(config.CSV_DATASET_SARCASM)
    print(f"Size: {len(sarc_dataset):,}")

    item = sarc_dataset[x]
    print(f"input_ids shape  : {item['input_ids'].shape}")
    print(f"attention_mask   : {item['attention_mask'].shape}")
    print(f"sarcasm label    : {item['sarcasm'].item()}")