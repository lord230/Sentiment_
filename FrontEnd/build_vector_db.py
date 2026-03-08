# Created by LORD
#
# build_vector_db.py
#
# Builds three polarity sub-indexes from emotion_library.py seeds:
#   EmotionDB/positive.index
#   EmotionDB/negative.index
#   EmotionDB/neutral.index
#   EmotionDB/emotions.meta   (shared metadata for all three)
#
# Vector per seed:
#   [CLS (H) || mean_pool (H) || neg_p*50 || neu_p*50 || pos_p*50 || sarcasm*50]
#   dim = 2*768+4 = 1540  (xlm-roberta-base)
#
# Run once before test.py:
#   python build_vector_db.py
#
# Re-run whenever you edit emotion_library.py.

import os
import json
from collections import Counter, defaultdict

import torch
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

import config
from emotion_library import EMOTION_SEEDS
from emotion_db import build_vector, POLARITIES   # shared pooling + constants

DB_DIR      = getattr(config, "VECTOR_DB_DIR", "EmotionDB")
INDEX_PATHS = {p: os.path.join(DB_DIR, f"{p}.index") for p in POLARITIES}
META_PATH   = os.path.join(DB_DIR, "emotions.meta")

EMBED_BATCH = 64


# -------------------------------------------------
# BATCH ENCODER
# -------------------------------------------------

class ContextEncoder:
    """
    Encodes texts into full context vectors using build_vector().
    Seeds use uniform sentiment probs (0.333) and sarcasm=0.0
    because live task scores are only available at inference time.
    """

    def __init__(self, model_name: str, device: torch.device):
        self.device    = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def encode(self,
               texts: list[str],
               batch_size: int = EMBED_BATCH,
               max_len: int = config.MAX_LEN) -> np.ndarray:
        """Returns (N, 2H+4) float32 numpy array, each row L2-normalised."""
        all_vecs = []

        for i in tqdm(range(0, len(texts), batch_size), desc="  Embedding"):
            batch = texts[i : i + batch_size]

            tokens = self.tokenizer(
                batch,
                padding="max_length",
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )

            input_ids = tokens["input_ids"].to(self.device)
            mask      = tokens["attention_mask"].to(self.device)
            outputs   = self.model(input_ids=input_ids, attention_mask=mask)

            batch_vecs = []
            for b in range(len(batch)):
                vec = build_vector(
                    outputs.last_hidden_state[b].unsqueeze(0),
                    mask[b].unsqueeze(0),
                    neg_p=0.333, neu_p=0.333, pos_p=0.333,
                    sarcasm_score=0.0,
                )
                batch_vecs.append(vec)

            all_vecs.append(np.vstack(batch_vecs))

        return np.vstack(all_vecs)


# -------------------------------------------------
# BUILD
# -------------------------------------------------

def build():
    os.makedirs(DB_DIR, exist_ok=True)

    n_seeds = len(EMOTION_SEEDS)

    print(f"\n{'=' * 58}")
    print(f"  Building Emotion Vector DB  ({n_seeds} seeds)")
    print(f"{'=' * 58}")
    print(f"  Encoder  : {config.MODEL_NAME}")
    print(f"  Device   : {config.DEVICE}")
    print(f"  Vector   : [CLS || mean_pool || neg*50 || neu*50 || pos*50 || sarc*50]")
    print(f"  Indexes  : positive / negative / neutral  (sub-index per polarity)\n")

    texts      = [s[0] for s in EMOTION_SEEDS]
    emotions   = [s[1] for s in EMOTION_SEEDS]
    polarities = [s[2] for s in EMOTION_SEEDS]
    intensities= [s[3] for s in EMOTION_SEEDS]

    # ── Validate ─────────────────────────────────
    valid_pol = set(POLARITIES)
    valid_int = {"low", "medium", "high"}
    for i, (t, e, p, iv) in enumerate(zip(texts, emotions, polarities, intensities)):
        assert p  in valid_pol, f"Seed {i}: invalid polarity '{p}'"
        assert iv in valid_int, f"Seed {i}: invalid intensity '{iv}'"
        assert len(t.strip()) > 0, f"Seed {i}: empty text"

    # ── Polarity distribution check ──────────────
    pol_counts = Counter(polarities)
    print(f"  Polarity distribution:")
    for p in POLARITIES:
        n   = pol_counts[p]
        pct = 100 * n / n_seeds
        print(f"    {p:>10} : {n:>3} seeds  ({pct:.1f}%)")
    print()

    # ── Embed all seeds ──────────────────────────
    encoder = ContextEncoder(config.MODEL_NAME, config.DEVICE)
    all_vecs = encoder.encode(texts)                    # (N, 2H+4)
    dim = all_vecs.shape[1]
    print(f"\n  Vector dim : {dim}")

    # ── Build one sub-index per polarity ─────────
    # Group indices by polarity first
    pol_indices = defaultdict(list)
    for i, p in enumerate(polarities):
        pol_indices[p].append(i)

    meta = []
    local_id_map = {p: 0 for p in POLARITIES}   # local index counter per sub-index

    for p in POLARITIES:
        idxs = pol_indices[p]
        if not idxs:
            print(f"  WARNING: no seeds for polarity '{p}' — creating empty index")
            empty_index = faiss.IndexFlatIP(dim)
            faiss.write_index(empty_index, INDEX_PATHS[p])
            continue

        vecs  = all_vecs[idxs]                          # (K, dim)
        index = faiss.IndexFlatIP(dim)
        index.add(vecs)
        faiss.write_index(index, INDEX_PATHS[p])
        print(f"  [{p:>10}.index] : {index.ntotal} vectors  → {INDEX_PATHS[p]}")

    # ── Save unified metadata ────────────────────
    # local_index tracks per-polarity position for query → meta lookup
    pol_counter = defaultdict(int)
    for i in range(n_seeds):
        p = polarities[i]
        meta.append({
            "id":           i,
            "local_index":  pol_counter[p],
            "text":         texts[i],
            "emotion":      emotions[i],
            "polarity":     p,
            "intensity":    intensities[i],
            "neg_p":        0.333,
            "neu_p":        0.333,
            "pos_p":        0.333,
            "sarcasm_score":0.0,
        })
        pol_counter[p] += 1

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\n  Metadata saved : {META_PATH}  ({len(meta)} entries)")

    # ── Emotion distribution ─────────────────────
    emotion_counts = Counter(emotions)
    print(f"\n  Emotion distribution ({len(emotion_counts)} unique):")
    for emotion, count in sorted(emotion_counts.items()):
        bar = "█" * count
        print(f"    {emotion:<24} {count:>3}  {bar}")

    print(f"\n  Build complete. Run python test.py to start.")
    print(f"{'=' * 58}\n")


# -------------------------------------------------

if __name__ == "__main__":
    build()