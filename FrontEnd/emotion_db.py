# Created by LORD
#
# emotion_db.py
#
# FAISS emotion DB with three polarity sub-indexes:
#   positive.index / negative.index / neutral.index
#
# At query time the dominant sentiment selects the sub-index,
# so a strongly-positive query never competes against negative seeds.
#
# Vector format (per entry):
#   [CLS (H) || mean_pool (H) || neg_p*S || neu_p*S || pos_p*S || sarcasm*S]
#   S = SCALAR_SCALE (50)   dim = 2*768+4 = 1540

import os
import json

import torch
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel

import config

DB_DIR   = getattr(config, "VECTOR_DB_DIR", "EmotionDB")


POLARITIES   = ("positive", "negative", "neutral")
INDEX_PATHS  = {p: os.path.join(DB_DIR, f"{p}.index") for p in POLARITIES}
META_PATH    = os.path.join(DB_DIR, "emotions.meta")   # single shared metadata

SCALAR_SCALE = 50

INTENSITY_RANK = {"low": 1, "medium": 2, "high": 3}

INTENSITY_LABEL = {
    "low":    "Mild",
    "medium": "Moderate",
    "high":   "Strong",
}

POLARITY_EMOJI = {
    "positive": "🟢",
    "negative": "🔴",
    "neutral":  "⚪",
}


# -------------------------------------------------
# SHARED ENCODER SINGLETON
# -------------------------------------------------

_encoder_model     = None
_encoder_tokenizer = None


def _get_encoder():
    global _encoder_model, _encoder_tokenizer
    if _encoder_model is None:
        print(f"  Loading encoder : {config.MODEL_NAME}")
        _encoder_tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        _encoder_model     = AutoModel.from_pretrained(config.MODEL_NAME).to(config.DEVICE)
        _encoder_model.eval()
    return _encoder_model, _encoder_tokenizer


# -------------------------------------------------
# POOLING + VECTOR CONSTRUCTION
# -------------------------------------------------

def _mean_pool(last_hidden_state: torch.Tensor,
               attention_mask: torch.Tensor) -> torch.Tensor:
    """Masked mean pool. Ignores padding tokens. Returns (B, H)."""
    mask_exp   = attention_mask.unsqueeze(-1).float()
    sum_hidden = (last_hidden_state * mask_exp).sum(dim=1)
    sum_mask   = mask_exp.sum(dim=1).clamp(min=1e-9)
    return sum_hidden / sum_mask


def build_vector(last_hidden_state: torch.Tensor,
                 attention_mask: torch.Tensor,
                 neg_p: float = 0.333,
                 neu_p: float = 0.333,
                 pos_p: float = 0.333,
                 sarcasm_score: float = 0.0) -> np.ndarray:
    """
    Build and L2-normalise the full context vector:
        [CLS (H) || mean_pool (H) || neg_p*S || neu_p*S || pos_p*S || sarcasm*S]

    SCALAR_SCALE=50 ensures the sentiment features contribute meaningfully
    to cosine distance instead of being drowned out by the 768-dim embeddings.

    Returns: (1, 2H+4) float32 numpy array, L2-normalised.
    """
    cls  = last_hidden_state[:, 0]
    mean = _mean_pool(last_hidden_state, attention_mask)

    scalars = torch.tensor(
        [[neg_p * SCALAR_SCALE,
          neu_p * SCALAR_SCALE,
          pos_p * SCALAR_SCALE,
          sarcasm_score * SCALAR_SCALE]],
        dtype=torch.float32,
        device=last_hidden_state.device,
    )

    vec = torch.cat([cls, mean, scalars], dim=1).cpu().float().numpy()
    faiss.normalize_L2(vec)
    return vec


@torch.no_grad()
def embed_text(text: str,
               neg_p: float = 0.333,
               neu_p: float = 0.333,
               pos_p: float = 0.333,
               sarcasm_score: float = 0.0) -> np.ndarray:
    """Tokenise + encode a single string into a full context vector."""
    enc_model, enc_tok = _get_encoder()

    tokens = enc_tok(
        text,
        padding="max_length",
        truncation=True,
        max_length=config.MAX_LEN,
        return_tensors="pt",
    )

    input_ids = tokens["input_ids"].to(config.DEVICE)
    mask      = tokens["attention_mask"].to(config.DEVICE)
    outputs   = enc_model(input_ids=input_ids, attention_mask=mask)

    return build_vector(
        outputs.last_hidden_state, mask,
        neg_p=neg_p, neu_p=neu_p,
        pos_p=pos_p, sarcasm_score=sarcasm_score,
    )


# -------------------------------------------------
# SUB-INDEX SELECTION
# -------------------------------------------------

def _select_polarity(neg_p: float, neu_p: float, pos_p: float) -> str:
    """Pick which sub-index to query based on dominant sentiment prob."""
    scores = {"positive": pos_p, "negative": neg_p, "neutral": neu_p}
    return max(scores, key=scores.get)


# -------------------------------------------------
# EMOTION DB
# -------------------------------------------------

class EmotionDB:
    """
    Three-polarity FAISS sub-index emotion search.

    At query time the dominant sentiment probability selects the matching
    sub-index (positive / negative / neutral), so results are always
    semantically coherent with the detected sentiment.

    Sub-index selection rules:
        pos_p dominant  → query positive.index  (Joy, Admiration, Love …)
        neg_p dominant  → query negative.index  (Anger, Grief, Sarcasm …)
        neu_p dominant  → query neutral.index   (Calm, Indifference …)
    """

    def __init__(self):
        missing = [p for p in POLARITIES
                   if not os.path.exists(INDEX_PATHS[p])]
        if missing or not os.path.exists(META_PATH):
            raise FileNotFoundError(
                "Emotion DB not found. Run  python build_vector_db.py  first.\n"
                f"Missing: {[INDEX_PATHS[p] for p in missing] + ([] if os.path.exists(META_PATH) else [META_PATH])}"
            )

        self.indexes = {p: faiss.read_index(INDEX_PATHS[p]) for p in POLARITIES}

        with open(META_PATH, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self._pol_entries = {p: [] for p in POLARITIES}
        for i, entry in enumerate(self.meta):
            self._pol_entries[entry["polarity"]].append((i, entry))

        for p in POLARITIES:
            n = self.indexes[p].ntotal
            print(f"  EmotionDB [{p:>8}] : {n} vectors  dim={self.indexes[p].d}")

    # ----------------------------------------------------------
    # QUERY
    # ----------------------------------------------------------

    def query(self,
              text: str,
              top_k: int = 5,
              neg_p: float = 0.333,
              neu_p: float = 0.333,
              pos_p: float = 0.333,
              sarcasm_score: float = 0.0) -> list[dict]:
        """
        Query the appropriate polarity sub-index and return top_k results.

        Always pass live neg_p / neu_p / pos_p / sarcasm_score from
        predict_sentiment() so the query vector matches stored format.
        """
        polarity = _select_polarity(neg_p, neu_p, pos_p)
        index    = self.indexes[polarity]
        entries  = self._pol_entries[polarity]

        vec = embed_text(
            text,
            neg_p=neg_p, neu_p=neu_p,
            pos_p=pos_p, sarcasm_score=sarcasm_score,
        )

        k            = min(top_k, index.ntotal)
        similarities, local_indices = index.search(vec, k)

        results = []
        for sim, local_idx in zip(similarities[0], local_indices[0]):
            if local_idx < 0 or local_idx >= len(entries):
                continue
            _, entry = entries[local_idx]
            results.append({
                "emotion":    entry["emotion"],
                "polarity":   entry["polarity"],
                "intensity":  entry["intensity"],
                "similarity": float(sim),
                "seed_text":  entry["text"],
            })

        return results

    # ----------------------------------------------------------
    # ANALYSE
    # ----------------------------------------------------------

    def analyse(self,
                text: str,
                top_k: int = 10,
                neg_p: float = 0.333,
                neu_p: float = 0.333,
                pos_p: float = 0.333,
                sarcasm_score: float = 0.0) -> dict:
        """Query top_k neighbours and aggregate into a structured report."""
        hits = self.query(
            text, top_k=top_k,
            neg_p=neg_p, neu_p=neu_p,
            pos_p=pos_p, sarcasm_score=sarcasm_score,
        )

        emotion_scores   = {}
        polarity_scores  = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        intensity_scores = {"low": 0.0, "medium": 0.0, "high": 0.0}

        for hit in hits:
            w = hit["similarity"] * INTENSITY_RANK[hit["intensity"]]
            emotion_scores[hit["emotion"]]     = emotion_scores.get(hit["emotion"], 0.0) + w
            polarity_scores[hit["polarity"]]  += w
            intensity_scores[hit["intensity"]] += hit["similarity"]

        total = sum(emotion_scores.values()) or 1.0
        emotion_pct = {
            k: round(v / total * 100, 1)
            for k, v in sorted(emotion_scores.items(), key=lambda x: -x[1])
        }

        return {
            "primary_emotion":    max(emotion_scores, key=emotion_scores.get),
            "emotion_breakdown":  emotion_pct,
            "dominant_polarity":  max(polarity_scores,  key=polarity_scores.get),
            "dominant_intensity": max(intensity_scores, key=intensity_scores.get),
            "top_matches":        hits,
        }

    # ----------------------------------------------------------
    # ADD  (permanently expand the DB)
    # ----------------------------------------------------------

    def add(self,
            text: str,
            emotion: str,
            polarity: str,
            intensity: str,
            neg_p: float = 0.333,
            neu_p: float = 0.333,
            pos_p: float = 0.333,
            sarcasm_score: float = 0.0):
        """Embed a new sentence and append it to the correct polarity sub-index."""
        assert polarity  in POLARITIES,               "Invalid polarity"
        assert intensity in ("low", "medium", "high"), "Invalid intensity"

        vec = embed_text(
            text,
            neg_p=neg_p, neu_p=neu_p,
            pos_p=pos_p, sarcasm_score=sarcasm_score,
        )

        # Append to the correct sub-index
        self.indexes[polarity].add(vec)

        new_entry = {
            "id":           len(self.meta),
            "text":         text,
            "emotion":      emotion,
            "polarity":     polarity,
            "intensity":    intensity,
            "neg_p":        round(neg_p, 4),
            "neu_p":        round(neu_p, 4),
            "pos_p":        round(pos_p, 4),
            "sarcasm_score":round(sarcasm_score, 4),
        }
        self.meta.append(new_entry)
        self._pol_entries[polarity].append((len(self.meta) - 1, new_entry))

        faiss.write_index(self.indexes[polarity], INDEX_PATHS[polarity])
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2, ensure_ascii=False)

        print(f"  Added [{polarity} | {emotion} | {intensity}] '{text[:55]}'")