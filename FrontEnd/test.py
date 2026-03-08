# Created by LORD
#
# test.py

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

import config
from Models import Model_aware
from emotion_db import EmotionDB, INTENSITY_LABEL, POLARITY_EMOJI


# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------

def load_model(path):
    model = Model_aware.SarcasmAwareSentimentTransformer()
    checkpoint = torch.load(path, map_location=config.DEVICE)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        epoch      = checkpoint.get("epoch", "?")
        best_score = checkpoint.get("best_score", "?")
        score_str  = f"{best_score:.4f}" if isinstance(best_score, float) else str(best_score)
        print(f"  Checkpoint : epoch {epoch} | best score {score_str}")
    else:
        model.load_state_dict(checkpoint)

    model.to(config.DEVICE)
    model.eval()
    return model


def load_tokenizer():
    return AutoTokenizer.from_pretrained(config.MODEL_NAME)


# -------------------------------------------------
# LABEL MAPS
# -------------------------------------------------

SENTIMENT_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}

NUANCED_EMOJI = {
    "Strongly Positive":  "🟢",
    "Positive":           "🟢",
    "Slightly Positive":  "🟢",
    "Completely Neutral": "⚪",
    "Mixed / Uncertain":  "🟠",
    "Slightly Negative":  "🔴",
    "Negative":           "🔴",
    "Strongly Negative":  "🔴",
    "Sarcastic Positive": "😏",
    "Sarcastic Negative": "😒",
}


# -------------------------------------------------
# COMBINED SCORE + NUANCED LABEL
# -------------------------------------------------

def compute_combined_score(neg_p, neu_p, pos_p):
    return round((pos_p - neg_p) * (1.0 - neu_p), 4)


def nuanced_label(score, neu_p, sarcasm_score):
    sarcastic = sarcasm_score > 0.5

    if   score >=  0.75: base = "Strongly Positive"
    elif score >=  0.40: base = "Positive"
    elif score >=  0.15: base = "Slightly Positive"
    elif score >  -0.15: base = "Completely Neutral" if neu_p > 0.6 else "Mixed / Uncertain"
    elif score >  -0.40: base = "Slightly Negative"
    elif score >  -0.75: base = "Negative"
    else:                base = "Strongly Negative"

    if sarcastic:
        if "Positive" in base: return "Sarcastic Positive"
        if "Negative" in base: return "Sarcastic Negative"

    return base


# -------------------------------------------------
# PREDICT
# -------------------------------------------------

def predict_sentiment(text, model, tokenizer):
    tokens = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=config.MAX_LEN,
        return_tensors="pt",
    )

    input_ids = tokens["input_ids"].to(config.DEVICE)
    mask      = tokens["attention_mask"].to(config.DEVICE)

    with torch.no_grad():
        sentiment_logits, sarcasm_logits = model(input_ids, mask)

    sent_probs    = F.softmax(sentiment_logits, dim=1).squeeze(0)
    neg_p         = sent_probs[0].item()
    neu_p         = sent_probs[1].item()
    pos_p         = sent_probs[2].item()
    base_idx      = sent_probs.argmax().item()
    confidence    = sent_probs[base_idx].item() * 100
    sarcasm_score = torch.sigmoid(sarcasm_logits).squeeze().item()
    combined      = compute_combined_score(neg_p, neu_p, pos_p)

    return {
        "base_label":      SENTIMENT_MAP[base_idx],
        "final_label":     nuanced_label(combined, neu_p, sarcasm_score),
        "sentiment_probs": {
            "Negative": round(neg_p * 100, 2),
            "Neutral":  round(neu_p * 100, 2),
            "Positive": round(pos_p * 100, 2),
        },
        "confidence":     confidence,
        "combined_score": combined,
        "sarcasm":        "Sarcastic" if sarcasm_score > 0.70 else "Not Sarcastic",
        "sarcasm_score":  sarcasm_score,
        "_neg_p":         neg_p,
        "_neu_p":         neu_p,
        "_pos_p":         pos_p,
        "_sarcasm_score": sarcasm_score,
    }


# -------------------------------------------------
# DISPLAY HELPERS
# -------------------------------------------------

DIVIDER_WIDE = "═" * 58
DIVIDER_THIN = "─" * 58


def score_bar(score):
    mid    = 10
    filled = min(int(abs(score) * mid), mid)
    left   = [" "] * mid
    right  = [" "] * mid
    if score >= 0:
        for i in range(filled):       right[i]          = "█"
    else:
        for i in range(filled):       left[mid - 1 - i] = "█"
    return "[" + "".join(left) + "|" + "".join(right) + "]"


def prob_bar(prob):
    return "█" * int(prob / 5)


def sim_bar(sim, width=10):
    filled = round(sim * width)
    return "█" * filled + "░" * (width - filled)


# -------------------------------------------------
# DISPLAY
# -------------------------------------------------

def display(sr, er):
    emoji = NUANCED_EMOJI.get(sr["final_label"], "🔵")

    print(f"\n{DIVIDER_WIDE}")
    print(f"  SENTIMENT ANALYSIS")
    print(DIVIDER_THIN)
    print(f"  Result        : {emoji}  {sr['final_label']}")
    print(f"  Base Model    : {sr['base_label']}  ({sr['confidence']:.1f}% confident)")
    print()
    print(f"  Probabilities :")
    for label, prob in sr["sentiment_probs"].items():
        print(f"    {label:<10} {prob:>6.2f}%  {prob_bar(prob)}")
    print()
    print(f"  Combined Score: {sr['combined_score']:+.4f}  {score_bar(sr['combined_score'])}")
    print(f"                  -1.0 ← negative  |  positive → +1.0")
    print()
    print(f"  Sarcasm       : {sr['sarcasm']}  (score: {sr['sarcasm_score']:.4f})")

    print(f"\n{DIVIDER_THIN}")
    print(f"  EMOTION ANALYSIS")
    print(DIVIDER_THIN)

    pol_emoji = POLARITY_EMOJI.get(er["dominant_polarity"], "")
    int_label = INTENSITY_LABEL.get(er["dominant_intensity"], er["dominant_intensity"])

    print(f"  Primary Emotion   : {er['primary_emotion']}")
    print(f"  Dominant Polarity : {pol_emoji}  {er['dominant_polarity'].capitalize()}")
    print(f"  Dominant Intensity: {int_label}")
    print()
    print(f"  Emotion Breakdown :")
    for emotion, pct in er["emotion_breakdown"].items():
        bar = "█" * int(pct / 5)
        print(f"    {emotion:<22} {pct:>5.1f}%  {bar}")

    # ── Top 5 neighbouring emotions with cosine similarity ──
    print()
    print(f"  Top 5 Neighbouring Emotions :")
    seen = set()
    rank = 1
    for hit in er["top_matches"]:
        if rank > 5:
            break
        key = hit["emotion"]
        if key in seen:
            continue
        seen.add(key)
        sbar = sim_bar(hit["similarity"])
        pol  = POLARITY_EMOJI.get(hit["polarity"], "")
        print(f"    {rank}. {hit['emotion']:<20} {pol}  "
              f"[{sbar}] {hit['similarity']:.4f}")
        rank += 1

    print(DIVIDER_WIDE)


# -------------------------------------------------
# MAIN
# -------------------------------------------------

def main():
    print(f"\n{DIVIDER_WIDE}")
    print(f"  Loading sentiment model...")
    model     = load_model(config.TEST_MODEL_PATH)
    tokenizer = load_tokenizer()

    print(f"\n  Loading emotion vector DB...")
    try:
        emotion_db = EmotionDB()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        return

    print(f"\n  Model      : {config.MODEL_NAME}")
    print(f"  Device     : {config.DEVICE}")
    print(f"  Checkpoint : {config.TEST_MODEL_PATH}")
    print(f"\n  Ready. Type 'exit' to quit.")
    print(DIVIDER_WIDE)

    while True:
        try:
            text = input("\nEnter text → ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not text:
            continue

        if text.lower() == "exit":
            print("Exiting.")
            break

        sr = predict_sentiment(text, model, tokenizer)

        er = emotion_db.analyse(
            text,
            top_k=15,                      
            neg_p=sr["_neg_p"],
            neu_p=sr["_neu_p"],
            pos_p=sr["_pos_p"],
            sarcasm_score=sr["_sarcasm_score"],
        )

        display(sr, er)


# -------------------------------------------------

if __name__ == "__main__":
    main()