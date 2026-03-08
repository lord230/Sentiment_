# Created by LORD

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

import config
from Models import Model_aware


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
        print(f"Checkpoint   : epoch {epoch} | best score {best_score:.4f}"
              if isinstance(best_score, float) else
              f"Checkpoint   : epoch {epoch}")
    else:
        model.load_state_dict(checkpoint)

    model.to(config.DEVICE)
    model.eval()
    return model


# -------------------------------------------------
# TOKENIZER
# -------------------------------------------------

def load_tokenizer():
    return AutoTokenizer.from_pretrained(config.MODEL_NAME)


# -------------------------------------------------
# LABEL MAPS
# -------------------------------------------------

SENTIMENT_MAP = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}

SENTIMENT_EMOJI = {
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
# COMBINED SCORE  [-1, +1]
# -------------------------------------------------
# Formula:
#   raw      = P(positive) - P(negative)
#   dampened = raw * (1 - P(neutral))
#
# High neutral probability pulls the score toward 0,
# reflecting genuine ambiguity in the text.
#
# If sarcasm is detected AND the signal is directional
# (|dampened| > 0.2), the polarity is flipped because
# sarcastic praise is actually negative and vice versa.

def compute_combined_score(neg_p, neu_p, pos_p, sarcasm_score):
    raw      = pos_p - neg_p
    dampened = raw * (1.0 - neu_p)

    if sarcasm_score > 0.5 and abs(dampened) > 0.2:
        dampened = -dampened

    return round(dampened, 4)


# -------------------------------------------------
# NUANCED LABEL
# -------------------------------------------------
#
#  Combined score  →  Label
#  ────────────────────────────────────────────────
#  [ 0.75,  1.00]  →  Strongly Positive
#  [ 0.40,  0.75)  →  Positive
#  [ 0.15,  0.40)  →  Slightly Positive
#  (-0.15,  0.15)  →  Completely Neutral  (if neu_p > 0.6)
#                  →  Mixed / Uncertain   (otherwise)
#  (-0.40, -0.15]  →  Slightly Negative
#  (-0.75, -0.40]  →  Negative
#  [-1.00, -0.75]  →  Strongly Negative
#
#  Sarcasm override (sarcasm_score > 0.5):
#    any Positive family  →  Sarcastic Positive
#    any Negative family  →  Sarcastic Negative

def nuanced_label(score, neu_p, sarcasm_score):
    sarcastic = sarcasm_score > 0.5

    if score >= 0.75:
        base = "Strongly Positive"
    elif score >= 0.40:
        base = "Positive"
    elif score >= 0.15:
        base = "Slightly Positive"
    elif score > -0.15:
        base = "Completely Neutral" if neu_p > 0.6 else "Mixed / Uncertain"
    elif score > -0.40:
        base = "Slightly Negative"
    elif score > -0.75:
        base = "Negative"
    else:
        base = "Strongly Negative"

    if sarcastic:
        if "Positive" in base:
            return "Sarcastic Positive"
        if "Negative" in base:
            return "Sarcastic Negative"

    return base


# -------------------------------------------------
# PREDICTION
# -------------------------------------------------

def predict(text, model, tokenizer):

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

    # ── Sentiment probabilities ──────────────────
    sent_probs = F.softmax(sentiment_logits, dim=1).squeeze(0)
    neg_p = sent_probs[0].item()
    neu_p = sent_probs[1].item()
    pos_p = sent_probs[2].item()

    base_idx   = sent_probs.argmax().item()
    base_label = SENTIMENT_MAP[base_idx]
    confidence = sent_probs[base_idx].item() * 100

    sent_probs_dict = {
        "Negative": round(neg_p * 100, 2),
        "Neutral":  round(neu_p * 100, 2),
        "Positive": round(pos_p * 100, 2),
    }

    # ── Sarcasm ──────────────────────────────────
    sarcasm_score = torch.sigmoid(sarcasm_logits).squeeze().item()
    sarcasm_label = "Sarcastic" if sarcasm_score > 0.5 else "Not Sarcastic"

    # ── Combined score & nuanced label ───────────
    combined_score = compute_combined_score(neg_p, neu_p, pos_p, sarcasm_score)
    final_label    = nuanced_label(combined_score, neu_p, sarcasm_score)

    return {
        "base_label":      base_label,
        "final_label":     final_label,
        "sentiment_probs": sent_probs_dict,
        "confidence":      confidence,
        "combined_score":  combined_score,
        "sarcasm":         sarcasm_label,
        "sarcasm_score":   sarcasm_score,
    }


# -------------------------------------------------
# DISPLAY HELPERS
# -------------------------------------------------

DIVIDER = "─" * 52

def score_bar(score):
    """
    Renders a [-1, +1] score as a 20-char bar with a centre pivot.

    score =  0.45  →  [          |████████   ]
    score = -0.70  →  [    ██████████|        ]
    score =  0.00  →  [          |            ]
    """
    total  = 20
    mid    = total // 2
    filled = min(int(abs(score) * mid), mid)

    left  = [" "] * mid
    right = [" "] * mid

    if score >= 0:
        for i in range(filled):
            right[i] = "█"
    else:
        for i in range(filled):
            left[mid - 1 - i] = "█"

    return "[" + "".join(left) + "|" + "".join(right) + "]"


def prob_bar(prob):
    """One block per 5%, max 20 blocks."""
    return "█" * int(prob / 5)


# -------------------------------------------------
# DISPLAY
# -------------------------------------------------

def display(result):
    emoji = SENTIMENT_EMOJI.get(result["final_label"], "🔵")

    print(f"\n{DIVIDER}")
    print(f"  Result        : {emoji}  {result['final_label']}")
    print(f"  Base Model    : {result['base_label']}  ({result['confidence']:.1f}% confident)")
    print()
    print(f"  Probabilities :")
    for label, prob in result["sentiment_probs"].items():
        print(f"    {label:<10} {prob:>6.2f}%  {prob_bar(prob)}")
    print()
    print(f"  Combined Score: {result['combined_score']:+.4f}  {score_bar(result['combined_score'])}")
    print(f"                  -1.0 ← negative  |  positive → +1.0")
    print()
    print(f"  Sarcasm       : {result['sarcasm']}  (score: {result['sarcasm_score']:.4f})")
    print(DIVIDER)


# -------------------------------------------------
# INTERACTIVE LOOP
# -------------------------------------------------

def main():
    print("\nLoading model...")
    model     = load_model(config.TEST_MODEL_PATH)
    tokenizer = load_tokenizer()

    print(f"Model        : {config.MODEL_NAME}")
    print(f"Device       : {config.DEVICE}")
    print(f"Checkpoint   : {config.TEST_MODEL_PATH}")
    print("\nModel loaded. Type 'exit' to quit.\n")

    while True:
        try:
            text = input("Enter text → ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not text:
            continue

        if text.lower() == "exit":
            print("Exiting.")
            break

        result = predict(text, model, tokenizer)
        display(result)


# -------------------------------------------------

if __name__ == "__main__":
    main()