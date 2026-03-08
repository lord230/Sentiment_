import torch
import torch.nn as nn
from transformers import AutoModel


class SarcasmAwareSentimentTransformer(nn.Module):
    """
    Transformer model that jointly learns:
    1. Sentiment classification (positive, neutral, negative)
    2. Sarcasm detection

    Sarcasm probability gates the sentiment representation via a learned
    gating mechanism instead of a hard suppression.
    """

    def __init__(
        self,
        model_name="xlm-roberta-base",
        num_sentiment_classes=3,
        dropout=0.3
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)

        # Sarcasm head
        self.sarcasm_head = nn.Linear(hidden_size, 1)

        # FIX 1: Learned gating projection instead of blind suppression.
        # Takes [cls || sarcasm_prob] and produces a gate vector.
        self.gate_proj = nn.Sequential(
            nn.Linear(hidden_size + 1, hidden_size),
            nn.Tanh()
        )

        # FIX 2: Layer norm to stabilise the gated representation
        self.gate_norm = nn.LayerNorm(hidden_size)

        # Sentiment head
        self.sentiment_head = nn.Linear(hidden_size, num_sentiment_classes)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls = outputs.last_hidden_state[:, 0]
        cls = self.dropout(cls)

        # Sarcasm prediction
        sarcasm_logits = self.sarcasm_head(cls)
        sarcasm_prob = self.sigmoid(sarcasm_logits)          # (B, 1)

        # FIX 1: Learned gate — concat cls + sarcasm signal, project to gate
        gate_input = torch.cat([cls, sarcasm_prob], dim=-1)  # (B, H+1)
        gate = self.gate_proj(gate_input)                    # (B, H)

        # FIX 2: Residual + norm so gradients still flow when sarcasm≈1
        gated_representation = self.gate_norm(cls + gate)    # (B, H)

        # Sentiment prediction on the adjusted representation
        sentiment_logits = self.sentiment_head(gated_representation)

        return sentiment_logits, sarcasm_logits