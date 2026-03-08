import torch
import torch.nn as nn
from transformers import AutoModel


class SarcasmAwareSentimentTransformer(nn.Module):
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

        #gate_proj takes cls + sarcasm_prob and produces a gate to modulate the representation
        self.gate_proj = nn.Sequential(
            nn.Linear(hidden_size + 1, hidden_size),
            nn.Tanh()
        )

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
        sarcasm_prob = self.sigmoid(sarcasm_logits)    

        gate_input = torch.cat([cls, sarcasm_prob], dim=-1) 
        gate = self.gate_proj(gate_input)                   


        gated_representation = self.gate_norm(cls + gate)   

        sentiment_logits = self.sentiment_head(gated_representation)

        return sentiment_logits, sarcasm_logits