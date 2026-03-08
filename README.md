# Sentiment Fusion

Sentiment Fusion is an advanced sentiment analysis and emotion detection system powered by state-of-the-art transformer models and Vector Databases. It robustly captures nuances in text, such as sarcasm and mixed emotions, to deliver highly accurate and nuanced sentiment predictions.

## Architecture & Model
![Model Architecture](Models/ChatGPT%20Image%20Mar%205,%202026,%2001_04_39%20PM.png)

The core models include `SarcasmAwareSentimentTransformer` and `Robust_SentimentModel`, built on top of `xlm-roberta-base`. 
The architecture intelligently handles sarcasm by incorporating a distinct sarcasm-detection head. The probability of sarcasm is used to modulate the primary `[CLS]` token via a dedicated gating mechanism (`gate_proj`). The resulting sarcasm-aware gated representation is then normalized and passed to the final sentiment classifier. This ensures the model explicitly understands and accounts for sarcastic tone when deriving the final sentiment logits.

## Vector DB in Inference
To provide rich, nuanced emotional context beyond simple "Positive/Negative/Neutral" classes, the system employs a FAISS Vector Database during inference.
1. The model extracts a rich 1540-dimensional context vector consisting of the `[CLS]` token, the mean-pooled sequence embeddings, and scaled sentiment/sarcasm probabilities.
2. The FAISS database is partitioned into three polarity-specific sub-indexes (`positive`, `negative`, `neutral`).
3. Based on the model's primary sentiment prediction, the system routes the query vector to the matching sub-index to perform a nearest-neighbor search. 
4. This retrieves the top closest emotions and intensities (e.g., "Mild Joy", "Strong Anger", "Moderate Sarcasm") that semantically align with the predicted sentiment, yielding a highly nuanced emotional profile.

## Training & Test Results
Extensive training runs were conducted to optimize the model. According to the final training reports (`Results/training_results.txt`), the model demonstrated significant learning and consistency.

**Highlights from the latest optimal runs (Batch Size 64, LR 3e-05):**
- **Epoch 14-17 Peak Performance:**
  - **Sentiment Accuracy:** ~94.40%
  - **Sarcasm Accuracy:** ~75.95%
  - **Overall Score:** ~0.8866
  
*Validation Loss reached a minimum of ~0.2284 by Epoch 15, showing highly robust convergence.*

## Repository Setup
To use this repository, ensure you have the required dependencies installed (e.g., `torch`, `transformers`, `faiss-cpu`, etc.).
Datasets are ignored in the version control to keep the repository lightweight, but pre-trained models and checkpoints can be loaded from the `Models` and `CheckPoints` directories.
