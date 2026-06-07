# Created by LORD

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from datetime import datetime

import os
import re

import config
from dataset import SentimentDataset, SarcasmDataset
from Models import Model_aware


# -------------------------------------------------
# CHECKPOINT HELPERS
# -------------------------------------------------

def get_last_checkpoint():
    """Find the most recent checkpoint in CHECK_DIR."""

    if not os.path.exists(config.CHECK_DIR):
        return None, 0

    files = os.listdir(config.CHECK_DIR)
    epochs = []

    for f in files:
        match = re.search(r"best_model_epoch_(\d+)\.pt", f)
        if match:
            epochs.append(int(match.group(1)))

    if not epochs:
        return None, 0

    last_epoch = max(epochs)
    path = os.path.join(config.CHECK_DIR, f"best_model_epoch_{last_epoch}.pt")

    return path, last_epoch


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_score):
    torch.save(
        {
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler":    scaler.state_dict(),
            "epoch":     epoch,
            "best_score": best_score,
        },
        path,
    )


def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    start_epoch = ckpt["epoch"]
    best_score  = ckpt["best_score"]
    return start_epoch, best_score


# -------------------------------------------------
# EVALUATION
# -------------------------------------------------

def evaluate_sentiment(model, loader):
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(config.DEVICE)
            mask      = batch["attention_mask"].to(config.DEVICE)
            labels    = batch["sentiment"].to(config.DEVICE)

            with autocast():
                logits, _ = model(input_ids, mask)

            preds    = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    return correct / total if total > 0 else 0.0


def evaluate_sarcasm(model, loader):
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(config.DEVICE)
            mask      = batch["attention_mask"].to(config.DEVICE)
            labels    = batch["sarcasm"].to(config.DEVICE)

            with autocast():
                _, logits = model(input_ids, mask)

            preds    = (torch.sigmoid(logits.squeeze()) > 0.5).float()
            correct += (preds == labels.float()).sum().item()
            total   += labels.size(0)

    return correct / total if total > 0 else 0.0


# -------------------------------------------------
# INTERLEAVED BATCH STEP HELPERS
# -------------------------------------------------

def sentiment_step(batch, model, scaler, loss_fn, accum_step, is_last_accum):
    """
    Single sentiment forward+backward.
    Divides loss by GRAD_ACCUM_STEPS so gradients average correctly.
    Returns (raw_loss, n_correct, n_total).
    """

    input_ids = batch["input_ids"].to(config.DEVICE)
    mask      = batch["attention_mask"].to(config.DEVICE)
    labels    = batch["sentiment"].to(config.DEVICE)

    with autocast():
        logits, _ = model(input_ids, mask)
        loss = loss_fn(logits, labels) / config.GRAD_ACCUM_STEPS

    scaler.scale(loss).backward()

    preds     = torch.argmax(logits.detach(), dim=1)
    n_correct = (preds == labels).sum().item()
    n_total   = labels.size(0)

    # raw loss (un-divided) for logging
    return loss.item() * config.GRAD_ACCUM_STEPS, n_correct, n_total


def sarcasm_step(batch, model, scaler, loss_fn, accum_step, is_last_accum):
    """
    Single sarcasm forward+backward.
    Divides loss by GRAD_ACCUM_STEPS so gradients average correctly.
    Returns (raw_loss, n_correct, n_total).
    """

    input_ids = batch["input_ids"].to(config.DEVICE)
    mask      = batch["attention_mask"].to(config.DEVICE)
    labels    = batch["sarcasm"].to(config.DEVICE).float()

    with autocast():
        _, logits = model(input_ids, mask)
        loss = loss_fn(logits.squeeze(), labels) / config.GRAD_ACCUM_STEPS

    scaler.scale(loss).backward()

    preds     = (torch.sigmoid(logits.detach().squeeze()) > 0.5).float()
    n_correct = (preds == labels).sum().item()
    n_total   = labels.size(0)

    return loss.item() * config.GRAD_ACCUM_STEPS, n_correct, n_total


def optimizer_step(model, optimizer, scheduler, scaler):
    """Unscale → clip → step → update → zero grad."""
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    optimizer.zero_grad()


# -------------------------------------------------
# MAIN TRAINING FUNCTION
# -------------------------------------------------

def train():

    os.makedirs(config.CHECK_DIR, exist_ok=True)
    os.makedirs("Results", exist_ok=True)

    # ── Results log ──────────────────────────────
    results_file = open("Results/training_results.txt", "a")
    run_header = (
        f"\n{'=' * 60}\n"
        f"Run started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"LR          : {config.LR}\n"
        f"Batch size  : {config.BATCH_SIZE}\n"
        f"Epochs      : {config.EPOCHS}\n"
        f"Device      : {config.DEVICE}\n"
        f"{'=' * 60}\n"
    )
    results_file.write(run_header)
    results_file.flush()
    print(run_header)

    # ── Datasets ─────────────────────────────────
    sentiment_dataset = SentimentDataset(config.CSV_DATASET)
    sarcasm_dataset   = SarcasmDataset(config.CSV_DATASET_SARCASM)

    indices_sen  = torch.randperm(len(sentiment_dataset))[120000:130000]
    sentiment_dataset = Subset(sentiment_dataset, indices_sen)

    indices_sarc = torch.randperm(len(sarcasm_dataset))[80000:200000]
    sarcasm_dataset   = Subset(sarcasm_dataset, indices_sarc)

    print(f"Sentiment dataset : {len(sentiment_dataset):,} samples")
    print(f"Sarcasm dataset   : {len(sarcasm_dataset):,} samples")

    # ── Train / test splits ───────────────────────
    sent_train_size = int(0.9 * len(sentiment_dataset))
    sent_test_size  = len(sentiment_dataset) - sent_train_size

    sarc_train_size = int(0.9 * len(sarcasm_dataset))
    sarc_test_size  = len(sarcasm_dataset) - sarc_train_size

    sent_train, sent_test = random_split(sentiment_dataset, [sent_train_size, sent_test_size])
    sarc_train, sarc_test = random_split(sarcasm_dataset,   [sarc_train_size, sarc_test_size])

    sentiment_loader      = DataLoader(sent_train, batch_size=config.BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    sarcasm_loader        = DataLoader(sarc_train, batch_size=config.BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    sentiment_test_loader = DataLoader(sent_test,  batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    sarcasm_test_loader   = DataLoader(sarc_test,  batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # ── Model ─────────────────────────────────────
    model = Model_aware.SarcasmAwareSentimentTransformer()
    model.to(config.DEVICE)

    # Freeze embeddings + first 6 transformer layers
    for param in model.encoder.embeddings.parameters():
        param.requires_grad = False

    for name, param in model.encoder.named_parameters():
        if "encoder.layer" in name:
            layer_num = int(name.split("encoder.layer.")[1].split(".")[0])
            if layer_num < 6:
                param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

    # ── Optimizer ────────────────────────────────
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LR,
        weight_decay=0.01,
    )

    # ── Loss functions ───────────────────────────
    sentiment_loss_fn = nn.CrossEntropyLoss()

    sarcasm_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([config.SARCASM_POS_WEIGHT]).to(config.DEVICE)
    )

    scaler = GradScaler()

    # ── Steps & scheduler ────────────────────────
    steps_per_epoch = max(len(sentiment_loader), len(sarcasm_loader))
    total_steps     = steps_per_epoch * config.EPOCHS

    # ── Checkpoint resume ────────────────────────
    checkpoint_path, _ = get_last_checkpoint()
    start_epoch = 0
    best_score  = -1.0

    # Build scheduler before (potentially) loading its state
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    if checkpoint_path:
        print(f"Resuming from checkpoint: {checkpoint_path}")
        start_epoch, best_score = load_checkpoint(
            checkpoint_path, model, optimizer, scheduler, scaler, config.DEVICE
        )
        print(f"Resumed at epoch {start_epoch} | best score so far: {best_score:.4f}")

    best_model_path = checkpoint_path  # to delete stale files

    # -------------------------------------------------
    # EPOCH LOOP
    # -------------------------------------------------

    for epoch in range(start_epoch, config.EPOCHS):

        model.train()

        # Running metrics
        total_loss    = 0.0
        correct_sent  = total_sent  = 0
        correct_sarc  = total_sarc  = 0
        n_steps       = 0

        # Iterators that cycle independently so neither task starves
        sent_iter = iter(sentiment_loader)
        sarc_iter = iter(sarcasm_loader)

        print(f"\nEpoch {epoch + 1}/{config.EPOCHS}")

        pbar = tqdm(range(steps_per_epoch), desc="Training")

        optimizer.zero_grad()  # zero once at the start of the epoch

        for step in pbar:

            model.train()

            accum_step    = step % config.GRAD_ACCUM_STEPS
            is_last_accum = (accum_step == config.GRAD_ACCUM_STEPS - 1)

            # ── Sentiment step ───────────────────
            try:
                sent_batch = next(sent_iter)
            except StopIteration:
                sent_iter  = iter(sentiment_loader)
                sent_batch = next(sent_iter)

            s_loss, s_correct, s_total = sentiment_step(
                sent_batch, model, scaler, sentiment_loss_fn, accum_step, is_last_accum
            )
            total_loss   += s_loss
            correct_sent += s_correct
            total_sent   += s_total

            # ── Sarcasm step ─────────────────────
            try:
                sarc_batch = next(sarc_iter)
            except StopIteration:
                sarc_iter  = iter(sarcasm_loader)
                sarc_batch = next(sarc_iter)

            r_loss, r_correct, r_total = sarcasm_step(
                sarc_batch, model, scaler, sarcasm_loss_fn, accum_step, is_last_accum
            )
            total_loss   += r_loss
            correct_sarc += r_correct
            total_sarc   += r_total

            # ── Optimizer step every GRAD_ACCUM_STEPS ──
            if is_last_accum:
                optimizer_step(model, optimizer, scheduler, scaler)

            n_steps += 2

            sent_acc = correct_sent / total_sent if total_sent > 0 else 0.0
            sarc_acc = correct_sarc / total_sarc if total_sarc > 0 else 0.0

            pbar.set_postfix(
                loss=f"{total_loss / n_steps:.4f}",
                sent=f"{sent_acc:.4f}",
                sarc=f"{sarc_acc:.4f}",
            )

        # ── Epoch-end evaluation ─────────────────
        avg_loss     = total_loss / n_steps
        sentiment_acc = evaluate_sentiment(model, sentiment_test_loader)
        sarcasm_acc   = evaluate_sarcasm(model, sarcasm_test_loader)

        # Weighted score: sentiment is the primary task
        score = 0.7 * sentiment_acc + 0.3 * sarcasm_acc

        epoch_summary = (
            f"\nEpoch {epoch + 1} Results\n"
            f"  Avg Loss         : {avg_loss:.4f}\n"
            f"  Sentiment Acc    : {sentiment_acc:.4f}\n"
            f"  Sarcasm Acc      : {sarcasm_acc:.4f}\n"
            f"  Weighted Score   : {score:.4f}\n"
        )
        print(epoch_summary)

        results_file.write(
            f"Epoch {epoch + 1} | "
            f"Loss {avg_loss:.4f} | "
            f"Sentiment Acc {sentiment_acc:.4f} | "
            f"Sarcasm Acc {sarcasm_acc:.4f} | "
            f"Score {score:.4f}\n"
        )
        results_file.flush()

        # ── Save best checkpoint ─────────────────
        if score > best_score:
            best_score = score

            new_ckpt_path = os.path.join(
                config.CHECK_DIR, f"best_model_epoch_{epoch + 1}.pt"
            )

            save_checkpoint(
                new_ckpt_path,
                model, optimizer, scheduler, scaler,
                epoch + 1,
                best_score,
            )

            print(f"best model saved → {new_ckpt_path}")

            # Remove the previous best to keep the directory clean
            if best_model_path and best_model_path != new_ckpt_path:
                if os.path.exists(best_model_path):
                    os.remove(best_model_path)

            best_model_path = new_ckpt_path

    results_file.write(f"\nTraining complete. Best score: {best_score:.4f}\n")
    results_file.close()
    print(f"\nTraining complete. Best score: {best_score:.4f}")


# -------------------------------------------------

if __name__ == "__main__":
    train()
