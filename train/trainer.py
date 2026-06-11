import json
import math
import os

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import config
from data.build_tokenizer import load_tokenizer
from data import PairDataset, pair_collate
from model.contrastive_learning import info_nce
from model.model import TransformerEncoderModel


def get_scheduler(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(step):
        if step < num_warmup_steps:
            return step / max(1, num_warmup_steps)
        progress = (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for (q_ids, q_mask), (p_ids, p_mask) in loader:
        q_ids, q_mask = q_ids.to(device), q_mask.to(device)
        p_ids, p_mask = p_ids.to(device), p_mask.to(device)
        loss = info_nce(model(q_ids, q_mask), model(p_ids, p_mask), config.TEMPERATURE)
        total += loss.item()
        n += 1
    model.train()
    return total / max(1, n)


def train(model, train_loader, val_loader, optimizer, scheduler, scaler, writer, epochs, device):
    best_val_loss = float("inf")
    step = 0

    for epoch in range(epochs):
        model.train()
        for (q_ids, q_mask), (p_ids, p_mask) in train_loader:
            q_ids, q_mask = q_ids.to(device), q_mask.to(device)
            p_ids, p_mask = p_ids.to(device), p_mask.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                loss = info_nce(model(q_ids, q_mask), model(p_ids, p_mask), config.TEMPERATURE)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
            step += 1

        val_loss = evaluate(model, val_loader, device)
        writer.add_scalar("val/loss", val_loss, epoch)
        print(f"epoch {epoch + 1}/{epochs}  val_loss={val_loss:.4f}")

        # save checkpoint every epoch to survive Kaggle session timeouts
        torch.save(model.state_dict(), f"checkpoint_epoch{epoch + 1}.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.MODEL_PATH)
            print(f"  -> best model saved (val_loss={val_loss:.4f})")

    return best_val_loss


def main():
    torch.manual_seed(config.SEED)
    device = torch.device(config.DEVICE)

    tokenize, pad_id, vocab_size = load_tokenizer()

    with open(config.TRAIN_PAIRS_PATH, encoding="utf-8") as f:
        train_pairs = json.load(f)

    with open(config.VAL_EVAL_PATH, encoding="utf-8") as f:
        val_data = json.load(f)
    with open(config.CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)

    chunk_lookup = {c["id"]: c["text"] for c in chunks}
    val_pairs = [[q, chunk_lookup[g]] for q, g in zip(val_data["queries"], val_data["gold"])]

    def collate(batch):
        return pair_collate(batch, pad_id)

    train_loader = DataLoader(
        PairDataset(train_pairs, tokenize),
        batch_size=config.FINETUNE_BS,
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        PairDataset(val_pairs, tokenize),
        batch_size=config.FINETUNE_BS,
        shuffle=False,
        collate_fn=collate,
    )

    model = TransformerEncoderModel(
        vocab_size=vocab_size,
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        num_layers=config.NUM_LAYERS,
        dim_feedforward=config.DIM_FF,
        max_len=config.MODEL_MAX_LEN,
        dropout=config.DROPOUT,
        pad_id=pad_id,
    ).to(device)

    num_training_steps = config.FINETUNE_EPOCHS * len(train_loader)
    num_warmup_steps = num_training_steps // 10

    optimizer = AdamW(model.parameters(), lr=config.FINETUNE_LR)
    scheduler = get_scheduler(optimizer, num_warmup_steps, num_training_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    writer = SummaryWriter("runs/training")

    print(f"device={device} | train={len(train_pairs)} pairs | val={len(val_pairs)} pairs")
    train(model, train_loader, val_loader, optimizer, scheduler, scaler, writer, config.FINETUNE_EPOCHS, device)
    writer.close()
    print("training complete.")


if __name__ == "__main__":
    main()
