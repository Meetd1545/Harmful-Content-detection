"""M2 - Fine-tuned multilingual transformer (MuRIL by default) with a three-output multi-label head.
Labels of -1 (not annotated) are masked out of the loss. Needs an NVIDIA GPU (tested on a 4 GB RTX 3050 Ti).
Run:  python src/transformer_model.py      (saves models/transformer/)
The word-embedding matrix (about 150M of MuRIL's 237M parameters, because of its large vocabulary) is frozen
so that training fits in 4 GB of GPU memory; all transformer layers and the classification head are trained.
"""
import json
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from config import (DATA_PROCESSED, MODELS, RESULTS, DIMENSIONS, UNKNOWN, SEED, ENCODER_NAME, MAX_LEN,
                    BATCH_SIZE, LEARNING_RATE, EPOCHS, WEIGHT_DECAY)
from preprocess import normalise

OUT_DIR = MODELS / "transformer"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MultiLabelModel(nn.Module):
    def __init__(self, encoder_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(hidden, len(DIMENSIONS))

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        return self.head(self.dropout(cls))  # logits, shape (batch, 3)


class TextDS(Dataset):
    def __init__(self, df, tok):
        self.texts = [normalise(t) for t in df["text"].tolist()]
        self.labels = df[DIMENSIONS].values.astype(np.float32) if all(d in df for d in DIMENSIONS) else None
        self.tok = tok

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[i])
        return item


def masked_bce(logits, labels, pos_weight):
    mask = (labels != UNKNOWN).float()
    target = labels.clamp(min=0)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp(min=1)


@torch.no_grad()
def predict_logits(model, tok, texts, batch_size=64):
    model.eval()
    ds = TextDS(pd.DataFrame({"text": list(texts)}), tok)
    out = []
    for b in DataLoader(ds, batch_size=batch_size):
        with torch.autocast(device_type="cuda", enabled=DEVICE == "cuda"):
            out.append(model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE)).float().cpu().numpy())
    return np.vstack(out) if out else np.zeros((0, len(DIMENSIONS)))


def load_trained():
    tok = AutoTokenizer.from_pretrained(OUT_DIR)
    model = MultiLabelModel(ENCODER_NAME)
    model.load_state_dict(torch.load(OUT_DIR / "model.pt", map_location=DEVICE))
    return model.to(DEVICE).eval(), tok


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    train = pd.read_csv(DATA_PROCESSED / "train.csv")
    val = pd.read_csv(DATA_PROCESSED / "val.csv")
    tok = AutoTokenizer.from_pretrained(ENCODER_NAME)
    model = MultiLabelModel(ENCODER_NAME).to(DEVICE)
    model.encoder.embeddings.word_embeddings.weight.requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"device: {DEVICE}; trainable parameters: {trainable / 1e6:.1f}M of {total_params / 1e6:.1f}M; training items: {len(train)}")

    pos = [(train[d] == 1).sum() for d in DIMENSIONS]
    neg = [(train[d] == 0).sum() for d in DIMENSIONS]
    pos_weight = torch.tensor([n / max(p, 1) for p, n in zip(pos, neg)], dtype=torch.float32).clamp(max=10).to(DEVICE)

    dl = DataLoader(TextDS(train, tok), batch_size=BATCH_SIZE, shuffle=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    steps = len(dl) * EPOCHS
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE == "cuda")

    t0, best, history = time.time(), np.inf, []
    for ep in range(EPOCHS):
        model.train()
        total = 0.0
        for step, b in enumerate(dl):
            if step % 100 == 0:
                print(f"  epoch {ep + 1}/{EPOCHS}  step {step}/{len(dl)}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", enabled=DEVICE == "cuda"):
                logits = model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE))
            loss = masked_bce(logits.float(), b["labels"].to(DEVICE), pos_weight)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
        vlog = torch.tensor(predict_logits(model, tok, val["text"]))
        vloss = masked_bce(vlog, torch.tensor(val[DIMENSIONS].values, dtype=torch.float32), pos_weight.cpu()).item()
        history.append({"epoch": ep + 1, "train_loss": total / len(dl), "val_loss": vloss})
        print(history[-1])
        if vloss < best:
            best = vloss
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), OUT_DIR / "model.pt")
            tok.save_pretrained(OUT_DIR)

    info = {"encoder": ENCODER_NAME, "frozen": "word embeddings", "trainable_params_m": round(trainable / 1e6, 1),
            "total_params_m": round(total_params / 1e6, 1), "max_len": MAX_LEN, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS, "pos_weight": pos_weight.cpu().tolist(), "train_items": len(train),
            "training_time_min": round((time.time() - t0) / 60, 1), "device": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu",
            "history": history}
    (RESULTS / "transformer_training.json").write_text(json.dumps(info, indent=2))
    print(info)


if __name__ == "__main__":
    main()
