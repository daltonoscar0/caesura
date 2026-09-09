"""Fine-tune System B.

A plain training loop rather than ``Trainer``: the whole thing is one linear
head over six transformer layers and the loop is short enough that the explicit
version is easier to read than the configuration that would drive the generic
one.

Selection is on dev macro-F1 over the two break classes only. Overall accuracy
is useless here because 85% of boundaries are ``none``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from typing import Dict, List, Sequence, Tuple

from .data import Example, read_jsonl
from .model import BASE_MODEL, MAX_LENGTH, encode
from .types import B2, B3, ID2LABEL, LABEL2ID, LABELS, NONE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUTPUT = os.path.join(ROOT, "outputs", "breaks")


def pick_device(requested: str = "auto") -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_features(tokenizer, examples: Sequence[Example]) -> List[Dict]:
    feats = []
    for ex in examples:
        enc, _ = encode(tokenizer, ex.tokens, ex.labels)
        feats.append(
            {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "labels": enc["labels"],
            }
        )
    return feats


def collate(batch, pad_id: int, device):
    import torch

    width = max(len(b["input_ids"]) for b in batch)
    ids = torch.tensor(
        [b["input_ids"] + [pad_id] * (width - len(b["input_ids"])) for b in batch],
        device=device,
    )
    mask = torch.tensor(
        [b["attention_mask"] + [0] * (width - len(b["attention_mask"])) for b in batch],
        device=device,
    )
    labels = torch.tensor(
        [b["labels"] + [-100] * (width - len(b["labels"])) for b in batch], device=device
    )
    return ids, mask, labels


def macro_break_f1(gold: Sequence[int], pred: Sequence[int]) -> Tuple[float, Dict[str, float]]:
    per_class: Dict[str, float] = {}
    scores = []
    for label in (B2, B3):
        lid = LABEL2ID[label]
        tp = sum(1 for g, p in zip(gold, pred) if g == lid and p == lid)
        fp = sum(1 for g, p in zip(gold, pred) if g != lid and p == lid)
        fn = sum(1 for g, p in zip(gold, pred) if g == lid and p != lid)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[label] = f1
        scores.append(f1)
    return sum(scores) / len(scores), per_class


def evaluate(model, feats, pad_id, device, batch_size: int = 32):
    import torch

    model.eval()
    gold: List[int] = []
    pred: List[int] = []
    with torch.no_grad():
        for start in range(0, len(feats), batch_size):
            batch = feats[start : start + batch_size]
            ids, mask, labels = collate(batch, pad_id, device)
            logits = model(input_ids=ids, attention_mask=mask).logits
            best = logits.argmax(-1)
            keep = labels != -100
            gold.extend(labels[keep].tolist())
            pred.extend(best[keep].tolist())
    model.train()
    return macro_break_f1(gold, pred)


def train(
    train_path: str,
    dev_path: str,
    output_dir: str = OUTPUT,
    base_model: str = BASE_MODEL,
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 3e-5,
    seed: int = 17,
    device: str = "auto",
    evals_per_epoch: int = 2,
    max_train: int = 0,
    dev_subsample: int = 500,
) -> Dict:
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    random.seed(seed)
    torch.manual_seed(seed)
    device = pick_device(device)

    tokenizer = AutoTokenizer.from_pretrained(base_model, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(
        base_model,
        num_labels=len(LABELS),
        id2label={i: ID2LABEL[i] for i in range(len(LABELS))},
        label2id=dict(LABEL2ID),
    ).to(device)

    train_ex = read_jsonl(train_path)
    dev_ex = read_jsonl(dev_path)
    if max_train:
        train_ex = train_ex[:max_train]
    train_feats = build_features(tokenizer, train_ex)
    dev_feats = build_features(tokenizer, dev_ex)
    # Model selection runs on a fixed dev subsample so that checkpointing does
    # not dominate wall clock; the reported dev score is the full-dev one below.
    if dev_subsample and dev_subsample < len(dev_feats):
        sel_feats = random.Random(seed).sample(dev_feats, dev_subsample)
    else:
        sel_feats = dev_feats

    pad_id = tokenizer.pad_token_id
    steps_per_epoch = math.ceil(len(train_feats) / batch_size)
    total_steps = steps_per_epoch * epochs
    eval_every = max(1, steps_per_epoch // evals_per_epoch)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps, pct_start=0.1, anneal_strategy="linear"
    )

    print(
        f"device={device} train={len(train_feats)} dev={len(dev_feats)} "
        f"steps={total_steps} eval_every={eval_every}"
    )

    best_f1 = -1.0
    best_step = -1
    history: List[Dict] = []
    step = 0
    started = time.perf_counter()
    model.train()

    for epoch in range(epochs):
        order = list(range(len(train_feats)))
        random.shuffle(order)
        for start in range(0, len(order), batch_size):
            batch = [train_feats[i] for i in order[start : start + batch_size]]
            ids, mask, labels = collate(batch, pad_id, device)
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % eval_every == 0 or step == total_steps:
                f1, per_class = evaluate(model, sel_feats, pad_id, device)
                elapsed = time.perf_counter() - started
                history.append(
                    {
                        "step": step,
                        "epoch": epoch,
                        "loss": round(float(loss.item()), 4),
                        "dev_macro_break_f1": round(f1, 4),
                        "dev_b2_f1": round(per_class[B2], 4),
                        "dev_b3_f1": round(per_class[B3], 4),
                        "seconds": round(elapsed, 1),
                    }
                )
                print(
                    f"  step {step:5d}/{total_steps} loss {loss.item():.4f} "
                    f"dev macro-F1 {f1:.4f} (b2 {per_class[B2]:.4f} b3 {per_class[B3]:.4f}) "
                    f"{elapsed:.0f}s"
                )
                if f1 > best_f1:
                    best_f1 = f1
                    best_step = step
                    os.makedirs(output_dir, exist_ok=True)
                    model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

    wall = time.perf_counter() - started

    # Reload the selected checkpoint and score it on the whole dev set.
    best_model = AutoModelForTokenClassification.from_pretrained(output_dir).to(device)
    full_f1, full_per_class = evaluate(best_model, dev_feats, pad_id, device)
    print(
        f"full dev macro-F1 {full_f1:.4f} "
        f"(b2 {full_per_class[B2]:.4f} b3 {full_per_class[B3]:.4f})"
    )

    summary = {
        "base_model": base_model,
        "device": device,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "train_examples": len(train_feats),
        "dev_examples": len(dev_feats),
        "best_selection_macro_break_f1": round(best_f1, 4),
        "dev_macro_break_f1": round(full_f1, 4),
        "dev_b2_f1": round(full_per_class[B2], 4),
        "dev_b3_f1": round(full_per_class[B3], 4),
        "best_step": best_step,
        "total_steps": total_steps,
        "wall_seconds": round(wall, 1),
        "history": history,
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "training_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"best dev macro-F1 {best_f1:.4f} at step {best_step}; {wall:.0f}s total")
    print(f"saved to {output_dir}")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fine-tune the Caesura break labeller")
    ap.add_argument("--train", default=os.path.join(DATA, "train.jsonl"))
    ap.add_argument("--dev", default=os.path.join(DATA, "dev.jsonl"))
    ap.add_argument("--output", default=OUTPUT)
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--dev-subsample", type=int, default=500)
    ap.add_argument("--evals-per-epoch", type=int, default=2)
    args = ap.parse_args(argv)

    train(
        args.train,
        args.dev,
        output_dir=args.output,
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        max_train=args.max_train,
        dev_subsample=args.dev_subsample,
        evals_per_epoch=args.evals_per_epoch,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
