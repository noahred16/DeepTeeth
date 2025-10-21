#!/usr/bin/env python3
"""
Manual grid search for DeepTeeth (metrics-driven tuning, Jupyter/macOS-safe).
- Single-folder data layout expected: ./data with file prefixes train_/validation_/test_
- Optimizes for validation Macro-F1; also prints Acc and Loss.
- Safe DataLoader settings (num_workers=0) to avoid macOS/Jupyter multiprocessing issues.
"""

import os
import numpy as np

# Tame OpenMP threads (prevents OMP mutex init errors on macOS)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")  # harmless if MKL not present

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.amp import GradScaler  # new API
from torch.cuda.amp import autocast
from collections import Counter
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
import multiprocessing as mp

# Safer start method for Jupyter/macOS (no effect if already set)
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

# Limit CPU threads further if needed
torch.set_num_threads(1)

# Import from your project
from resnet_cnn import ResNet152, DentexDataset, CLASS_NAMES, IMG_HEIGHT, IMG_WIDTH

# Device auto-detect: CUDA -> MPS (Apple Silicon) -> CPU
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print("Device:", DEVICE)


# ---------- helpers ----------
def build_transforms(aug_jitter=0.10):
    train_tf = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=7),
        transforms.ColorJitter(brightness=aug_jitter, contrast=aug_jitter),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return train_tf, val_tf


def class_weights_from_dataset(ds):
    counts = Counter([y for _, y in ds])
    total = sum(counts.values())
    weights = [total / max(1, counts[i]) for i in range(len(CLASS_NAMES))]
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def run_one_epoch(model, loader, criterion, optimizer, scaler, train=True):
    model.train() if train else model.eval()
    losses, y_true, y_pred = [], [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if train:
                optimizer.zero_grad(set_to_none=True)
            # autocast only meaningful if CUDA; harmless otherwise
            with autocast(enabled=(DEVICE.type == "cuda")):
                out = model(x)
                loss = criterion(out, y)
            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            losses.append(loss.item())
            y_true.append(y.detach().cpu())
            y_pred.append(out.argmax(1).detach().cpu())
    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    acc = accuracy_score(y_true, y_pred)
    return float(np.mean(losses)), float(macro_f1), float(acc)


def train_config(lr, dropout, aug_jitter, epochs=6, batch_size=32):
    # Single-folder layout with train_/validation_ prefixes
    DATA_DIR = "data"
    assert os.path.isdir(DATA_DIR), f"Missing data directory: {DATA_DIR}"

    train_tf, val_tf = build_transforms(aug_jitter)
    train_ds = DentexDataset(DATA_DIR, "train",      transform=train_tf)
    val_ds   = DentexDataset(DATA_DIR, "validation", transform=val_tf)

    # Jupyter/macOS-safe DataLoaders (no multiprocessing)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False, persistent_workers=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False, persistent_workers=False
    )

    model = ResNet152(num_classes=len(CLASS_NAMES), pretrained=True, freeze_conv=True).to(DEVICE)

    # ensure classifier head is trainable and set dropout
    for m in model.base_model.fc.modules():
        if isinstance(m, nn.Dropout):
            m.p = dropout
    for p in model.base_model.fc.parameters():
        p.requires_grad = True

    class_weights = class_weights_from_dataset(train_ds)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # GradScaler: pass device type; on CPU/MPS it will be disabled automatically
    scaler = GradScaler("cuda") if DEVICE.type == "cuda" else GradScaler(enabled=False)

    best_val_f1, last_val_acc, last_val_loss = 0.0, 0.0, 0.0
    for ep in range(1, epochs+1):
        tr_loss, tr_f1, tr_acc = run_one_epoch(model, train_loader, criterion, optimizer, scaler, train=True)
        vl_loss, vl_f1, vl_acc = run_one_epoch(model, val_loader,   criterion, optimizer, scaler, train=False)
        scheduler.step()
        best_val_f1 = max(best_val_f1, vl_f1)
        last_val_acc, last_val_loss = vl_acc, vl_loss
        print(f"  Ep{ep:02d} | TrainF1={tr_f1:.3f} | ValF1={vl_f1:.3f} | ValAcc={vl_acc:.3f} | ValLoss={vl_loss:.3f}")
    return best_val_f1, last_val_acc, last_val_loss


def grid_search():
    # You can tweak the grid here
    lrs       = [1e-3, 5e-4, 1e-4]
    dropouts  = [0.3, 0.5]
    augments  = [0.10, 0.20]
    epochs    = 6
    batch_sz  = 32

    results = []
    trial = 0
    print("Starting manual grid search...\n")
    for lr in lrs:
        for dr in dropouts:
            for aug in augments:
                trial += 1
                print(f"\nTrial {trial}: lr={lr}, dropout={dr}, aug={aug}")
                best_val_f1, val_acc, val_loss = train_config(
                    lr=lr, dropout=dr, aug_jitter=aug,
                    epochs=epochs, batch_size=batch_sz
                )
                results.append({
                    "trial": trial,
                    "lr": lr,
                    "dropout": dr,
                    "aug": aug,
                    "epochs": epochs,
                    "batch": batch_sz,
                    "val_f1": best_val_f1,
                    "val_acc": val_acc,
                    "val_loss": val_loss
                })
                print("-" * 56)
    return results


def main():
    results = grid_search()
    df = pd.DataFrame(results).sort_values("val_f1", ascending=False).reset_index(drop=True)
    print("\nGrid Search Summary:")
    print(df.to_string(index=False))

    os.makedirs("metrics", exist_ok=True)
    out_path = os.path.join("metrics", "manual_grid_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
