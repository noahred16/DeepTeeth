import re, random
from pathlib import Path
from collections import Counter
from typing import Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.models import resnet50, ResNet50_Weights
import torch


print(torch.version.cuda)
print(torch.cuda.is_available())
# ============== CONFIG ==============
EPOCHS       = 10
BATCH_SIZE   = 128
LR           = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS  = 0
SEED         = 42

KEEP_ORIGINAL_SIZE = True   # False: resize+crop; True: augment original size

# Data augmentation params (if KEEP_ORIGINAL_SIZE)
ALLOW_HFLIP   = True
MAX_ROT_DEG   = 8
BRIGHTNESS_J  = 0.15
CONTRAST_J    = 0.10
TRANS_FRAC    = 0.03

TRAIN_DIR = "datas/balanced_train"
VAL_DIR   = "datas/val"
TEST_DIR  = "datas/test"

# ============ SAME MAPPING ===========
SUPER_CLASSES = {
    "Caries":     ["Caries"],
    "DeepCaries": ["DeepCaries", "Curettage"],
    "Impacted":   ["Impacted"],
    "Lesion":     ["PeriapicalLesion", "Lesion"],
    "RootCanal":  ["RootCanal"],
    "Healthy":    ["healthy"],
}
EXCLUDED_CLASSES = {"Extraction", "Fracture"}
CLASSES = ["Caries", "DeepCaries", "Healthy", "Impacted", "Lesion", "RootCanal"]

def _canon(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.strip().lower())

_VARIANT2SUPER = {}
for sup, vs in SUPER_CLASSES.items():
    for v in vs:
        _VARIANT2SUPER[_canon(v)] = sup
_VARIANT2SUPER.update({
    "periapical": "Lesion",
    "periapicallesion": "Lesion",
    "intact": "Healthy",
})
_EXCLUDED_CANON = {_canon(x) for x in EXCLUDED_CLASSES}

def to_super(prefix: str) -> Optional[str]:
    c = _canon(prefix)
    if c in _EXCLUDED_CANON:
        return None
    return _VARIANT2SUPER.get(c, None)

# ========= Normalization =========
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

# ============== DATASET ==============
class ToothFlat(Dataset):
    def __init__(self, root, classes, split="train", keep_original=True):
        self.root = Path(root)
        self.paths = sorted([p for p in self.root.glob("*.png")])
        self.classes = classes
        self.class2idx = {c:i for i,c in enumerate(classes)}
        self.split = split
        self.keep_original = keep_original

        if keep_original:
            if split == "train":
                self.tf = T.Compose([
                    T.Lambda(self._train_aug_original),
                    T.ToTensor(), T.Normalize(MEAN, STD),
                ])
            else:
                self.tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
        else:
            if split == "train":
                self.tf = T.Compose([
                    T.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                    T.RandomHorizontalFlip(p=0.5 if ALLOW_HFLIP else 0.0),
                    T.ColorJitter(brightness=BRIGHTNESS_J, contrast=CONTRAST_J),
                    T.ToTensor(), T.Normalize(MEAN, STD),
                ])
            else:
                self.tf = T.Compose([T.Resize(256), T.CenterCrop(224),
                                     T.ToTensor(), T.Normalize(MEAN, STD)])

        keep_p, keep_y = [], []
        skip = 0
        for p in self.paths:
            prefix = p.name.split("_", 1)[0]
            sup = to_super(prefix)
            if sup is None or sup not in self.class2idx:
                skip += 1; continue
            keep_p.append(p); keep_y.append(self.class2idx[sup])

        self.paths, self.labels = keep_p, keep_y
        print(f"[{self.root.name or self.split}] kept={len(self.paths)} | skip={skip}")
        cnt = Counter(self.labels)
        print(" class dist:", {self.classes[k]: v for k,v in sorted(cnt.items())})

    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img), self.labels[i]

    def _train_aug_original(self, img: Image.Image) -> Image.Image:
        if self.split != "train": return img
        w, h = img.size
        if ALLOW_HFLIP and random.random() < 0.5:
            img = TF.hflip(img)
        angle = random.uniform(-MAX_ROT_DEG, MAX_ROT_DEG)
        tx = int(TRANS_FRAC * w * random.uniform(-1.0, 1.0))
        ty = int(TRANS_FRAC * h * random.uniform(-1.0, 1.0))
        img = TF.affine(img, angle=angle, translate=[tx, ty], scale=1.0, shear=[0.0, 0.0], fill=0)
        if BRIGHTNESS_J > 0 or CONTRAST_J > 0:
            img = T.ColorJitter(brightness=BRIGHTNESS_J, contrast=CONTRAST_J)(img)
        return img

# ============== MODEL ===============
def build_resnet50(num_classes: int, freeze=True):
    m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    feat_dim = m.fc.in_features
    m.fc = nn.Identity()
    head = nn.Linear(feat_dim, num_classes)
    model = nn.Sequential(m, head)
    if freeze:
        for p in model[0].parameters(): p.requires_grad = False
        model[0].eval()
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] ResNet-50 loaded. total={total:,} trainable={trainable:,} (head only)")
    return model

def accuracy(logits, y): return (logits.argmax(1) == y).float().mean().item()

@torch.no_grad()
def eval_report(model, loader, device, classes, title="Eval"):
    model.eval()
    ce = nn.CrossEntropyLoss()
    losses, accs = [], []
    y_all, p_all = [], []
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        o = model(x)
        losses.append(ce(o,y).item()); accs.append(accuracy(o,y))
        y_all.append(y.cpu().numpy()); p_all.append(o.argmax(1).cpu().numpy())
    loss = float(np.mean(losses)) if losses else 0.0
    acc  = float(np.mean(accs)) if accs else 0.0
    print(f"[{title}] loss={loss:.4f} acc={acc:.4f}")
    if y_all:
        y_true = np.concatenate(y_all); y_pred = np.concatenate(p_all)
        print_cls_report(y_true, y_pred, classes)
    return loss, acc

def print_cls_report(y_true, y_pred, classes):
    n = len(classes)
    cm = np.zeros((n, n), dtype=np.int64)
    for t,p in zip(y_true, y_pred): cm[t,p] += 1
    print(" per-class Precision  Recall  F1   Support")
    for i,name in enumerate(classes):
        tp = cm[i,i]; fp = cm[:,i].sum()-tp; fn = cm[i,:].sum()-tp
        prec = tp/(tp+fp) if tp+fp>0 else 0.0
        rec  = tp/(tp+fn) if tp+fn>0 else 0.0
        f1   = (2*prec*rec)/(prec+rec) if prec+rec>0 else 0.0
        sup  = cm[i,:].sum()
        print(f"{name:>11s}  {prec:7.4f} {rec:7.4f} {f1:7.4f} {sup:8d}")

# ============== RUN ================
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Env] device={device}")

    train_ds = ToothFlat(TRAIN_DIR, CLASSES, split="train", keep_original=KEEP_ORIGINAL_SIZE)
    val_ds   = ToothFlat(VAL_DIR,   CLASSES, split="val",   keep_original=KEEP_ORIGINAL_SIZE)
    test_ds  = ToothFlat(TEST_DIR,  CLASSES, split="test",  keep_original=KEEP_ORIGINAL_SIZE) \
               if Path(TEST_DIR).exists() else None

    pin = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    test_loader  = (DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=NUM_WORKERS, pin_memory=pin)
                    if test_ds else None)

    model = build_resnet50(num_classes=len(CLASSES), freeze=True).to(device)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=LR, weight_decay=WEIGHT_DECAY)
    ce = nn.CrossEntropyLoss()

    best = -1.0
    ckpt = Path("checkpoints"); ckpt.mkdir(exist_ok=True)
    ckpt_path = ckpt / "resnet50_super_linearprobe.pt"

    for ep in range(1, EPOCHS+1):
        model.train()
        tr_loss = tr_acc = 0.0; seen = 0
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(x); loss = ce(out,y)
            loss.backward(); opt.step()
            with torch.no_grad():
                b = x.size(0); tr_loss += loss.item()*b
                tr_acc += (out.argmax(1)==y).float().sum().item(); seen += b
        tr_loss /= max(1,seen); tr_acc /= max(1,seen)
        va_loss, va_acc = eval_report(model, val_loader, device, CLASSES, title=f"Val@Ep{ep}")
        print(f"[Train] Ep {ep:02d}/{EPOCHS} | loss={tr_loss:.4f} acc={tr_acc:.4f}")
        if va_acc > best:
            best = va_acc
            torch.save({"model": model.state_dict(), "classes": CLASSES}, ckpt_path)
            print(f"  ✓ Saved best → {ckpt_path} (val acc {best:.4f})")

    if test_loader is not None and ckpt_path.exists():
        d = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(d["model"])
        eval_report(model, test_loader, device, CLASSES, title="Test")

if __name__ == "__main__":
    main()