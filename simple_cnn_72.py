#!/usr/bin/env python3
"""
Simple CNN for tooth disease classification on DENTEX dataset. (should achive 72+% ac)

Loss function, switched from CrossEntropyLoss to Focal Loss (gamma=1.8) (for Adams)
for class balancing, added Class Balanced alpha (from effective number of samples, beta=0.995) and applied it inside Focal Loss
Learning rate schedule- added CosineAnnealingLR over the training epochs


This variant uses **Class-Balanced Focal Loss** (CB-Focal):
- Alpha per class computed from the effective number of samples
- No WeightedRandomSampler (shuffle=True instead) / as sets are quite balnaced already
- Cosine LR scheduler kept
- TensorBoard scalars, embeddings, figures; t-SNE accumulation fixed
"""

"""
Simple CNN for tooth disease classification on DENTEX dataset.

Architecture: Very lightweight CNN with 2 conv layers + 2 FC layers
- Conv1: 3 -> 16 channels
- Conv2: 16 -> 32 channels
- FC1: 20480 -> 64
- FC2: 64 -> 6 classes

Input: 160x256 RGB images
Output: 6 classes (Caries, DeepCaries, Impacted, Lesion, RootCanal, Healthy)
"""

import os
import time
from collections import Counter
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score

from model_visualizer import ModelVisualizer

# -------------------- Config --------------------
TRAIN_DIR   = "data_balanced_train"
VAL_DIR     = "data_validation"
TEST_DIR    = "data_test"
MODEL_DIR   = "models"
METRICS_DIR = "metrics"
RUNS_DIR    = os.path.join(METRICS_DIR, "runs")

IMG_HEIGHT  = 256
IMG_WIDTH   = 160
BATCH_SIZE  = 32
NUM_EPOCHS  = 5
BASE_LR     = 1e-3
WEIGHT_DECAY = 0.0
USE_SCHEDULER = True

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)
MODEL_NAME = "SimpleCNN"

# -------------------- Superclass map --------------------
SUPER_CLASSES = {
    "Caries": ["Caries", "CariesTest"],
    "DeepCaries": ["DeepCaries", "Curettage"],
    "Impacted": ["Impacted"],
    "Lesion": ["PeriapicalLesion", "Lesion"],
    "RootCanal": ["RootCanal"],
    "Healthy": ["Intact"],
}
EXCLUDED_CLASSES = ["Extraction", "Fracture"]

CLASS_NAMES  = sorted(SUPER_CLASSES.keys())
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}


def get_superclass(class_name):
    for superclass, classes in SUPER_CLASSES.items():
        if class_name in classes:
            return superclass
    return None


def parse_filename(filename):
    # sourcetype_classname_idx_imagefilename.png
    parts = filename.split("_")
    if len(parts) >= 2:
        sourcetype = parts[0]
        classname = parts[1]
        superclass = get_superclass(classname)
        return sourcetype, classname, superclass
    return None, None, None


# -------------------- Dataset --------------------
class DentexDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []

        for filename in os.listdir(data_dir):
            if not filename.endswith(".png"):
                continue
            _, classname, superclass = parse_filename(filename)
            if classname in EXCLUDED_CLASSES:
                continue
            if superclass and superclass in CLASS_TO_IDX:
                self.samples.append((filename, CLASS_TO_IDX[superclass]))

        print(f"Loaded {len(self.samples)} samples from {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        img_path = os.path.join(self.data_dir, filename)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# -------------------- Model --------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 160x256 -> 80x128
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 80x128 -> 40x64
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 40x64  -> 20x32
        self.fc1 = nn.Linear(32 * 20 * 32, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# -------------------- Focal + Class-Balanced alpha --------------------
class FocalLoss(nn.Module):
    """
    Multiclass Focal Loss with logits.
    - gamma: focusing parameter (typical 1.5~2.0)
    - alpha: per-class weighting tensor [C] or scalar float
    """
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        elif isinstance(alpha, torch.Tensor):
            self.alpha = alpha.to(torch.float32)
        else:
            self.alpha = torch.tensor([alpha], dtype=torch.float32)

    def forward(self, logits, target):
        # logits: [B,C], target: [B]
        ce = nn.functional.cross_entropy(logits, target, reduction="none")
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce

        if self.alpha is not None:
            if self.alpha.numel() == 1:
                alpha_t = self.alpha.to(logits.device)
            else:
                alpha_t = self.alpha.to(logits.device)[target]
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def class_balanced_alpha(effective_beta: float, class_counts: List[int]) -> torch.Tensor:
    """
    Cui et al., Class-Balanced Loss Based on Effective Number of Samples.
    alpha_c = (1 - beta) / (1 - beta^{n_c}), normalized so mean(alpha)=1.
    """
    counts = torch.tensor(class_counts, dtype=torch.float32)
    # avoid zeros
    counts[counts <= 0] = 1.0
    beta = effective_beta
    en = 1.0 - torch.pow(beta, counts)
    alpha = (1.0 - beta) / en
    # normalize to mean 1 for stability
    alpha = alpha * (alpha.numel() / alpha.sum())
    return alpha


# -------------------- Train / Eval --------------------
def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch_num):
    from tqdm import tqdm
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch_num} [Train]", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = torch.as_tensor(labels, device=device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, pred = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()

        pbar.set_postfix(loss=f"{loss.item():.4f}",
                         acc=f"{(correct/max(total,1)):.4f}")
    return running_loss / max(total, 1), correct / max(total, 1)


def evaluate(model, dataloader, criterion, device, visualizer=None, desc="Eval"):
    from tqdm import tqdm
    model.eval()
    running_loss = 0.0
    all_labels, all_preds = [], []
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=desc, leave=False)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = torch.as_tensor(labels, device=device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            _, pred = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())

            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if visualizer is not None:
                # log logits + labels for projector
                visualizer.embeddings.append(outputs.detach().cpu())
                visualizer.labels.append(labels.cpu())

    avg_loss = running_loss / max(len(all_labels), 1)
    acc = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    return avg_loss, acc, all_labels, all_preds


# -------------------- Main --------------------
def main():
    # Dirs
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(RUNS_DIR, exist_ok=True)

    print("=" * 60)
    print("Simple CNN for Tooth Disease Classification")
    print("=" * 60)
    print(f"Device: {DEVICE.type}")
    print(f"Image size: {IMG_WIDTH}x{IMG_HEIGHT}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Learning rate: {BASE_LR}")
    print(f"Classes: {CLASS_NAMES}")
    print("=" * 60)

    # Transforms
    common_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        common_norm,
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        common_norm,
    ])

    # Datasets
    print("\nLoading datasets...")
    train_dataset = DentexDataset(TRAIN_DIR, transform=train_transform)
    val_dataset   = DentexDataset(VAL_DIR,  transform=val_transform)
    test_dataset  = DentexDataset(TEST_DIR, transform=val_transform)

    # Class counts from train set
    train_labels = [lbl for _, lbl in train_dataset.samples]
    class_counts = Counter(train_labels)
    counts_list = [class_counts.get(c, 1) for c in range(len(CLASS_NAMES))]
    print(f"Class counts: {dict(class_counts)}")

    # Class-Balanced alpha from effective number of samples (beta close to 1)
    # Try beta in {0.99, 0.995, 0.999}. Larger -> stronger minority upweight.
    alpha = class_balanced_alpha(effective_beta=0.995, class_counts=counts_list)
    print(f"CB-Focal alpha (per-class): {alpha.tolist()}")

    # Dataloaders (no sampler; use regular shuffling)
    pin_mem = (DEVICE.type == "cuda")
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=pin_mem
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=pin_mem
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=pin_mem
    )

    # Model, loss, optimizer, scheduler
    print("\nInitializing model...")
    model = SimpleCNN(num_classes=len(CLASS_NAMES)).to(DEVICE)

    # CB-Focal (gamma tweakable: 1.5, 2.0)
    criterion = FocalLoss(gamma=1.8, alpha=alpha.to(DEVICE), reduction="mean")

    optimizer = optim.Adam(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS) if USE_SCHEDULER else None

    # TensorBoard visualizer with unique run dir
    run_id = time.strftime("%Y%m%d-%H%M%S")
    tb_dir = os.path.join(RUNS_DIR, f"dentex_{MODEL_NAME}_cbfocal_{run_id}")
    visualizer = ModelVisualizer(log_dir=tb_dir, use_tsne=True)
    visualizer.class_names = CLASS_NAMES

    # Summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    start_time = time.time()
    train_hist, val_hist = [], []
    best_val = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} ({time.time()-start_time:.1f}s so far)")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Reset per-epoch buffers for projector
        visualizer.embeddings.clear()
        visualizer.labels.clear()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch + 1)
        val_loss, val_acc, y_val_true, _ = evaluate(model, val_loader, criterion, DEVICE, visualizer, desc=f"Epoch {epoch+1} [Val]")

        # Log scalars + embeddings
        visualizer.log_metrics(epoch, train_loss, val_loss, train_acc, val_acc)
        visualizer.log_embeddings(epoch)
        visualizer.writer.flush()

        # Accumulate for t-SNE
        visualizer.all_embeddings.extend(visualizer.embeddings)  # list of [B,D]
        visualizer.all_labels.extend(visualizer.labels)          # list of [B]

        train_hist.append((train_loss, train_acc))
        val_hist.append((val_loss, val_acc))

        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"  Val   Loss: {val_loss:.4f},   Val Acc: {val_acc:.4f} ({val_acc*100:.2f}%)")

        if scheduler is not None:
            scheduler.step()

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "simple_cnn_best.pth"))
            print(f"  ✓ Saved best model (val_acc: {val_acc:.4f})")

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Training completed in {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print("=" * 60)

    # Load best
    state = torch.load(os.path.join(MODEL_DIR, "simple_cnn_best.pth"), map_location=DEVICE)
    model.load_state_dict(state)

    # Test
    print("\nEvaluating on test set...")
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, DEVICE, desc="Testing")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

    # One sample for feature maps
    sample_img, _ = test_dataset[0]
    sample_img = sample_img.unsqueeze(0).to(DEVICE)  # [1,C,H,W]

    final_step = NUM_EPOCHS
    fm1 = os.path.join(METRICS_DIR, "conv1_feature_maps.png")
    visualizer.visualize_feature_maps(model, sample_img, "conv1", save_path=fm1, model_name=MODEL_NAME, step=final_step)
    fm2 = os.path.join(METRICS_DIR, "conv2_feature_maps.png")
    visualizer.visualize_feature_maps(model, sample_img, "conv2", save_path=fm2, model_name=MODEL_NAME, step=final_step)

    cm_path = os.path.join(METRICS_DIR, f"{MODEL_NAME}_confusion_matrix.png")
    visualizer.plot_confusion_matrix(y_true, y_pred, CLASS_NAMES, cm_path, model_name=MODEL_NAME, step=final_step)

    # t-SNE (now populated)
    if visualizer.all_embeddings and visualizer.all_labels:
        all_features = torch.cat(visualizer.all_embeddings)
        all_labels_t = torch.cat(visualizer.all_labels)
        fs_path = os.path.join(METRICS_DIR, f"{MODEL_NAME}_feature_space.png")
        visualizer.plot_feature_space(all_features, all_labels_t, save_path=fs_path, model_name=MODEL_NAME)
    else:
        print("Skipping t-SNE: no accumulated features/labels found.")

    # Loss curves
    train_losses = [t[0] for t in train_hist]
    val_losses   = [v[0] for v in val_hist]
    loss_png = os.path.join(METRICS_DIR, f"{MODEL_NAME}_loss_curves.png")
    visualizer.plot_loss_curves(train_losses, val_losses, save_path=loss_png, model_name=MODEL_NAME, step=final_step)

    # Close writer
    visualizer.remove_hook()
    visualizer.close()

    # Report
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0)
    print(report)

    # Save results
    results_path = os.path.join(METRICS_DIR, f"{MODEL_NAME}_results.txt")
    with open(results_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("Simple CNN - Tooth Disease Classification Results (CB-Focal)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Device: {DEVICE}\n")
        f.write(f"Image size: {IMG_WIDTH}x{IMG_HEIGHT}\n")
        f.write(f"Batch size: {BATCH_SIZE}\n")
        f.write(f"Epochs: {NUM_EPOCHS}\n")
        f.write(f"Learning rate: {BASE_LR}\n")
        f.write(f"Classes: {CLASS_NAMES}\n")
        f.write(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}\n")
        f.write(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")
        f.write(f"\nTraining time: {elapsed:.1f}s ({elapsed/60:.1f}m)\n")
        f.write(f"\n{'=' * 60}\nTraining History\n{'=' * 60}\n")
        for i, ((t_loss, t_acc), (v_loss, v_acc)) in enumerate(zip(train_hist, val_hist), 1):
            f.write(f"Epoch {i}:\n")
            f.write(f"  Train Loss: {t_loss:.4f}, Train Acc: {t_acc:.4f}\n")
            f.write(f"  Val   Loss: {v_loss:.4f}, Val   Acc: {v_acc:.4f}\n")
        f.write(f"\n{'=' * 60}\nTest Set Results\n{'=' * 60}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)\n\n")
        f.write(f"{report}\n")
        f.write("=" * 60 + "\n")

    print(f"\nResults saved to {results_path}")
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
