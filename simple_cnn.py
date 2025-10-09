#!/usr/bin/env python3
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
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

# Configuration
TRAIN_DIR = "data_balanced_train"
VAL_DIR = "data_validation"
TEST_DIR = "data_test"
MODEL_DIR = "models"
METRICS_DIR = "metrics"
IMG_HEIGHT = 256
IMG_WIDTH = 160
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Superclass definitions
SUPER_CLASSES = {
    "Caries": ["Caries", "CariesTest"],
    "DeepCaries": ["DeepCaries", "Curettage"],
    "Impacted": ["Impacted"],
    "Lesion": ["PeriapicalLesion", "Lesion"],
    "RootCanal": ["RootCanal"],
    "Healthy": ["Intact"],
}
EXCLUDED_CLASSES = ["Extraction", "Fracture"]

# Create class to index mapping
CLASS_NAMES = sorted(SUPER_CLASSES.keys())
CLASS_TO_IDX = {class_name: idx for idx, class_name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: class_name for class_name, idx in CLASS_TO_IDX.items()}


def get_superclass(class_name):
    """Map a class name to its superclass."""
    for superclass, classes in SUPER_CLASSES.items():
        if class_name in classes:
            return superclass
    return None


def parse_filename(filename):
    """
    Parse filename in format: sourcetype_classname_idx_imagefilename.png
    Returns: (sourcetype, classname, superclass)
    """
    parts = filename.split("_")
    if len(parts) >= 2:
        sourcetype = parts[0]
        classname = parts[1]
        superclass = get_superclass(classname)
        return sourcetype, classname, superclass
    return None, None, None


class DentexDataset(Dataset):
    """PyTorch Dataset for DENTEX tooth disease classification."""

    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir: Directory containing the image files
            transform: Optional transforms to apply to images
        """
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []

        # Load all valid samples
        for filename in os.listdir(data_dir):
            if not filename.endswith(".png"):
                continue

            sourcetype, classname, superclass = parse_filename(filename)

            # Skip excluded classes
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

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label


class SimpleCNN(nn.Module):
    """Very lightweight CNN for tooth disease classification."""

    def __init__(self, num_classes=6):
        super(SimpleCNN, self).__init__()

        # Convolutional layers - reduced to 2 layers with fewer channels
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 160x256 -> 80x128

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 80x128 -> 40x64

        # More aggressive pooling to reduce spatial dimensions
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 40x64 -> 20x32

        # Calculate flattened size: 32 channels * 20 * 32 = 20480
        self.fc1 = nn.Linear(32 * 20 * 32, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        # Conv block 1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)

        # Conv block 2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)

        # Additional pooling
        x = self.pool3(x)

        # Flatten and FC layers
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch_num):
    """Train for one epoch."""
    from tqdm import tqdm

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch_num} [Train]", leave=False)

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Update progress bar
        current_acc = correct / total
        progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{current_acc:.4f}")

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device, desc="Eval"):
    """Evaluate model on validation/test set."""
    from tqdm import tqdm

    model.eval()
    running_loss = 0.0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=desc, leave=False)
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            # Update progress bar
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = running_loss / len(all_labels)
    accuracy = accuracy_score(all_labels, all_predictions)

    return avg_loss, accuracy, all_labels, all_predictions


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Generate and save confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix - Simple CNN")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def main():
    # Create output directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    print("=" * 60)
    print("Simple CNN for Tooth Disease Classification")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Image size: {IMG_WIDTH}x{IMG_HEIGHT}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Classes: {CLASS_NAMES}")
    print("=" * 60)

    # Define transforms
    train_transform = transforms.Compose(
        [
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create datasets
    print("\nLoading datasets...")
    train_dataset = DentexDataset(TRAIN_DIR, transform=train_transform)
    val_dataset = DentexDataset(VAL_DIR, transform=val_transform)
    test_dataset = DentexDataset(TEST_DIR, transform=val_transform)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    # Initialize model, loss, optimizer
    print("\nInitializing model...")
    model = SimpleCNN(num_classes=len(CLASS_NAMES)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    start_time = time.time()
    train_history = []
    val_history = []
    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE, epoch + 1
        )

        # Validate
        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, DEVICE, desc=f"Epoch {epoch+1} [Val]"
        )

        # Record history
        train_history.append((train_loss, train_acc))
        val_history.append((val_loss, val_acc))

        epoch_time = time.time() - epoch_start

        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} ({epoch_time:.1f}s)")
        print(
            f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} ({train_acc*100:.2f}%)"
        )
        print(
            f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} ({val_acc*100:.2f}%)"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(MODEL_DIR, "simple_cnn_best.pth")
            torch.save(model.state_dict(), model_path)
            print(f"  ✓ Saved best model (val_acc: {val_acc:.4f})")

    training_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Training completed in {training_time:.1f}s ({training_time/60:.1f}m)")
    print("=" * 60)

    # Load best model for evaluation
    model.load_state_dict(
        torch.load(os.path.join(MODEL_DIR, "simple_cnn_best.pth"), weights_only=True)
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc, y_true, y_pred = evaluate(
        model, test_loader, criterion, DEVICE, desc="Testing"
    )
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

    # Generate classification report
    print("\nClassification Report:")
    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0
    )
    print(report)

    # Generate confusion matrix
    cm_path = os.path.join(METRICS_DIR, "simple_cnn_confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, CLASS_NAMES, cm_path)

    # Save results to file
    results_path = os.path.join(METRICS_DIR, "simple_cnn_results.txt")
    with open(results_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("Simple CNN - Tooth Disease Classification Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Device: {DEVICE}\n")
        f.write(f"Image size: {IMG_WIDTH}x{IMG_HEIGHT}\n")
        f.write(f"Batch size: {BATCH_SIZE}\n")
        f.write(f"Epochs: {NUM_EPOCHS}\n")
        f.write(f"Learning rate: {LEARNING_RATE}\n")
        f.write(f"Classes: {CLASS_NAMES}\n")
        f.write(f"\nModel parameters: {total_params:,}\n")
        f.write(f"Trainable parameters: {trainable_params:,}\n")
        f.write(f"\nTraining time: {training_time:.1f}s ({training_time/60:.1f}m)\n")
        f.write(f"\n{'=' * 60}\n")
        f.write("Training History\n")
        f.write("=" * 60 + "\n")
        for epoch, ((t_loss, t_acc), (v_loss, v_acc)) in enumerate(
            zip(train_history, val_history), 1
        ):
            f.write(f"Epoch {epoch}:\n")
            f.write(f"  Train Loss: {t_loss:.4f}, Train Acc: {t_acc:.4f}\n")
            f.write(f"  Val Loss: {v_loss:.4f}, Val Acc: {v_acc:.4f}\n")
        f.write(f"\n{'=' * 60}\n")
        f.write("Test Set Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)\n")
        f.write(f"\n{report}\n")
        f.write("=" * 60 + "\n")

    print(f"\nResults saved to {results_path}")
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
