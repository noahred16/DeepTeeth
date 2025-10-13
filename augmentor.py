"""
DeepTeeth — augmentor.py

Runtime data augmentations for dental X‑ray classification using Albumentations.
Designed to plug into your existing DataLoader pipeline after your
resize/pad step. Safe defaults preserve diagnostic structure.

Usage (example):
    from augmentor import build_transforms
    train_tf, val_tf = build_transforms(img_size=512, to_3ch=True)

    # In your Dataset.__getitem__
    #   img: numpy array (H, W) or (H, W, C) in [0..255]
    #   label: int
    sample = train_tf(image=img)
    tensor = sample["image"]   # torch.FloatTensor shape (C, H, W)

Requires:
    pip install albumentations==1.* opencv-python-headless==4.*
    (and albumentations.pytorch.ToTensorV2)
"""
from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ----------------------------
# Helpers
# ----------------------------

def _replicate_to_3ch() -> A.BasicTransform:
    """Ensure image has 3 channels by replicating a single channel.
    Works for (H, W) or (H, W, 1) arrays.
    """
    return A.Lambda(
        name="replicate_to_3ch",
        image=lambda x: (
            np.repeat(x[..., None], 3, axis=2)
            if x.ndim == 2
            else (x if x.shape[2] == 3 else np.repeat(x[..., :1], 3, axis=2))
        ),
    )


def _get_norm(norm: str, to_3ch: bool) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Return (mean, std) by scheme.
    norm in {"custom", "imagenet"}
    """
    if norm.lower() == "imagenet":
        # ImageNet stats
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        if not to_3ch:
            # Collapse to grayscale-equivalent averages if staying 1ch
            m = float(np.mean(mean))
            s = float(np.mean(std))
            return (m,), (s,)
        return mean, std
    # Custom grayscale-friendly stats
    return (0.5,) * (3 if to_3ch else 1), (0.25,) * (3 if to_3ch else 1)


# ----------------------------
# Public API
# ----------------------------

def build_transforms(
    img_size: int = 512,
    *,
    to_3ch: bool = True,
    norm: str = "custom",  # or "imagenet"
    use_clahe: bool = True,
    rotate_limit: int = 8,
    scale_limit: float = 0.10,
    shift_limit: float = 0.05,
    p_noise: float = 0.20,
    p_gamma: float = 0.40,
    p_bc: float = 0.40,
    p_ssr: float = 0.60,
    p_hflip: float = 0.00,  # keep 0 by default; enable only if orientation is label-invariant
) -> Tuple[A.Compose, A.Compose]:
    """Build train and validation transforms.

    Parameters
    ----------
    img_size : int
        Final square size; increase to 768 if GPU allows.
    to_3ch : bool
        Replicate grayscale to 3 channels (useful for ImageNet-pretrained CNNs).
    norm : str
        Normalization scheme: "custom" or "imagenet".
    use_clahe : bool
        Apply light CLAHE to stabilize local contrast across exposures.
    rotate_limit, scale_limit, shift_limit :
        Geometric jitter limits for ShiftScaleRotate.
    p_noise, p_gamma, p_bc, p_ssr, p_hflip : float
        Probabilities for noise, gamma, brightness/contrast, SSR, horizontal flip.

    Returns
    -------
    train_tf, val_tf : albumentations.Compose
        Call with dict(image=np.ndarray) to obtain transformed image tensor.
    """
    mean, std = _get_norm(norm, to_3ch=to_3ch)

    base_resize = A.Resize(img_size, img_size, interpolation=1)  # cv2.INTER_LINEAR

    # Photometric block — safe for dental X-rays
    photometric = [
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.60) if use_clahe else A.NoOp(),
        A.RandomGamma(gamma_limit=(90, 110), p=p_gamma),
        A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.15, p=p_bc),
        A.GaussNoise(var_limit=(5.0, 15.0), p=p_noise),
    ]

    # Geometric block — conservative
    geometric = [
        A.ShiftScaleRotate(
            shift_limit=shift_limit,
            scale_limit=scale_limit,
            rotate_limit=rotate_limit,
            border_mode=0,  # cv2.BORDER_CONSTANT; pad with zeros (normalize mitigates edges)
            p=p_ssr,
        ),
        A.HorizontalFlip(p=p_hflip),  # keep disabled unless safe for your labels
    ]

    # Channel handling
    ch = [_replicate_to_3ch()] if to_3ch else []

    train_tf = A.Compose(
        [
            *photometric,
            *geometric,
            base_resize,
            *ch,
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    # Validation — deterministic, no random jitter
    val_tf = A.Compose(
        [
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0) if use_clahe else A.NoOp(),
            base_resize,
            *ch,
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    return train_tf, val_tf


# ----------------------------
# CLI preview utility (optional)
# ----------------------------
if __name__ == "__main__":
    import argparse
    import cv2

    parser = argparse.ArgumentParser(description="Preview DeepTeeth augmentations.")
    parser.add_argument("--src", type=str, default="data", help="Folder with PNG/JPG images (recursively scanned)")
    parser.add_argument("--out", type=str, default="figures/aug_preview", help="Output directory for previews")
    parser.add_argument("--n", type=int, default=8, help="Number of images to preview")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--to_3ch", action="store_true", help="Replicate grayscale to 3 channels")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    train_tf, _ = build_transforms(img_size=args.img_size, to_3ch=args.to_3ch)

    # Gather images
    paths = []
    for root, _, files in os.walk(args.src):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                paths.append(os.path.join(root, f))
    paths = paths[: args.n]

    for i, p in enumerate(paths):
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        # Convert to grayscale if image has alpha or multiple channels
        if img.ndim == 3 and img.shape[2] > 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Albumentations expects HWC in uint8
        img = img.astype(np.uint8)
        out = train_tf(image=img)["image"]  # torch tensor (C,H,W)
        # Convert back to HWC uint8 for saving preview
        arr = out.float().numpy()
        # de-normalize for preview (roughly)
        mean, std = _get_norm("custom", to_3ch=args.to_3ch)
        mean = np.array(mean)[:, None, None]
        std = np.array(std)[:, None, None]
        arr = (arr * std + mean) * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[2] == 1:
            arr = arr[..., 0]
        cv2.imwrite(os.path.join(args.out, f"aug_{i}.png"), arr)

    print(f"Saved {len(paths)} preview(s) to {args.out}")
