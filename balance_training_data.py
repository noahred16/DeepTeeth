#!/usr/bin/env python3
"""
Balance training data class distribution by generating synthetic images for minority classes.
Uses rotations, flips, and brightness changes to augment images.

This script ONLY processes training data to prevent data leakage.
Validation and test sets are kept with their original distributions.

Augmentation strategy:
- 2 flips (no flip, horizontal flip)
- 4 rotations for each flip (0°, 90°, 180°, 270°)
- 4 brightness changes for each rotation (±10%, ±20%)
- Total: 2 * 4 * 4 = 32 augmentations per image
"""

import os
from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageEnhance
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

# Configuration
INPUT_DIR = "data_train"
OUTPUT_DIR = "data_balanced_train"
TARGET_WIDTH = 160
TARGET_HEIGHT = 256

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


def get_superclass(class_name):
    """Map a class name to its superclass."""
    for superclass, classes in SUPER_CLASSES.items():
        if class_name in classes:
            return superclass
    return None


def parse_filename(filename):
    """
    Parse filename in format: sourcetype_classname_idx_imagefilename.png
    Returns: (sourcetype, classname, idx, imagefilename, superclass)
    """
    parts = filename.split("_")
    if len(parts) >= 2:
        sourcetype = parts[0]
        classname = parts[1]
        superclass = get_superclass(classname)
        return sourcetype, classname, superclass
    return None, None, None


def analyze_class_distribution(input_dir):
    """Analyze the class distribution in the input directory."""
    superclass_files = defaultdict(list)

    for filename in os.listdir(input_dir):
        if not filename.endswith(".png"):
            continue

        sourcetype, classname, superclass = parse_filename(filename)

        # Skip excluded classes
        if classname in EXCLUDED_CLASSES:
            continue

        if superclass:
            superclass_files[superclass].append(filename)

    return superclass_files


def apply_augmentation(image, flip_horizontal, rotation_angle, brightness_factor):
    """
    Apply augmentation to an image.

    Args:
        image: PIL Image object
        flip_horizontal: Boolean - whether to flip horizontally
        rotation_angle: Rotation angle in degrees (0, 90, 180, 270)
        brightness_factor: Brightness adjustment factor (0.8, 0.9, 1.1, 1.2)

    Returns:
        Augmented PIL Image object
    """
    img = image.copy()

    # Apply horizontal flip first (if requested)
    if flip_horizontal:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # Apply rotation
    if rotation_angle != 0:
        img = img.rotate(rotation_angle, expand=False)

    # Apply brightness adjustment
    if brightness_factor != 1.0:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness_factor)

    return img


def generate_augmentation_params():
    """
    Generate all possible augmentation parameter combinations without redundancies.

    Strategy:
    - 2 flip states (no flip, horizontal flip)
    - 4 rotations (0°, 90°, 180°, 270°)
    - 4 brightness levels (0.8, 0.9, 1.1, 1.2 = ±20%, ±10%)

    Total: 2 × 4 × 4 = 32 unique augmentations

    Returns list of tuples: (flip_horizontal, rotation_angle, brightness_factor, aug_id)
    """
    augmentations = []
    aug_id = 0

    flip_states = [False, True]  # No flip, horizontal flip
    rotation_angles = [0, 90, 180, 270]
    brightness_factors = [0.8, 0.9, 1.1, 1.2]

    for flip in flip_states:
        for rotation in rotation_angles:
            for brightness in brightness_factors:
                augmentations.append((flip, rotation, brightness, aug_id))
                aug_id += 1

    return augmentations


def balance_classes(input_dir, output_dir):
    """
    Balance class distribution by augmenting minority classes in TRAINING DATA ONLY.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("Analyzing training data class distribution...")
    superclass_files = analyze_class_distribution(input_dir)

    # Print distribution
    print("\nCurrent superclass distribution in training data:")
    for superclass in sorted(
        superclass_files.keys(), key=lambda x: len(superclass_files[x]), reverse=True
    ):
        print(f"  {superclass}: {len(superclass_files[superclass])} images")

    # Determine target size (size of largest superclass)
    target_size = max(len(files) for files in superclass_files.values())
    print(f"\nTarget size per superclass: {target_size}")

    # Get all augmentation parameters
    augmentation_params = generate_augmentation_params()
    print(f"Total augmentation combinations available: {len(augmentation_params)}")

    # Process each superclass
    total_copied = 0
    total_augmented = 0

    for superclass, files in sorted(superclass_files.items()):
        print(f"\n{'='*60}")
        print(f"Processing superclass: {superclass}")
        print(f"Current size in input: {len(files)}, Target size: {target_size}")

        # First, copy all original images
        print("Copying original images...")
        for filename in tqdm(files, desc=f"Copying {superclass}"):
            src_path = Path(input_dir) / filename
            dst_path = Path(output_dir) / filename

            # Skip if already exists (idempotent)
            if dst_path.exists():
                continue

            try:
                img = Image.open(src_path)
                img.save(dst_path, "PNG")
                total_copied += 1
            except Exception as e:
                print(f"\nError copying {filename}: {str(e)}")

        # Check how many files already exist in output for this superclass
        existing_output_files = analyze_class_distribution(output_dir)
        current_count = len(existing_output_files.get(superclass, []))

        print(f"Current count in output: {current_count}")

        # Calculate how many augmented images we still need
        images_needed = target_size - current_count

        if images_needed > 0:
            print(f"Generating {images_needed} augmented images...")

            # Determine augmentation strategy
            augmentations_per_image = (images_needed // len(files)) + 1

            # If we need more augmentations than available, use all and repeat
            if augmentations_per_image > len(augmentation_params):
                print(
                    f"  Warning: Need {augmentations_per_image} augmentations per image, but only {len(augmentation_params)} available"
                )
                augmentations_per_image = len(augmentation_params)

            print(f"  Applying ~{augmentations_per_image} augmentations per image")

            # Shuffle files for random selection
            random.seed(42)
            files_to_augment = files.copy()
            random.shuffle(files_to_augment)

            augmented_count = 0
            file_idx = 0
            max_iterations = images_needed * 2  # Safety limit to prevent infinite loops
            iterations = 0

            with tqdm(total=images_needed, desc=f"Augmenting {superclass}") as pbar:
                while augmented_count < images_needed and iterations < max_iterations:
                    iterations += 1

                    # Get the next file (cycle through if needed)
                    filename = files_to_augment[file_idx % len(files_to_augment)]
                    src_path = Path(input_dir) / filename

                    # Select augmentation parameters
                    # Use different augmentations for each pass through the dataset
                    aug_idx = file_idx // len(files_to_augment)
                    if aug_idx >= len(augmentation_params):
                        aug_idx = aug_idx % len(augmentation_params)

                    flip, rotation, brightness, aug_id = augmentation_params[aug_idx]

                    # Generate output filename
                    base_name = filename.rsplit(".", 1)[0]
                    flip_str = "h" if flip else "n"  # h=horizontal flip, n=no flip
                    aug_filename = f"{base_name}_aug{aug_id}_f{flip_str}r{rotation}b{int(brightness*10)}.png"
                    dst_path = Path(output_dir) / aug_filename

                    # Skip if already exists (idempotent)
                    if dst_path.exists():
                        file_idx += 1
                        continue

                    try:
                        # Load and augment image
                        img = Image.open(src_path)
                        if img.mode != "RGB":
                            img = img.convert("RGB")

                        augmented_img = apply_augmentation(
                            img, flip, rotation, brightness
                        )
                        augmented_img.save(dst_path, "PNG")

                        augmented_count += 1
                        total_augmented += 1
                        pbar.update(1)

                    except Exception as e:
                        print(f"\nError augmenting {filename}: {str(e)}")

                    file_idx += 1

                if iterations >= max_iterations:
                    print(
                        f"\n  Warning: Reached maximum iterations. Created {augmented_count}/{images_needed} augmented images."
                    )

    # Print final summary
    print(f"\n{'='*60}")
    print("Balancing Complete!")
    print(f"Total original images copied: {total_copied}")
    print(f"Total augmented images created: {total_augmented}")
    print(
        f"Total images in balanced training dataset: {total_copied + total_augmented}"
    )
    print(f"Output saved to: {output_dir}")
    print(f"{'='*60}")

    # Print final distribution
    print("\nFinal superclass distribution in balanced training data:")
    final_superclass_files = analyze_class_distribution(output_dir)
    for superclass in sorted(
        final_superclass_files.keys(),
        key=lambda x: len(final_superclass_files[x]),
        reverse=True,
    ):
        print(f"  {superclass}: {len(final_superclass_files[superclass])} images")

    # Create pie chart for balanced distribution
    print("\nGenerating distribution pie chart...")
    create_distribution_chart(final_superclass_files, output_dir)
    print("Pie chart saved to: figures/balanced_training_super_class_distribution.png")


def create_distribution_chart(superclass_files, output_dir):
    """Create a pie chart showing the balanced superclass distribution."""
    # Prepare data
    labels = list(superclass_files.keys())
    sizes = [len(files) for files in superclass_files.values()]
    total = sum(sizes)

    # Create figure
    fig1, ax1 = plt.subplots(figsize=(10, 7))

    wedges, texts, autotexts = ax1.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=180,
        pctdistance=0.85,
        labeldistance=1.1,
    )

    # Make percentage text smaller and bold
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontsize(10)
        autotext.set_weight("bold")

    # Create legend with counts and percentages
    legend_labels = [
        f"{label}: {count:,} ({count/total*100:.1f}%)"
        for label, count in zip(labels, sizes)
    ]

    ax1.legend(
        wedges,
        legend_labels,
        title="Super Classes (Balanced Training)",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
        fontsize=10,
    )

    ax1.axis("equal")
    plt.title("Balanced Training Super Class Distribution", fontsize=14, pad=20)
    plt.tight_layout()

    # Create figures directory if it doesn't exist
    os.makedirs("figures", exist_ok=True)

    # Save the figure
    plt.savefig(
        "figures/balanced_training_super_class_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    balance_classes(INPUT_DIR, OUTPUT_DIR)
