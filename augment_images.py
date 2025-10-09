#!/usr/bin/env python3
"""
Image augmentation script to resize images to consistent 160x256 dimensions.
Preserves aspect ratio by adding grey padding to borders.
"""

import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Configuration
INPUT_DIRS = ["data_train", "data_validation", "data_test"]
TARGET_WIDTH = 160
TARGET_HEIGHT = 256
PADDING_COLOR = (128, 128, 128)  # Grey color


def resize_with_padding(image, target_width, target_height, padding_color):
    """
    Resize image to target dimensions while preserving aspect ratio.
    Adds grey padding to fill the remaining space.

    Args:
        image: PIL Image object
        target_width: Target width in pixels
        target_height: Target height in pixels
        padding_color: RGB tuple for padding color

    Returns:
        PIL Image object with target dimensions
    """
    # Get original dimensions
    orig_width, orig_height = image.size

    # Calculate aspect ratios
    target_aspect = target_width / target_height
    orig_aspect = orig_width / orig_height

    # Determine scaling to fit within target dimensions
    if orig_aspect > target_aspect:
        # Image is wider - scale based on width
        new_width = target_width
        new_height = int(target_width / orig_aspect)
    else:
        # Image is taller - scale based on height
        new_height = target_height
        new_width = int(target_height * orig_aspect)

    # Resize image
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)

    # Create new image with target dimensions and grey background
    new_image = Image.new("RGB", (target_width, target_height), padding_color)

    # Calculate position to paste resized image (center it)
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2

    # Paste resized image onto grey background
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image


def process_images(input_dir, target_width, target_height, padding_color):
    """
    Process all images in input directory in place.

    Args:
        input_dir: Path to input directory
        target_width: Target width in pixels
        target_height: Target height in pixels
        padding_color: RGB tuple for padding color
    """

    # Get list of all image files
    input_path = Path(input_dir)
    image_extensions = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}
    image_files = [
        f for f in input_path.iterdir() if f.is_file() and f.suffix in image_extensions
    ]

    print(f"Found {len(image_files)} images to process in {input_dir}")
    print(f"Target dimensions: {target_width}x{target_height}")
    print(f"Padding color: RGB{padding_color}")
    print()

    # Process each image
    successful = 0
    failed = 0
    skipped = 0

    for img_file in tqdm(image_files, desc=f"Processing {input_dir}"):
        try:
            # Open image
            img = Image.open(img_file)

            # Check if image already has target dimensions
            if img.size == (target_width, target_height):
                skipped += 1
                continue

            # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Resize with padding
            processed_img = resize_with_padding(
                img, target_width, target_height, padding_color
            )

            # Save in place, overwriting the original file
            processed_img.save(img_file, "PNG")

            successful += 1

        except Exception as e:
            print(f"\nError processing {img_file.name}: {str(e)}")
            failed += 1

    # Print summary
    print()
    print("=" * 50)
    print(f"Processing Complete for {input_dir}!")
    print(f"Successfully processed: {successful}")
    print(f"Skipped (already correct size): {skipped}")
    print(f"Failed: {failed}")
    print("=" * 50)

    return successful, skipped, failed


if __name__ == "__main__":
    print("Starting image augmentation process...")
    print(f"Processing {len(INPUT_DIRS)} directories")
    print()

    total_successful = 0
    total_skipped = 0
    total_failed = 0

    for input_dir in INPUT_DIRS:
        if not os.path.exists(input_dir):
            print(f"Warning: Directory {input_dir} does not exist. Skipping...")
            print()
            continue

        successful, skipped, failed = process_images(
            input_dir, TARGET_WIDTH, TARGET_HEIGHT, PADDING_COLOR
        )
        total_successful += successful
        total_skipped += skipped
        total_failed += failed
        print()

    # Print overall summary
    print("\n" + "=" * 50)
    print("OVERALL SUMMARY")
    print("=" * 50)
    print(f"Total directories processed: {len(INPUT_DIRS)}")
    print(f"Total images successfully processed: {total_successful}")
    print(f"Total images skipped: {total_skipped}")
    print(f"Total images failed: {total_failed}")
    print("=" * 50)
