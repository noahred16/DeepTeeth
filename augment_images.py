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
INPUT_DIR = "data"
OUTPUT_DIR = "data_augmented"
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


def process_images(input_dir, output_dir, target_width, target_height, padding_color):
    """
    Process all images in input directory and save to output directory.

    Args:
        input_dir: Path to input directory
        output_dir: Path to output directory
        target_width: Target width in pixels
        target_height: Target height in pixels
        padding_color: RGB tuple for padding color
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get list of all image files
    input_path = Path(input_dir)
    image_extensions = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}
    image_files = [
        f for f in input_path.iterdir() if f.is_file() and f.suffix in image_extensions
    ]

    print(f"Found {len(image_files)} images to process")
    print(f"Target dimensions: {target_width}x{target_height}")
    print(f"Padding color: RGB{padding_color}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Process each image
    successful = 0
    failed = 0
    skipped = 0

    for img_file in tqdm(image_files, desc="Processing images"):
        try:
            # Check if output file already exists
            output_path = Path(output_dir) / img_file.name
            if output_path.exists():
                skipped += 1
                continue

            # Open image
            img = Image.open(img_file)

            # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Resize with padding
            processed_img = resize_with_padding(
                img, target_width, target_height, padding_color
            )

            # Save to output directory with same filename
            processed_img.save(output_path, "PNG")

            successful += 1

        except Exception as e:
            print(f"\nError processing {img_file.name}: {str(e)}")
            failed += 1

    # Print summary
    print()
    print("=" * 50)
    print("Processing Complete!")
    print(f"Successfully processed: {successful}")
    print(f"Skipped (already exist): {skipped}")
    print(f"Failed: {failed}")
    print(f"Output saved to: {output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    process_images(INPUT_DIR, OUTPUT_DIR, TARGET_WIDTH, TARGET_HEIGHT, PADDING_COLOR)
