import os
import random
import shutil
import math
from sklearn.model_selection import train_test_split

# do a train/val/test split
# keep the same distribution of classes in each split
# 70% train, 15% val, 15% test
# good library options: sklearn.model_selection.train_test_split, iterstrat

# directory structure:
# data_train
# data_validation
# data_test

# data sourced from data.
# All files in the data directory follow the format:
# sourcetype_classname_idx_imagefilename.png
# where sourcetype is one of "train", "val", or "test" but doesnt matter here. we are re-doing the split

source_dir = "./data/"
train_dir = "./data_train/"
val_dir = "./data_validation/"
test_dir = "./data_test/"

super_classes = {
    "Caries": ["Caries", "CariesTest"],
    "DeepCaries": ["DeepCaries", "Curettage"],
    "Impacted": ["Impacted"],
    "Lesion": ["PeriapicalLesion", "Lesion"],
    "RootCanal": ["RootCanal"],
    "Healthy": ["Intact"],
}
excluded_classes = ["Extraction", "Fracture"]


def split_data(super_classes, excluded_classes):
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # read all files in source_dir
    all_files = [f for f in os.listdir(source_dir) if f.endswith(".png")]

    # group files by super_class
    class_files = {}
    for filename in all_files:
        parts = filename.split("_")
        class_name = parts[1]

        # Skip excluded classes
        if class_name in excluded_classes:
            continue

        # Find which superclass this class belongs to
        super_class = None
        for sc_name, sc_classes in super_classes.items():
            if class_name in sc_classes:
                super_class = sc_name
                break

        if not super_class:
            continue

        if super_class not in class_files:
            class_files[super_class] = []
        class_files[super_class].append(filename)

    # split each class into train/val/test
    train_files = []
    val_files = []
    test_files = []

    for super_class, files in class_files.items():
        train, temp = train_test_split(files, test_size=0.3, random_state=42)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)
        train_files.extend(train)
        val_files.extend(val)
        test_files.extend(test)

    # copy files to respective directories
    for filename in train_files:
        shutil.copy(
            os.path.join(source_dir, filename), os.path.join(train_dir, filename)
        )
    for filename in val_files:
        shutil.copy(os.path.join(source_dir, filename), os.path.join(val_dir, filename))
    for filename in test_files:
        shutil.copy(
            os.path.join(source_dir, filename), os.path.join(test_dir, filename)
        )


# after we do the split, we will augment that training data to balance the classes.

# even out the class distribution by generating synthetic images for the minority classes
# use rotations, flips, brightness changes (+-10-20%)

# 2 flips (horizontal, vertical)
# 4 rotations for each flip
# 4 brightness changes for each rotation
# 2 * 4 * 4 = 32 augmentations per image

# lets go ahead an upscall as much as possible using these combinations.
# so smallest class would be fracture, 11. we can get 11 * 32 = 352 images for fracture.

# Follow naming convention
# ex. validation_Caries_49_val_31.png
# source_class_index_split_index.png

# balance out the data_augmented directory using image augmentations.
# create a new directory data_balanced
# use combinations of flips, rotations, brightness changes to augment images
# to balance out the classes.
# target is to upscall the minority classes to have up to the majority class size.
# will need to calculate that majority class size first.

# we should use super classes to do this.

# don't include these, too small.
# Extraction
# Fracture


def get_class_distribution(directory, super_classes, excluded_classes):
    class_counts = {key: 0 for key in super_classes.keys()}
    for filename in os.listdir(directory):
        if not filename.endswith(".png"):
            continue
        parts = filename.split("_")
        class_name = parts[1]
        for super_class, sub_classes in super_classes.items():
            if class_name in sub_classes and class_name not in excluded_classes:
                class_counts[super_class] += 1
                break
    return class_counts


if __name__ == "__main__":
    split_data(super_classes, excluded_classes)

    data_sets = {
        "train": train_dir,
        "validation": val_dir,
        "test": test_dir,
    }

    for set_name, directory in data_sets.items():
        print(f"\nClass distribution in {set_name} set:")
        class_distribution = get_class_distribution(
            directory, super_classes, excluded_classes
        )
        total = sum(class_distribution.values())
        for class_name, count in class_distribution.items():
            print(f"  {class_name}: {count:,} ({count / total:.1%})")
