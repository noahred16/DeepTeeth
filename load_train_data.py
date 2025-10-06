import os
import json
from PIL import Image

# see readme for download instructions.
DENTEX_PATH = "~/repos/DENTEX"  # REPLACE WITH PATH TO DENTEX

# directory to save processed, cropped annotation files
processed_dir = "./data/"

# quadrant-enumeration-disease is the one that has the disease annotations
sources = {
    "train": (
        DENTEX_PATH + "/DENTEX/training_data/quadrant-enumeration-disease/xrays/",
        DENTEX_PATH
        + "/DENTEX/training_data/quadrant-enumeration-disease/train_quadrant_enumeration_disease.json",
    ),
    "validation": (
        DENTEX_PATH + "/DENTEX/validation_data/quadrant_enumeration_disease/xrays/",
        DENTEX_PATH + "/DENTEX/validation_triple.json",
    ),
}

# Json file structure:
"""
{
    "images": [
        {
            "height": 1316,
            "width": 2744,
            "id": 1,
            "file_name": "train_673.png"
        },
        ... 705 TOTAL
    ],
    "annotations": [
        {
            "iscrowd": 0,
            "image_id": 1,
            "bbox": [
                542.0, 698.0, 220.0, 271.0
            ],
            "segmentation": [
                [621, 703, 573, 744, 542, 885, 580, 945, 650, 969, 711, 883, 762, 807, 748, 741, 649, 698]
            ],
            "id": 1,
            "area": 39683,
            "category_id_1": 3,
            "category_id_2": 7,
            "category_id_3": 0
        },
        ... 705 TOTAL
    ],
    "categories_1": [... not relevent], # segment the 4 quadrants of the mouth
    "categories_2": [... not relevent], # identify individual teeth (1-8 per quadrant)
    # Detect quadrants + enumerate teeth + diagnose diseases on abnormal teeth
    "categories_3": [
        {
            "id": 0,
            "name": "Impacted",
            "supercategory": "Impacted"
        },
        {
            "id": 1,
            "name": "Caries",
            "supercategory": "Caries"
        },
        {
            "id": 2,
            "name": "Periapical Lesion",
            "supercategory": "Periapical Lesion"
        },
        {
            "id": 3,
            "name": "Deep Caries",
            "supercategory": "Deep Caries"
        }
    ]
}
"""


# train_dir -> data_source
# train_json -> annotation_file
# "train" -> source_type
def process_dir(data_source, annotation_file, source_type):

    with open(os.path.expanduser(annotation_file), "r") as f:
        train_data = json.load(f)

        # images
        print("Number of image labels:", len(train_data["images"]))  # 705

        print(
            "Number of image files:", len(os.listdir(os.path.expanduser(data_source)))
        )  # 705

        # annotations
        print("Number of annotations:", len(train_data["annotations"]))  # 1856

    print("Train directory:", os.path.expanduser(data_source))

    print("")

    categories = ["Impacted", "Caries", "Periapical Lesion", "Deep Caries"]

    category_count = {
        "Impacted": 0,
        "Caries": 0,
        "Periapical Lesion": 0,
        "Deep Caries": 0,
    }

    for filename in sorted(os.listdir(os.path.expanduser(data_source))):
        if not filename.endswith(".png"):
            print(f"skipping {filename}")
            continue

        print("Processing file:", filename)
        # print("Full path:", os.path.join(os.path.expanduser(data_source), filename))

        # get json image json entry
        # use filename to find the corresponding entry in train_data["images"]
        image_entry = next(
            (item for item in train_data["images"] if item["file_name"] == filename),
            None,
        )
        if image_entry is None:
            print(f"Could not find entry for {filename} in json")
            break

        # print("Image entry:", image_entry)
        image_id = image_entry["id"]

        # use image_id to find all corresponding annotations
        annotations = [
            item for item in train_data["annotations"] if item["image_id"] == image_id
        ]

        # print(f"Found {len(annotations)} annotations for image id {image_id}")
        for annotation in annotations:

            # use category_id_3 to get disease name
            category_id_3 = annotation["category_id_3"]
            category_entry = next(
                (
                    item
                    for item in train_data["categories_3"]
                    if item["id"] == category_id_3
                ),
                None,
            )
            class_name = category_entry["name"]

            # use bbox to get bounding box
            bbox = annotation["bbox"]

            # update count
            category_count[class_name] += 1

            # save file sourcetype_classname_idx_imagefilename.png
            class_name_idx = category_count[class_name]

            # classname should remove spaces
            file_class_name = class_name.replace(" ", "")
            new_filename = (
                f"{source_type}_{file_class_name}_{class_name_idx}_{filename}"
            )

            # use the bbox to crop the image and save it to new_filename use processed_dir
            image_path = os.path.join(os.path.expanduser(data_source), filename)
            with Image.open(image_path) as img:
                # bbox is [x, y, width, height]
                x, y, width, height = bbox
                cropped_img = img.crop((x, y, x + width, y + height))
                # save to current directory
                cropped_img.save(os.path.join(processed_dir, new_filename))
                print(
                    f"   Saved annotated image to {os.path.join(processed_dir, new_filename)}"
                )

    # total count
    total_count = sum(category_count.values())
    print("")
    print(f"Total annotated images: {total_count}")

    # print info about category_count
    print("")
    print("Category counts:")
    for category, count in category_count.items():
        print(f"{category}: {count} ({(count/total_count)*100:.1f}%)")


# process both train and validation sets
for source_type, (data_source, annotation_file) in sources.items():
    print("")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f"~~~~~~~~~~~ Processing {source_type} set ~~~~~~~~~~~~")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    process_dir(data_source, annotation_file, source_type)
