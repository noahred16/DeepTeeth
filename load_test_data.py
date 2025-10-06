import os
import json
from PIL import Image

# see readme for download instructions. Replace with your path to DENTEX
DENTEX_PATH = "~/repos/DENTEX"  #

# directory to save processed, cropped annotation files
processed_dir = "./data/"

# test images directory (ex. test_0.png, test_1.png, ...)
test_images_dir = DENTEX_PATH + "/DENTEX/disease/input/"

# corresponding test annotations directory (ex. test_0.json, test_1.json, ...)
test_labels_dir = DENTEX_PATH + "/DENTEX/disease/label/"

# For test data, each file has a corresponding json file structure:
""" test_0.json
{
    "version": "4.5.5",
    "flags": {},
    "shapes": [
    {
            "label": "3-kanal-36",
            "points": [
                [1837.0, 849.0],
                [1859.0, 886.0],
                [1878.0, 955.0],
                [1895.0, 1026.0],
                [1893.0, 1099.0],
                [1865.0, 1106.0],
                [1815.0, 1116.0],
                [1767.0, 1025.0],
                [1753.0, 951.0],
                [1712.0, 892.0],
                [1728.0, 858.0],
                [1785.0, 847.0]
            ],
            "group_id": null,
            "shape_type": "polygon",
            "flags": {}
        },
        ... around between 1-20 annotations per image
    ],
    "imagePath": "test_0.png",
    "imageData": null,
    "imageHeight": 1316,
    "imageWidth": 2829
}
"""

# categories = shapes.label

# use a set to avoid duplicates
categories = set()

# 0-saglam
# 1-çürük
# 2-küretaj
# 3-kanal
# 5-çekim
# 6-gömülü
# 7-lezyon
# 8-kirik
translation_dict = {
    "saglam": "Intact",
    "\u00e7\u00fcr\u00fck": "CariesTest",
    "k\u00fcretaj": "Curettage",  # (deep caries requiring surgical cleaning)
    "kanal": "RootCanal",
    "\u00e7ekim": "Extraction",
    "g\u00f6m\u00fcl\u00fc": "Impacted",
    "lezyon": "Lesion",
    "k\u0131r\u0131k": "Fracture",
}

category_count = {
    "Intact": 0,
    "CariesTest": 0,
    "Curettage": 0,
    "RootCanal": 0,
    "Extraction": 0,
    "Impacted": 0,
    "Lesion": 0,
    "Fracture": 0,
}

# loop through all files in test_images_dir (250)
for filename in sorted(os.listdir(os.path.expanduser(test_images_dir))):
    print("Processing file:", filename)

    if not filename.endswith(".png"):
        print(f"skipping {filename}")
        continue

    # find corresponding json file in test_labels_dir
    json_filename = filename.replace(".png", ".json")
    json_path = os.path.join(os.path.expanduser(test_labels_dir), json_filename)

    with open(json_path, "r") as f:
        data = json.load(f)
        # print(f"  Found {len(data['shapes'])} annotations in {json_filename}")

        for shape in data["shapes"]:
            label = shape["label"]
            # label has format "0-saglam-11" or "1-çürük-32"
            if label not in categories:
                class_id = label.split("-")[0]
                disease_name = label.split("-")[1]
                tooth_number = label.split("-")[2]
                categories.add(disease_name)

            # use translation_dict to get english name
            # idx is count of that disease so far
            # "test"_class_idx_imagefilename.

            class_name = translation_dict[disease_name]
            category_count[class_name] += 1
            class_name_idx = category_count[class_name]
            new_filename = f"test_{class_name}_{class_name_idx}_{filename}"

            # use the points to get bounding box
            points = shape["points"]
            x_coordinates = [point[0] for point in points]
            y_coordinates = [point[1] for point in points]
            x_min = min(x_coordinates)
            x_max = max(x_coordinates)
            y_min = min(y_coordinates)
            y_max = max(y_coordinates)
            bbox = [x_min, y_min, x_max, y_max]
            # print(f"  Label: {label}, Class name: {class_name}, BBox: {bbox}")

            # use the bbox to crop the image and save it to new_filename use processed_dir
            image_path = os.path.join(os.path.expanduser(test_images_dir), filename)
            with Image.open(image_path) as img:
                # bbox is [x_min, y_min, x_max, y_max]
                x_min, y_min, x_max, y_max = bbox
                cropped_img = img.crop((x_min, y_min, x_max, y_max))
                # save to processed_dir
                cropped_img.save(os.path.join(processed_dir, new_filename))
                print(
                    f"   Saved annotated image to {os.path.join(processed_dir, new_filename)}"
                )

# total count
total_count = sum(category_count.values())
print("")
print(f"Total annotated images: {total_count}")

# totals
print("")
print("Category counts:")
for category, count in category_count.items():
    print(f"{category}: {count} ({(count/total_count)*100:.1f}%)")
