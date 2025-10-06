import os
import matplotlib.pyplot as plt

figure_dir = "./figures/"
data_dir = "./data/"

# these are how they are labeled in huggingface. not necessarily how we will use them.
sources = {
    "train": 0,
    "validation": 0,
    "test": 0,
}

classes = {
    # Core 4
    "Caries": 0,
    "DeepCaries": 0,
    "Impacted": 0,
    "PeriapicalLesion": 0,
    # 8 classes from test set (disease)
    "Intact": 0,
    "CariesTest": 0,
    "Curettage": 0,
    "RootCanal": 0,
    "Extraction": 0,
    "Impacted": 0,
    "Lesion": 0,
    "Fracture": 0,
}


# All files in the data directory follow the format:
# sourcetype_classname_idx_imagefilename.png
# where sourcetype is one of "train", "val", or "test"


# Loop through all data in the data directory
# do some EDA on what we have
for filename in os.listdir(data_dir):
    if not filename.endswith(".png"):
        print(f"skipping {filename}")
        continue

    # ex. validation_Impacted_29_val_38.png
    # source: validation
    # class: Impacted
    # idx: 29
    # image filename: val_38.png

    parts = filename.split("_")
    source = parts[0]
    class_name = parts[1]

    # update count
    classes[class_name] += 1
    sources[source] += 1

# print results
print("Total images:", sum(sources.values()))
print("Class distribution:")
# order by count descending
classes = dict(sorted(classes.items(), key=lambda item: item[1], reverse=True))
for class_name, count in classes.items():
    print(f"  {class_name}: {count} ({count / sum(classes.values()):.1%})")

print("")
print("Source distribution:")
for source, count in sources.items():
    print(f"  {source}: {count} ({count / sum(sources.values()):.1%})")


# fun pie chart time with legend instead of labels
labels = list(classes.keys())
sizes = list(classes.values())
total = sum(sizes)

fig1, ax1 = plt.subplots(figsize=(12, 8))

# Create pie chart with class labels and spaced out labels
wedges, texts, autotexts = ax1.pie(
    sizes,
    labels=labels,  # Use class labels
    autopct="%1.1f%%",
    startangle=180,  # Rotate pie so small slices are less likely to be at the top
    pctdistance=0.85,
    labeldistance=1.2,  # Push labels outward to avoid overlap
)

# Make percentage text smaller and bold
for autotext in autotexts:
    autotext.set_color("white")
    autotext.set_fontsize(9)
    autotext.set_weight("bold")

# Create legend with counts and percentages
legend_labels = [
    f"{label}: {count} ({count/total*100:.1f}%)" for label, count in zip(labels, sizes)
]

ax1.legend(
    wedges,
    legend_labels,
    title="Classes",
    loc="center left",
    bbox_to_anchor=(1, 0, 0.5, 1),
    fontsize=9,
)

ax1.axis("equal")
plt.title("Class Distribution", fontsize=14, pad=20)
plt.tight_layout()
plt.savefig(
    os.path.join(figure_dir, "class_distribution.png"), dpi=300, bbox_inches="tight"
)


############ core 4 pie chart
core4_classes = {
    "Caries": classes["Caries"],
    "DeepCaries": classes["DeepCaries"],
    "Impacted": classes["Impacted"],
    "PeriapicalLesion": classes["PeriapicalLesion"],
}
labels = list(core4_classes.keys())
sizes = list(core4_classes.values())
total = sum(sizes)
fig1, ax1 = plt.subplots(figsize=(8, 6))
wedges, texts, autotexts = ax1.pie(
    sizes,
    labels=labels,  # Use class labels
    autopct="%1.1f%%",
    startangle=180,  # Rotate pie so small slices are less likely to be at the top
    pctdistance=0.85,
    labeldistance=1.2,  # Push labels outward to avoid overlap
)
# Make percentage text smaller and bold
for autotext in autotexts:
    autotext.set_color("white")
    autotext.set_fontsize(10)
    autotext.set_weight("bold")
# Create legend with counts and percentages
legend_labels = [
    f"{label}: {count} ({count/total*100:.1f}%)" for label, count in zip(labels, sizes)
]
ax1.legend(
    wedges,
    legend_labels,
    title="Core 4 Classes",
    loc="center left",
    bbox_to_anchor=(1, 0, 0.5, 1),
    fontsize=10,
)
ax1.axis("equal")
plt.title("Core 4 Class Distribution", fontsize=14, pad=20)
plt.tight_layout()
plt.savefig(
    os.path.join(figure_dir, "core4_class_distribution.png"),
    dpi=300,
    bbox_inches="tight",
)
