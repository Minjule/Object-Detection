import os
import shutil
import random
from pathlib import Path

SOURCE_ROOT = Path("D:\\HAR\\mine2")    
TARGET_ROOT = Path("D:\\HAR\\mine2_divided_dataset") 
TARGET_ROOT.mkdir(parents=True, exist_ok=True)

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2
TEST_SPLIT = 0.0

IMAGE_EXTS = [".jpg", ".jpeg", ".png"]

for split in ["train", "val", "test"]:
    (TARGET_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
    (TARGET_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

pairs = []

# Support multiple dataset layouts:
# - SOURCE_ROOT/images/<class>/*.jpg  with labels in SOURCE_ROOT/labels/<class>/*.txt
# - SOURCE_ROOT/images/*.jpg with labels in SOURCE_ROOT/labels/*.txt
# - older layout: SOURCE_ROOT/<class>/<subdir>/*.jpg with labels in SOURCE_ROOT/<class>/labels/*.txt

images_root = SOURCE_ROOT / "images" if (SOURCE_ROOT / "images").exists() else SOURCE_ROOT

def find_label_for_image(img_path):
    """Try several candidate locations for the label file corresponding to img_path."""
    stem = img_path.stem
    # candidate 1: SOURCE_ROOT/labels/<class>/<stem>.txt (when images are in images/<class>)
    parent = img_path.parent
    candidates = []
    # if parent is a class folder (images/<class>/...)
    if parent != images_root:
        class_name = parent.name
        candidates.append(SOURCE_ROOT / "labels" / class_name / (stem + ".txt"))
    # flat labels folder
    candidates.append(SOURCE_ROOT / "labels" / (stem + ".txt"))
    # legacy: SOURCE_ROOT/<class>/labels/<stem>.txt
    if parent != images_root:
        class_name = parent.name
        candidates.append(SOURCE_ROOT / class_name / "labels" / (stem + ".txt"))
    # fallback: sibling labels folder inside parent (e.g. images in nested folder with labels nearby)
    candidates.append(parent / "labels" / (stem + ".txt"))

    for c in candidates:
        if c.exists():
            return c
    return None


if any(p.is_dir() for p in images_root.iterdir()):
    # images are organized by class subfolders
    for class_dir in images_root.iterdir():
        if not class_dir.is_dir():
            continue
        for img_file in class_dir.iterdir():
            if img_file.suffix.lower() in IMAGE_EXTS:
                label_file = find_label_for_image(img_file)
                if label_file:
                    pairs.append((img_file, label_file))
                else:
                    print(f"[WARNING] Missing label for: {img_file}")
else:
    # images are directly under images_root
    for img_file in images_root.iterdir():
        if img_file.suffix.lower() in IMAGE_EXTS:
            label_file = find_label_for_image(img_file)
            if label_file:
                pairs.append((img_file, label_file))
            else:
                print(f"[WARNING] Missing label for: {img_file}")

print(f"Found {len(pairs)} image-label pairs.")

random.shuffle(pairs)

n = len(pairs)
train_end = int(n * TRAIN_SPLIT)
val_end = int(n * (TRAIN_SPLIT + VAL_SPLIT))

train_pairs = pairs[:train_end]
val_pairs = pairs[train_end:val_end]
test_pairs = pairs[val_end:]

print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}")

def copy_pairs(pairs, split):
    for img, lbl in pairs:
        img_target = TARGET_ROOT / "images" / split / img.name
        lbl_target = TARGET_ROOT / "labels" / split / lbl.name

        shutil.copy2(img, img_target)
        shutil.copy2(lbl, lbl_target)


copy_pairs(train_pairs, "train")
copy_pairs(val_pairs, "val")
copy_pairs(test_pairs, "test")

print("Dataset restructuring completed successfully.")
print(f"New structure located at: {TARGET_ROOT}")
