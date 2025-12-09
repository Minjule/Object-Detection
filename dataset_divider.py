import os
import shutil
import random
from pathlib import Path

SOURCE_ROOT = Path("D:\\HAR\\dataset")    
TARGET_ROOT = Path("D:\\HAR\\divided_dataset") 
TARGET_ROOT.mkdir(parents=True, exist_ok=True)

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.2
TEST_SPLIT = 0.1

IMAGE_EXTS = [".jpg", ".jpeg", ".png"]

for split in ["train", "val", "test"]:
    (TARGET_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
    (TARGET_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

pairs = []

for class_dir in SOURCE_ROOT.iterdir():
    print(class_dir)
    if class_dir.is_dir():
        for dir in class_dir.glob("*"):
            print(dir)
            if not dir.is_dir():
                continue
            for img_file in dir.iterdir():
                if img_file.suffix.lower() in IMAGE_EXTS:
                    label_file = SOURCE_ROOT / class_dir.name / "labels" / (img_file.stem + ".txt")
                    label_file = Path(label_file)
                    if label_file.exists():
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
