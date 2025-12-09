# original dataset ee uurt heregteigeer uurchluh 
import os
from collections import Counter
import os
import re
import argparse
import shutil
import random
import math
import uuid
import numpy as np
import cv2

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

def get_folder_size(folder_path):
    """Return total size of all files in a folder (in bytes)."""
    total_size = 0
    for dirpath, _, filenames in os.walk(folder_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                if os.path.isfile(fp):  # Skip broken symlinks
                    total_size += os.path.getsize(fp)
            except Exception as e:
                print(f"Could not access {fp}: {e}")
    return total_size

def list_folder_sizes(parent_dir):
    """Iterate through subfolders and print their sizes."""
    for item in os.listdir(parent_dir):
        item_path = os.path.join(parent_dir, item)
        if os.path.isdir(item_path):
            size_bytes = get_folder_size(item_path)
            size_mb = size_bytes / (1024 * 1024)
            print(f"{item}: {size_mb:.2f} MB")

def extract_leading_token(text):
    """
    Return leading number token if present (e.g. "10"), otherwise return the first non-whitespace character.
    Returns None if text is empty/whitespace.
    """
    if text is None:
        return None
    s = text.lstrip()
    if not s:
        return None
    m = re.match(r"(\d+)", s)
    if m:
        return m.group(1)
    return s[0]

def remove_label_from_dataset(root_dir, class_idxs_to_remove=None):
    """
    Remove all samples whose label file's leading token (number or first non-space char)
    is in class_idxs_to_remove. If a label file is empty (whitespace only) it will be removed.
    Args:
        root_dir: dataset root containing class folders
        class_idxs_to_remove: iterable of numbers/strings (e.g. [0,3,10]) or a comma-separated string "0,3,10"
    Returns:
        dict with counts: {'removed_txt': int, 'removed_imgs': int, 'skipped': int}
    """
    if class_idxs_to_remove is None:
        print("[WARN] no class_idxs_to_remove provided, nothing to do.")
        return {"removed_txt": 0, "removed_imgs": 0, "skipped": 0}

    # normalize to set of strings (numbers as strings for comparison)
    if isinstance(class_idxs_to_remove, str):
        class_idxs = {c.strip() for c in class_idxs_to_remove.split(",") if c.strip() != ""}
    else:
        class_idxs = {str(c) for c in class_idxs_to_remove}

    removed_txt = 0
    removed_imgs = 0
    skipped = 0

    for cls in sorted(os.listdir(root_dir)):
        cls_path = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        for folder_name in sorted(os.listdir(cls_path)):
            label_path = os.path.join(cls_path, folder_name)
            if not os.path.isdir(label_path):
                continue
            for name in sorted(os.listdir(label_path)):
                if not name.lower().endswith(".txt"):
                    continue
                txt_path = os.path.join(label_path, name)
                try:
                    with open(txt_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    token = extract_leading_token(content)
                    if token is None:
                        # empty/whitespace-only file -> remove it and its images
                        try:
                            os.remove(txt_path)
                            removed_txt += 1
                        except Exception as e:
                            print(f"[ERR] Failed to remove empty txt {txt_path}: {e}")
                        base = os.path.splitext(name)[0]
                        candidate_dirs = [
                            label_path,
                            os.path.join(label_path, "images"),
                            os.path.join(cls_path, "images"),
                            cls_path
                        ]
                        for d in candidate_dirs:
                            if not os.path.isdir(d):
                                continue
                            for ext in IMAGE_EXTS:
                                img_path = os.path.join(d, base + ext)
                                if os.path.exists(img_path):
                                    try:
                                        os.remove(img_path)
                                        removed_imgs += 1
                                    except Exception as e:
                                        print(f"[ERR] Failed to remove image {img_path}: {e}")
                        continue

                    if token in class_idxs:
                        # remove .txt
                        try:
                            os.remove(txt_path)
                            removed_txt += 1
                        except Exception as e:
                            print(f"[ERR] Failed to remove {txt_path}: {e}")
                        # remove matching image(s) with same base name
                        base = os.path.splitext(name)[0]
                        candidate_dirs = [
                            label_path,
                            os.path.join(label_path, "images"),
                            os.path.join(cls_path, "images"),
                            cls_path
                        ]
                        for d in candidate_dirs:
                            if not os.path.isdir(d):
                                continue
                            for ext in IMAGE_EXTS:
                                img_path = os.path.join(d, base + ext)
                                if os.path.exists(img_path):
                                    try:
                                        os.remove(img_path)
                                        removed_imgs += 1
                                    except Exception as e:
                                        print(f"[ERR] Failed to remove image {img_path}: {e}")
                    else:
                        skipped += 1
                except Exception as e:
                    print(f"[ERR] Processing {txt_path}: {e}")

    summary = {"removed_txt": removed_txt, "removed_imgs": removed_imgs, "skipped": skipped}
    print(f"[INFO] done. removed {removed_txt} .txt files, removed {removed_imgs} image files, skipped {skipped} files.")
    return summary

def update_mapping(root_dir, mapping=None):
    """Update leading character in label .txt files according to mapping dict."""
    updated_txt = 0
    for cls in sorted(os.listdir(root_dir)):
        cls_path = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        for folder_name in sorted(os.listdir(cls_path)):
            label_path = os.path.join(cls_path, folder_name)
            if not os.path.isdir(label_path):
                continue
            for name in sorted(os.listdir(label_path)):
                if not name.lower().endswith(".txt"):
                    continue
                txt_path = os.path.join(label_path, name)
                try:
                    with open(txt_path, "r+", encoding="utf-8") as f:
                        lines = f.readlines()
                        new_lines = []
                        for line in lines:
                            if line is None:
                                break

                            first_ch = line[0]
                            if first_ch in mapping:
                                new_ch = mapping[first_ch]
                                new_line = new_ch + line[1:]
                                updated_txt += 1
                            else:
                                new_line = line
                            new_lines.append(new_line)

                        # Rewind and overwrite file
                        f.seek(0)
                        f.truncate()
                        f.writelines(new_lines)
                except Exception as e:
                    print(f"[ERR] updating the label {txt_path}: {e}")
    print(f"[INFO] done. Updated {updated_txt} .txt rows.")

def number_of_instances_train_test_val(root_dir):
    """Count and print number of label files and image files per class and totals."""
    label_counts = Counter()
    img_counts = Counter()
    total_labels = 0
    total_images = 0

    for cls in sorted(os.listdir(root_dir)):
        cls_path = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        lbl_cnt = 0
        img_cnt = 0
        for folder_name in sorted(os.listdir(cls_path)):
            label_path = os.path.join(cls_path, folder_name)
            if not os.path.isdir(label_path):
                continue
            for name in sorted(os.listdir(label_path)):
                lname = name.lower()
                if lname.endswith(".txt"):
                    lbl_cnt += 1
                elif any(lname.endswith(ext) for ext in IMAGE_EXTS):
                    img_cnt += 1
        label_counts[cls] = lbl_cnt
        img_counts[cls] = img_cnt
        total_labels += lbl_cnt
        total_images += img_cnt

    print("Number of instances per class:")
    for cls in sorted(label_counts.keys()):
        print(f" Class '{cls}': {label_counts[cls]} label files, {img_counts[cls]} images")
    print(f"Totals: {total_labels} label files, {total_images} images")

def count_first_chars_per_class(root_dir):
    """
    For each class folder under root_dir:
      - walks into each subfolder and reads .txt files
      - extracts the first non-whitespace character from each txt
      - counts occurrences and distinct characters per class
    Prints per-class counters and global totals.
    """
    global_counter = Counter()
    classes_stats = {}

    for cls in sorted(os.listdir(root_dir)):
        cls_path = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        cls_counter = Counter()
        # iterate subfolders (existing dataset layout)
        for folder_name in sorted(os.listdir(cls_path)):
            label_path = os.path.join(cls_path, folder_name)
            if not os.path.isdir(label_path):
                continue
            for fname in sorted(os.listdir(label_path)):
                if not fname.lower().endswith(".txt"):
                    continue
                txt_path = os.path.join(label_path, fname)
                try:
                    with open(txt_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    # find first non-whitespace char
                    stripped = content.lstrip()
                    if not stripped:
                        continue
                    first_ch = stripped[0]
                    cls_counter[first_ch] += 1
                    global_counter[first_ch] += 1
                except Exception as e:
                    print(f"[ERR] reading {txt_path}: {e}")

        classes_stats[cls] = cls_counter
    # print detailed per-class stats
    print("First-char label counts per class:")
    for cls, counter in classes_stats.items():
        distinct = len(counter)
        total = sum(counter.values())
        print(f" Class '{cls}': {total} labels, {distinct} distinct first-chars -> {dict(counter)}")
    print(f"Global totals: {sum(global_counter.values())} labels, {len(global_counter)} distinct first-chars -> {dict(global_counter)}")
    return classes_stats, global_counter

def combine_datasets(source_dirs, output_dir, filter_label=None):
    """
    Combine multiple dataset folders into one output folder.
    For each .txt label file, scan line-by-line:
      - If any line's leading token matches filter_label, copy the entire .txt and all matching images
      - If filter_label is None, copy everything
    
    Args:
        source_dirs: list of source root directories
        output_dir: destination root directory
        filter_label: if not None, only copy samples with lines starting with these labels (list/set of strings)
    """
    if not isinstance(source_dirs, list):
        source_dirs = [source_dirs]
    
    if filter_label is not None:
        if isinstance(filter_label, str):
            filter_label = {filter_label}
        else:
            filter_label = set(str(x) for x in filter_label)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    out_label_path = os.path.join(output_dir, "labels")
    out_images_path = os.path.join(output_dir, "images")
    if not os.path.exists(out_label_path):
        os.makedirs(out_label_path)
    if not os.path.exists(out_images_path):
        os.makedirs(out_images_path)
    
    total_copied_txt = 0
    total_copied_img = 0
    total_skipped = 0
    print(source_dirs)
    for cls_path in source_dirs:
        print(cls_path)
        if not os.path.isdir(cls_path):
            continue
            
        label_path = os.path.join(cls_path, "labels")
            
        for name in sorted(os.listdir(label_path)):
            txt_path = os.path.join(label_path, name)
            should_copy_txt = True
                    
            # check if any line in .txt matches filter_label
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    for line in lines:
                        token = extract_leading_token(line)
                        if token not in filter_label:
                            should_copy_txt = False
                            break
            except Exception as e:
                print(f"[WARN] error reading {txt_path}: {e}")
                continue
                    
            if not should_copy_txt:
                total_skipped += 1
                continue
                    
            # copy .txt file
            out_txt_path = os.path.join(out_label_path, name)
            try:
                shutil.copy2(txt_path, out_txt_path)
                total_copied_txt += 1
            except Exception as e:
                print(f"[ERR] copying txt {txt_path} to {out_txt_path}: {e}")
                    
            # copy matching image files (same base name)
            base = os.path.splitext(name)[0]
            print(base)
            image_path = os.path.join(cls_path, "images")
            print(image_path)
            
                    
            img_name = base + ".jpg"
            src_img = os.path.join(image_path, img_name)
            dst_img = os.path.join(out_images_path, img_name)
            shutil.copy2(src_img, dst_img)
            total_copied_img += 1
    
    print(f"[INFO] Combined datasets. Copied {total_copied_txt} .txt files, {total_copied_img} images, skipped {total_skipped} samples.")
    return {"copied_txt": total_copied_txt, "copied_img": total_copied_img, "skipped": total_skipped}

def find_label_for_image(base_name, cls_path):
    """Try common locations for a label file matching base_name under cls_path."""
    candidates = [
        os.path.join(cls_path, "labels", base_name + ".txt"),
        os.path.join(cls_path, "labels", base_name + ".TXT"),
    ]
    # also search shallowly under labels folder
    lbl_dir = os.path.join(cls_path, "labels")
    if os.path.isdir(lbl_dir):
        for fname in os.listdir(lbl_dir):
            if os.path.splitext(fname)[0] == base_name:
                return os.path.join(lbl_dir, fname)
    # fallback: recursive search for matching basename
    for dirpath, _, files in os.walk(cls_path):
        for f in files:
            if os.path.splitext(f)[0] == base_name and f.lower().endswith(".txt"):
                return os.path.join(dirpath, f)
    return None

def list_images_in_class(cls_path):
    """Return list of image file paths under cls_path/images (or cls_path)."""
    paths = []
    images_dir = os.path.join(cls_path, "images")
    search_dir = images_dir if os.path.isdir(images_dir) else cls_path
    for fname in sorted(os.listdir(search_dir)):
        if any(fname.lower().endswith(ext) for ext in IMAGE_EXTS):
            paths.append(os.path.join(search_dir, fname))
    return paths, search_dir

def random_affine(img, max_rotate=30, max_translate=0.05, max_scale_delta=0.12):
    h, w = img.shape[:2]
    angle = random.uniform(-max_rotate, max_rotate)
    scale = 1.0 + random.uniform(-max_scale_delta, max_scale_delta)
    tx = random.uniform(-max_translate, max_translate) * w
    ty = random.uniform(-max_translate, max_translate) * h
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
    M[0,2] += tx
    M[1,2] += ty
    out = cv2.warpAffine(img, M, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    return out

def random_perspective(img, max_warp=0.03):
    h, w = img.shape[:2]
    margin_x = w * max_warp
    margin_y = h * max_warp
    pts1 = np.float32([[0,0],[w,0],[w,h],[0,h]])
    pts2 = np.float32([
        [random.uniform(-margin_x, margin_x), random.uniform(-margin_y, margin_y)],
        [w + random.uniform(-margin_x, margin_x), random.uniform(-margin_y, margin_y)],
        [w + random.uniform(-margin_x, margin_x), h + random.uniform(-margin_y, margin_y)],
        [random.uniform(-margin_x, margin_x), h + random.uniform(-margin_y, margin_y)]
    ])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    out = cv2.warpPerspective(img, M, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    return out

def jitter_brightness_contrast(img, max_brightness=30, max_contrast=0.25):
    # brightness add [-max_brightness, max_brightness], contrast multiply [1-max_contrast,1+max_contrast]
    b = random.uniform(-max_brightness, max_brightness)
    c = 1.0 + random.uniform(-max_contrast, max_contrast)
    out = img.astype(np.float32) * c + b
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def add_gaussian_noise(img, sigma=8):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def maybe_blur(img, max_ksize=3):
    if random.random() < 0.3:
        k = random.choice([1,3])  # 1 -> no blur, 3 small blur
        if k > 1:
            return cv2.GaussianBlur(img, (k,k), 0)
    return img

def augment_image(img):
    """Apply a random sequence of augmentations (no flip)."""
    # apply either affine or perspective
    if random.random() < 0.9:
        img = random_affine(img, max_rotate=30, max_translate=0.06, max_scale_delta=0.13)
    else:
        img = random_perspective(img, max_warp=0.04)
    img = jitter_brightness_contrast(img, max_brightness=60, max_contrast=0.5)
    img = maybe_blur(img, max_ksize=3)
    if random.random() < 0.6:
        img = add_gaussian_noise(img, sigma=random.uniform(4,12))
    return img

def augment_class_images(root_dir, target_count=510, classes=None, seed=42, dry_run=False):
    """
    For each class folder under root_dir (or limited to 'classes'):
      - find images in <cls>/images (or <cls>)
      - augment images (no flipping) until each class has at least target_count images
      - copy corresponding label .txt for each new image (keeping base name)
    Returns summary dict.
    """
    random.seed(seed)
    np.random.seed(seed)

    if classes is not None:
        classes = set(str(c) for c in classes)

    summary = {}
    for cls in sorted(os.listdir(root_dir)):
        if classes and cls not in classes:
            continue
        cls_path = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        imgs, images_dir = list_images_in_class(cls_path)
        cur_n = len(imgs)
        if cur_n == 0:
            print(f"[WARN] no images found in {cls_path} (search dir {images_dir})")
            continue
        needed = max(0, target_count - cur_n)
        print(f"[INFO] class '{cls}': {cur_n} images, need {needed} more -> target {target_count}")
        created = 0
        i_trial = 0
        while created < needed and i_trial < needed * 6:
            i_trial += 1
            src = imgs[random.randrange(cur_n)]
            try:
                img = cv2.imread(src)
                if img is None:
                    print(f"[WARN] failed to read {src}")
                    continue
                aug = augment_image(img)
                # generate unique name
                base = os.path.splitext(os.path.basename(src))[0]
                new_base = f"{base}_aug_{uuid.uuid4().hex[:8]}"
                new_fname = new_base + ".jpg"
                dst_path = os.path.join(images_dir, new_fname)
                if dry_run:
                    created += 1
                    continue
                # save jpg
                ok = cv2.imwrite(dst_path, aug)
                if not ok:
                    print(f"[ERR] failed to write {dst_path}")
                    continue
                # find & copy label file if exists
                lbl_src = find_label_for_image(base, cls_path)
                if lbl_src:
                    lbl_dst_dir = os.path.join(cls_path, "labels")
                    if not os.path.isdir(lbl_dst_dir):
                        os.makedirs(lbl_dst_dir, exist_ok=True)
                    dst_lbl = os.path.join(lbl_dst_dir, new_base + ".txt")
                    try:
                        shutil.copy2(lbl_src, dst_lbl)
                    except Exception as e:
                        print(f"[ERR] copying label {lbl_src} -> {dst_lbl}: {e}")
                else:
                    print(f"[WARN] no label found for base '{base}' in class '{cls}' (image {src})")
                created += 1
            except Exception as e:
                print(f"[ERR] augmenting {src}: {e}")
        summary[cls] = {"initial": cur_n, "created": created, "final": cur_n + created}
        print(f"[INFO] class '{cls}' done: created {created} new images (final {cur_n+created})")
    return summary

# integrate CLI
if __name__ == "__main__":
    root_dir = "D:\\HAR\\harmful_12"
    mapping = {"0": "2", "8": "0", "9": "1"}
    # remove_label_from_dataset(root_dir, class_idxs_to_remove=["1", "3", "4", "2", "5", "6", "7", "10", "11"])
    # update_mapping(root_dir, mapping)
    # number_of_instances_train_test_val(root_dir)
    # count_first_chars_per_class(root_dir)
    
    parser = argparse.ArgumentParser(description="Dataset utility: combine, filter, count labels")
    parser.add_argument("--action", choices=["combine", "count", "remove", "update", "augment"], 
                        default="count", help="Action to perform")
    
    # combine action args
    parser.add_argument("--sources", nargs="+", help="Source dataset directories (for combine action)")
    parser.add_argument("--output", help="Output dataset directory")
    parser.add_argument("--filter-label", help="Filter by label (single or comma-separated, e.g. '0' or '0,2')")
    
    # remove action args
    parser.add_argument("--remove-labels", help="Labels to remove (comma-separated, e.g. '1,3,4')")
    
    # update action args
    parser.add_argument("--mapping", help="Label mapping as JSON-like string, e.g. '{\"0\":\"2\",\"8\":\"0\"}'")
    
    # augment action args
    parser.add_argument("--target", type=int, default=510, help="target images per class for augment action")
    parser.add_argument("--classes", help="comma-separated class names to augment (default all)")
    parser.add_argument("--seed", type=int, default=42, help="random seed")

    # general args
    parser.add_argument("--root-dir", default=".", help="Root dataset directory")
    
    args = parser.parse_args()

    if args.action == "combine":
        if not args.sources or not args.output:
            print("[ERR] --combine requires --sources and --output")
        else:
            filter_lbl = None
            if args.filter_label:
                filter_lbl = [x.strip() for x in args.filter_label.split(",")]
            combine_datasets(args.sources, args.output, filter_label=filter_lbl)
    
    elif args.action == "count":
        count_first_chars_per_class(args.root_dir)
        number_of_instances_train_test_val(args.root_dir)
    
    elif args.action == "remove":
        if not args.remove_labels:
            print("[ERR] --remove requires --remove-labels")
        else:
            remove_label_from_dataset(args.root_dir, args.remove_labels)
    
    elif args.action == "update":
        if not args.mapping:
            print("[ERR] --update requires --mapping")
        else:
            import json
            try:
                mapping = json.loads(args.mapping)
                update_mapping(args.root_dir, mapping)
            except Exception as e:
                print(f"[ERR] parsing mapping: {e}")
    
    elif args.action == "augment":
        classes = None
        if args.classes:
            classes = [c.strip() for c in args.classes.split(",") if c.strip()]
        res = augment_class_images(args.root_dir, target_count=args.target, classes=classes, seed=args.seed)
        print("Augmentation summary:", res)

