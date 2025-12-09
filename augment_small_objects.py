#!/usr/bin/env python3
"""
augment_small_objects.py

Create augmentations that improve small-object detection:
- shrink & paste (downscale objects and paste back)
- copy-paste small objects from other images
- random crop + downscale
- mosaic augmentations
- random multi-scale resizing

Input: YOLO-format dataset:
  src/
    images/<split>/*.jpg
    labels/<split>/*.txt

Output: dst/ (same structure) with augmented images and updated YOLO labels.

Author: adapted for HAR small-object augmentation
"""

import argparse
import random
import shutil
from pathlib import Path
from typing import List, Tuple, Dict
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import math
import os


def load_yolo_label(txt_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """
    Returns list of (cls, x_center_norm, y_center_norm, w_norm, h_norm)
    """
    res = []
    if not txt_path.exists():
        return res
    with txt_path.open('r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:5])
            res.append((cls, x, y, w, h))
    return res

def yolo_to_xyxy(yolo_box, img_w, img_h):
    cls, x, y, w, h = yolo_box
    cx = x * img_w
    cy = y * img_h
    bw = w * img_w
    bh = h * img_h
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2
    return cls, int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

def xyxy_to_yolo(box, img_w, img_h):
    x1, y1, x2, y2 = box
    # clip
    x1 = max(0, min(x1, img_w-1))
    x2 = max(0, min(x2, img_w-1))
    y1 = max(0, min(y1, img_h-1))
    y2 = max(0, min(y2, img_h-1))
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return None
    cx = x1 + w / 2
    cy = y1 + h / 2
    return (cx / img_w, cy / img_h, w / img_w, h / img_h)

def write_yolo_label(txt_path: Path, labels: List[Tuple[int, float, float, float, float]]):
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with txt_path.open('w') as f:
        for (cls, x, y, w, h) in labels:
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def read_img_cv2(p: Path):
    img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        img = cv2.imread(str(p))  # fallback
    return img

def write_img_cv2(path: Path, img):
    # use imencode to handle unicode paths & Windows
    ext = path.suffix.lower()
    if ext in ['.jpg', '.jpeg']:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), 92]
    else:
        params = []
    ok, enc = cv2.imencode(ext, img, params)
    if not ok:
        raise RuntimeError("Failed to encode image")
    enc.tofile(str(path))



def shrink_and_paste(img: np.ndarray, labels: List[Tuple[int, int, int, int, int]],
                     downscale_min=0.15, downscale_max=0.5, prob=0.8, paste_perturb_px=30):
    """
    For each object (bbox), with probability `prob`, crop it and paste a downscaled version
    back into the image at a nearby random position.
    - labels: list of (cls, x1, y1, x2, y2) in pixel coords
    Returns (img_out, new_labels_pixel)
    """
    H, W = img.shape[:2]
    out = img.copy()
    new_boxes = []
    for (cls, x1, y1, x2, y2) in labels:
        if random.random() > prob:
            new_boxes.append((cls, x1, y1, x2, y2))
            continue
        # ensure box is inside
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(W-1, x2), min(H-1, y2)
        w = x2c - x1c
        h = y2c - y1c
        if w <= 4 or h <= 4:
            new_boxes.append((cls, x1, y1, x2, y2))
            continue
        crop = out[y1c:y2c, x1c:x2c].copy()
        # choose scale factor to shrink (final object size relative to original bbox)
        scale = random.uniform(downscale_min, downscale_max)
        nw = max(2, int(round(w * scale)))
        nh = max(2, int(round(h * scale)))
        small = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
        # choose paste center randomly near original bbox center
        cx = int(round((x1c + x2c) / 2.0))
        cy = int(round((y1c + y2c) / 2.0))
        px = cx + random.randint(-paste_perturb_px, paste_perturb_px)
        py = cy + random.randint(-paste_perturb_px, paste_perturb_px)
        # compute paste top-left
        px1 = px - nw // 2
        py1 = py - nh // 2
        # clip within image
        px1_clipped = max(0, min(px1, W - nw))
        py1_clipped = max(0, min(py1, H - nh))
        # paste
        out[py1_clipped:py1_clipped+nh, px1_clipped:px1_clipped+nw] = small
        # add new bbox
        new_boxes.append((cls, px1_clipped, py1_clipped, px1_clipped + nw, py1_clipped + nh))
        # keep original as well? Usually we remove or keep both; here we keep both occasionally
        if random.random() < 0.3:
            new_boxes.append((cls, x1c, y1c, x2c, y2c))
    return out, new_boxes

def copy_paste_from_pool(img: np.ndarray, labels: List[Tuple[int,int,int,int,int]],
                         pool_images: List[Tuple[np.ndarray, List[Tuple[int,int,int,int,int]]]],
                         n_copies=5, scale_min=0.03, scale_max=0.35, iou_thresh=0.1):
    """
    Randomly sample objects from pool_images and paste them (small scaled) into img.
    pool_images: list of (img_array, boxes_pixel_list)
    Returns out_img, new_boxes (pixel coords)
    """
    H, W = img.shape[:2]
    out = img.copy()
    new_boxes = list(labels)  # keep originals
    # avoid too many attempts
    attempts = 0
    pasted = 0
    while pasted < n_copies and attempts < n_copies * 10:
        attempts += 1
        # pick random pool image and object
        src_img, src_boxes = random.choice(pool_images)
        if not src_boxes:
            continue
        src_box = random.choice(src_boxes)
        cls, sx1, sy1, sx2, sy2 = src_box
        sw = sx2 - sx1
        sh = sy2 - sy1
        if sw <= 2 or sh <= 2:
            continue
        # choose scale factor for pasted object relative to target image long side
        scale = random.uniform(scale_min, scale_max)
        # desired size in target image
        target_size = int(round(max(W, H) * scale))
        # preserve aspect ratio
        ar = sh / sw
        nw = max(2, int(round(min(target_size, sw) )))  # ensure >0
        nh = max(2, int(round(nw * ar)))
        src_crop = src_img[sy1:sy2, sx1:sx2].copy()
        # resize
        small = cv2.resize(src_crop, (nw, nh), interpolation=cv2.INTER_AREA)
        # random paste location
        px1 = random.randint(0, max(0, W - nw))
        py1 = random.randint(0, max(0, H - nh))
        candidate_box = (px1, py1, px1 + nw, py1 + nh)
        # check small IoU with existing boxes to avoid excessive overlap
        too_overlap = False
        for ob in new_boxes:
            _, ox1, oy1, ox2, oy2 = ob
            iou = box_iou(candidate_box, (ox1, oy1, ox2, oy2))
            if iou > iou_thresh:
                too_overlap = True
                break
        if too_overlap:
            continue
        # paste (no alpha blending)
        out[py1:py1+nh, px1:px1+nw] = small
        new_boxes.append((cls, px1, py1, px1+nw, py1+nh))
        pasted += 1
    return out, new_boxes

def box_iou(a, b):
    # a and b: (x1,y1,x2,y2)
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    iw = max(0, x2 - x1)
    ih = max(0, y2 - y1)
    inter = iw * ih
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union

def random_rescale_image(img: np.ndarray, labels_pixel: List[Tuple[int,int,int,int,int]],
                         short_side_min=160, short_side_max=800):
    """Randomly rescale image to a different size. Wider range (160-800) for more variation."""
    H, W = img.shape[:2]
    short = random.randint(short_side_min, short_side_max)
    if W < H:
        nw = short
        nh = int(round(H * short / W))
    else:
        nh = short
        nw = int(round(W * short / H))
    out = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    # optionally apply brightness/contrast jitter
    if random.random() < 0.5:
        alpha = random.uniform(0.7, 1.3)  # contrast
        beta = random.randint(-30, 30)     # brightness
        out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)
    # scale boxes
    sx = nw / W
    sy = nh / H
    scaled = []
    for (cls, x1, y1, x2, y2) in labels_pixel:
        nx1 = int(round(x1 * sx))
        ny1 = int(round(y1 * sy))
        nx2 = int(round(x2 * sx))
        ny2 = int(round(y2 * sy))
        scaled.append((cls, nx1, ny1, nx2, ny2))
    return out, scaled

def random_crop_with_downscale(img: np.ndarray, labels_pixel: List[Tuple[int,int,int,int,int]],
                               crop_frac_min=0.3, crop_frac_max=0.8):
    """
    Randomly crop a region of the image and then resize crop back to original size (simulates zoom-out).
    Wider crop range (0.3-0.8) for stronger downscale effect.
    """
    H, W = img.shape[:2]
    frac = random.uniform(crop_frac_min, crop_frac_max)
    cw = int(round(W * frac))
    ch = int(round(H * frac))
    if cw < 2 or ch < 2:
        return img, labels_pixel
    x1 = random.randint(0, W - cw)
    y1 = random.randint(0, H - ch)
    crop = img[y1:y1+ch, x1:x1+cw].copy()
    resized = cv2.resize(crop, (W, H), interpolation=cv2.INTER_LINEAR)
    # optionally apply slight blur or noise to increase variation
    if random.random() < 0.4:
        resized = cv2.GaussianBlur(resized, (3, 3), 0)
    if random.random() < 0.3:
        noise = np.random.normal(0, 5, resized.shape).astype(np.uint8)
        resized = cv2.add(resized, noise)
    # adjust boxes: map original box to crop, then to resized coords
    new_boxes = []
    sx = W / cw
    sy = H / ch
    for (cls, bx1, by1, bx2, by2) in labels_pixel:
        # shift coords relative to crop
        rx1 = bx1 - x1
        ry1 = by1 - y1
        rx2 = bx2 - x1
        ry2 = by2 - y1
        # if box outside crop, skip or clip
        if rx2 <= 0 or ry2 <= 0 or rx1 >= cw or ry1 >= ch:
            continue
        # clip to crop
        rx1 = max(0, rx1)
        ry1 = max(0, ry1)
        rx2 = min(cw-1, rx2)
        ry2 = min(ch-1, ry2)
        nx1 = int(round(rx1 * sx))
        ny1 = int(round(ry1 * sy))
        nx2 = int(round(rx2 * sx))
        ny2 = int(round(ry2 * sy))
        # discard tiny boxes
        if nx2 - nx1 < 2 or ny2 - ny1 < 2:
            continue
        new_boxes.append((cls, nx1, ny1, nx2, ny2))
    return resized, new_boxes

def make_mosaic(imgs_with_boxes: List[Tuple[np.ndarray, List[Tuple[int,int,int,int,int]]]],
                out_size=640):
    """
    Create 4-image mosaic. Each input should be (img, boxes_pixel).
    Returns mosaic_img, mosaic_boxes_pixel
    """
    s = out_size
    mosaic = np.full((s, s, 3), 114, dtype=np.uint8)
    # mosaic center
    xc = int(random.uniform(s * 0.25, s * 0.75))
    yc = int(random.uniform(s * 0.25, s * 0.75))

    positions = [
        (0, 0, xc, yc),       # top-left
        (xc, 0, s, yc),       # top-right
        (0, yc, xc, s),       # bottom-left
        (xc, yc, s, s)        # bottom-right
    ]
    mosaic_boxes = []
    for (i, (img, boxes)) in enumerate(imgs_with_boxes):
        x1a, y1a, x2a, y2a = positions[i]
        pw = x2a - x1a
        ph = y2a - y1a
        h0, w0 = img.shape[:2]
        # scale img to fit
        scale = min(pw / w0, ph / h0)
        if scale == 0:
            continue
        nw, nh = int(w0 * scale), int(h0 * scale)
        img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        # place top-left of resized image
        x_offset = x1a
        y_offset = y1a
        mosaic[y_offset:y_offset+nh, x_offset:x_offset+nw] = img_resized
        # map boxes
        sx = scale
        sy = scale
        for (cls, bx1, by1, bx2, by2) in boxes:
            nbx1 = int(round(bx1 * sx)) + x_offset
            nby1 = int(round(by1 * sy)) + y_offset
            nbx2 = int(round(bx2 * sx)) + x_offset
            nby2 = int(round(by2 * sy)) + y_offset
            # clip
            nbx1 = max(0, min(nbx1, s-1))
            nby1 = max(0, min(nby1, s-1))
            nbx2 = max(0, min(nbx2, s-1))
            nby2 = max(0, min(nby2, s-1))
            if nbx2 - nbx1 > 1 and nby2 - nby1 > 1:
                mosaic_boxes.append((cls, nbx1, nby1, nbx2, nby2))
    return mosaic, mosaic_boxes

# -----------------------
# Main pipeline
# -----------------------

def process_image_single(img_path: Path, label_path: Path, args, pool_images_pixel):
    """
    Read one image + label, produce augmented images (including original copy),
    and return a list of (img_array, labels_pixel) for saving.
    """
    src_img = read_img_cv2(img_path)
    if src_img is None:
        raise RuntimeError(f"Cannot read image {img_path}")
    h, w = src_img.shape[:2]
    yolo_labels = load_yolo_label(label_path)
    # convert to pixel boxes
    boxes_pixel = []
    for yb in yolo_labels:
        cls, x, y, bw, bh = yb
        _, x1, y1, x2, y2 = yolo_to_xyxy(yb, w, h)
        # ensure valid boxes
        if x2 <= x1 or y2 <= y1:
            continue
        boxes_pixel.append((cls, x1, y1, x2, y2))

    outputs = []
    # always include original (copied)
    outputs.append((src_img.copy(), boxes_pixel.copy(), img_path.name))

    # perform N augmentations
    for aug_i in range(args.n_aug):
        aug_type = random.choices(args.aug_types, k=1)[0]
        if aug_type == "shrink_paste":
            out_img, new_boxes = shrink_and_paste(src_img, boxes_pixel,
                                                  downscale_min=args.shrink_min,
                                                  downscale_max=args.shrink_max,
                                                  prob=args.shrink_prob,
                                                  paste_perturb_px=args.paste_perturb)
        elif aug_type == "copy_paste":
            out_img, new_boxes = copy_paste_from_pool(src_img, boxes_pixel, pool_images_pixel,
                                                      n_copies=args.copy_paste_n,
                                                      scale_min=args.copy_scale_min,
                                                      scale_max=args.copy_scale_max)
        elif aug_type == "random_rescale":
            out_img, new_boxes = random_rescale_image(src_img, boxes_pixel,
                                                      short_side_min=args.rescale_min,
                                                      short_side_max=args.rescale_max)
        elif aug_type == "crop_downscale":
            out_img, new_boxes = random_crop_with_downscale(src_img, boxes_pixel,
                                                            crop_frac_min=args.crop_min,
                                                            crop_frac_max=args.crop_max)
        elif aug_type == "mosaic":
            # sample 3 random other images from pool
            picks = [ (src_img, boxes_pixel) ]
            pool = pool_images_pixel
            if len(pool) >= 3:
                others = random.sample(pool, 3)
            else:
                others = random.choices(pool, k=3)
            picks.extend(others)
            out_img, new_boxes = make_mosaic(picks, out_size=args.mosaic_size)
        else:
            out_img, new_boxes = src_img.copy(), boxes_pixel.copy()
        # optional: clip boxes and filter tiny boxes (user tune)
        filtered = []
        H, W = out_img.shape[:2]
        for (cls, bx1, by1, bx2, by2) in new_boxes:
            # clip
            bx1 = max(0, min(bx1, W-1))
            bx2 = max(0, min(bx2, W-1))
            by1 = max(0, min(by1, H-1))
            by2 = max(0, min(by2, H-1))
            if bx2 - bx1 < args.min_box_wh or by2 - by1 < args.min_box_wh:
                continue
            filtered.append((cls, bx1, by1, bx2, by2))
        outputs.append((out_img, filtered, f"{img_path.stem}_aug{aug_i}{img_path.suffix}"))

    return outputs

def build_pool(src_img_dir: Path, src_lbl_dir: Path, max_pool=500):
    """
    Build a pool of (img_array, boxes_pixel) sampled from dataset to use for copy-paste.
    """
    pool = []
    img_paths = list(src_img_dir.glob("*"))
    random.shuffle(img_paths)
    for p in img_paths[:max_pool]:
        lbl = src_lbl_dir / (p.stem + ".txt")
        img = read_img_cv2(p)
        if img is None:
            continue
        yolo = load_yolo_label(lbl)
        boxes_pixel = []
        H, W = img.shape[:2]
        for yb in yolo:
            cls, x, y, bw, bh = yb
            _, x1, y1, x2, y2 = yolo_to_xyxy(yb, W, H)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes_pixel.append((cls, x1, y1, x2, y2))
        if boxes_pixel:
            pool.append((img, boxes_pixel))
    return pool

# -----------------------
# Save outputs
# -----------------------

def save_outputs(outputs: List[Tuple[np.ndarray, List[Tuple[int,int,int,int,int]], str]],
                 dst_img_dir: Path, dst_lbl_dir: Path, orig_fname: str, copy_original=True):
    saved = []
    for (img, boxes_pixel, out_name) in outputs:
        out_path = dst_img_dir / out_name
        # handle duplicates by adding suffix if exists
        if out_path.exists():
            base = out_path.stem
            out_path = dst_img_dir / (base + f"_{random.randint(0,9999)}" + out_path.suffix)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_img_cv2(out_path, img)
        # convert boxes to yolo format
        H, W = img.shape[:2]
        yolo_labels = []
        for (cls, bx1, by1, bx2, by2) in boxes_pixel:
            yolo_box = xyxy_to_yolo((bx1, by1, bx2, by2), W, H)
            if yolo_box is None:
                continue
            x, y, w, h = yolo_box
            yolo_labels.append((cls, x, y, w, h))
        out_label_path = dst_lbl_dir / (out_path.stem + ".txt")
        write_yolo_label(out_label_path, yolo_labels)
        saved.append((out_path, out_label_path))
    return saved

# -----------------------
# CLI and orchestration
# -----------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, required=True, help='Source dataset root (with images/labels/<split>)')
    parser.add_argument('--dst', type=str, required=True, help='Destination augmented dataset root')
    parser.add_argument('--split', type=str, default='train', help='Which split to augment: train/val/test')
    parser.add_argument('--n_aug', type=int, default=2, help='Number of augmented variants per image')
    parser.add_argument('--aug_types', nargs='+', default=['shrink_paste','copy_paste','crop_downscale','random_rescale','mosaic'],
                        help='List of augmentation types to sample from')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--shrink_min', type=float, default=0.12)
    parser.add_argument('--shrink_max', type=float, default=0.45)
    parser.add_argument('--shrink_prob', type=float, default=0.9)
    parser.add_argument('--paste_perturb', type=int, default=30)
    parser.add_argument('--copy_paste_n', type=int, default=5)
    parser.add_argument('--copy_scale_min', type=float, default=0.03)
    parser.add_argument('--copy_scale_max', type=float, default=0.35)
    parser.add_argument('--rescale_min', type=int, default=160)
    parser.add_argument('--rescale_max', type=int, default=800)
    parser.add_argument('--crop_min', type=float, default=0.3)
    parser.add_argument('--crop_max', type=float, default=0.8)
    parser.add_argument('--mosaic_size', type=int, default=640)
    parser.add_argument('--min_box_wh', type=int, default=12, help='Minimum box width/height in pixels to keep')
    parser.add_argument('--pool_max', type=int, default=500, help='Max samples to build pool for copy-paste')
    parser.add_argument('--copy_original', action='store_true', help='Also copy original images to dst (default yes)')
    return parser.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    src = Path(args.src)
    dst = Path(args.dst)
    split = args.split

    src_img_dir = src / "images" / split
    src_lbl_dir = src / "labels" / split
    if not src_img_dir.exists() or not src_lbl_dir.exists():
        raise RuntimeError(f"Source images/labels for split '{split}' not found at {src_img_dir} / {src_lbl_dir}")

    dst_img_dir = dst / "images" / split
    dst_lbl_dir = dst / "labels" / split
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    # build pool for copy-paste
    print("Building copy-paste pool (this may take a while)...")
    pool = build_pool(src_img_dir, src_lbl_dir, max_pool=args.pool_max)
    print(f"Pool size: {len(pool)} images with objects")

    img_paths = sorted(list(src_img_dir.glob("*")))
    if not img_paths:
        raise RuntimeError("No images found in source split folder")

    for p in tqdm(img_paths, desc="Augmenting images"):
        lbl = src_lbl_dir / (p.stem + ".txt")
        try:
            outs = process_image_single(p, lbl, args, pool)
            saved = save_outputs(outs, dst_img_dir, dst_lbl_dir, p.name, copy_original=True)
        except Exception as e:
            print(f"Error processing {p}: {e}")

    print("Done. Augmented dataset written to:", dst)

if __name__ == "__main__":
    main()
