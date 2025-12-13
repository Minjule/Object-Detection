#!/usr/bin/env python3
"""
video_infer.py

Run object-detection inference on a video (or webcam) and save an annotated output.

Supports model formats: .engine, .onnx, .pt (Ultralytics runtime)
Requires: pip install ultralytics opencv-python-headless   (or opencv-python if you need GUI)

Usage:
    python3 video_infer.py --model /path/to/model.engine --source /path/to/input.mp4 --out /path/to/out.mp4 --imgsz 640

Examples:
    # run on file
    python3 video_infer.py --model ./best.engine --source ./input.mp4 --out ./out.mp4 --imgsz 320

    # run on webcam and show window
    python3 video_infer.py --model ./best.onnx --source 0 --out out.mp4 --display

Notes:
 - If you pass a camera index (0,1,...) for --source the script uses webcam.
 - For .engine files on Jetson, ensure engine was built on-device and is compatible with your JetPack.
"""

import argparse
import os
import time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

def draw_boxes(frame, detections, names):
    # detections: list of dicts {'xyxy':(x1,y1,x2,y2), 'conf':float, 'cls':int}
    for d in detections:
        x1, y1, x2, y2 = map(int, d['xyxy'])
        label = f"{names.get(d['cls'], str(d['cls']))} {d['conf']:.2f}"
        # bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (14, 204, 46), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - 18), (x1 + tw, y1), (14, 204, 46), -1)
        cv2.putText(frame, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    return frame

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to model (.engine, .onnx, .pt)")
    p.add_argument("--source", required=True, help="Video file path or camera index (0,1,...)")
    p.add_argument("--out", default="output.mp4", help="Output video path")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size (resize long side)")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--display", action="store_true", help="Show annotated frames in window")
    p.add_argument("--fps", type=float, default=None, help="Force output FPS (default = input FPS)")
    return p.parse_args()

def main():
    args = parse_args()
    model_path = args.model
    src = args.source
    out_path = Path(args.out)

    # Normalize source: camera index if numeric
    try:
        src_in = int(src)
    except Exception:
        src_in = src

    # Load model (Ultralytics will handle .engine/.onnx/.pt)
    print(f"[INFO] Loading model: {model_path}")
    model = YOLO(model_path)

    # Open video capture — try multiple candidate paths for string sources
    cap = None
    attempted = []

    if isinstance(src_in, int):
        cap = cv2.VideoCapture(src_in)
        attempted.append(str(src_in))
    else:
        s = str(src_in)
        # generate candidate paths in order
        candidates = []
        # as provided (may be absolute or relative)
        candidates.append(Path(os.path.expanduser(s)))
        # relative to current working dir
        candidates.append(Path.cwd() / s)
        # relative to script directory
        try:
            script_dir = Path(__file__).resolve().parent
            candidates.append(script_dir / s)
        except Exception:
            pass
        # user desktop, downloads, pictures (common places)
        home = Path.home()
        candidates.append(home / 'Desktop' / s)
        candidates.append(home / 'Downloads' / s)
        candidates.append(home / 'Pictures' / s)

        # also try as-is string (cv2 may accept it)
        candidates.append(Path(s))

        # attempt each candidate until VideoCapture opens
        for c in candidates:
            c_str = str(c)
            if c_str in attempted:
                continue
            attempted.append(c_str)
            if not c.exists():
                # still try - sometimes cv2 accepts non-existent-looking paths
                cap = cv2.VideoCapture(c_str)
            else:
                cap = cv2.VideoCapture(str(c))
            if cap is not None and cap.isOpened():
                source_to_open = c_str
                break

    if cap is None or not cap.isOpened():
        msg = f"Cannot open source: {src}. Tried: {attempted}"
        raise RuntimeError(msg)

    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    in_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_fps = args.fps or in_fps

    # Prepare output writer (H264/MPEG-4)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4v widely compatible
    out_writer = cv2.VideoWriter(str(out_path), fourcc, out_fps, (in_w, in_h))

    names = {}
    # try to read names if model exposes them
    try:
        # model.model.names may exist
        if hasattr(model, "model") and hasattr(model.model, "names"):
            names = model.model.names
            # ensure dict mapping int->str
            if isinstance(names, list):
                names = {i: n for i, n in enumerate(names)}
    except Exception:
        names = {}

    print(f"[INFO] Input: {in_w}x{in_h} @ {in_fps:.2f} fps. Output: {in_w}x{in_h} @ {out_fps:.2f} fps.")
    print(f"[INFO] Writing output to: {out_path}")
    print("Press 'q' in the display window to stop early.")

    frame_count = 0
    t0 = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # Inference: Ultralytics accepts BGR numpy arrays directly
            # set imgsz & thresholds on call
            results = model(frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou)

            # Parse results: take first result
            dets = []
            if len(results) > 0:
                r = results[0]
                # r.boxes might be empty
                boxes = getattr(r, "boxes", None)
                if boxes is not None and len(boxes) > 0:
                    # boxes.xyxy, boxes.conf, boxes.cls
                    xyxy = boxes.xyxy.cpu().numpy()  # shape (N,4)
                    confs = boxes.conf.cpu().numpy()
                    cls_idxs = boxes.cls.cpu().numpy().astype(int)
                    for bb, cf, cid in zip(xyxy, confs, cls_idxs):
                        dets.append({"xyxy": bb.tolist(), "conf": float(cf), "cls": int(cid)})

            # draw boxes
            annotated = draw_boxes(frame.copy(), dets, names)

            # compute FPS display
            elapsed = time.time() - t0
            fps = frame_count / elapsed if elapsed > 0 else 0.0
            cv2.putText(annotated, f"FPS {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

            # write frame
            out_writer.write(annotated)

            # show if requested
            if args.display:
                cv2.imshow("Inference", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        cap.release()
        out_writer.release()
        if args.display:
            cv2.destroyAllWindows()
        total_time = time.time() - t0
        avg_fps = frame_count / total_time if total_time > 0 else 0.0
        print(f"[INFO] Done. Processed {frame_count} frames in {total_time:.2f}s (avg {avg_fps:.2f} FPS).")
        print(f"[INFO] Output saved to: {out_path}")

if __name__ == "__main__":
    main()
