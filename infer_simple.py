#!/usr/bin/env python3
"""
infer_simple.py

Minimal real-time inference test script using Ultralytics YOLO.
- Loads a model (.pt/.onnx/.engine)
- Opens a camera index or video file
- Runs inference, prints detections, and shows annotated frames

Usage examples:
    python infer_simple.py --model runs/exp/weights/best.pt --source 0 --device cpu --display
    python infer_simple.py --model runs/exp/weights/best.pt --source path/to/video.mp4 --device cpu
"""

import argparse
import time
from datetime import datetime

import cv2
from ultralytics import YOLO
import numpy as np

DEFAULT_CONF = 0.50
DEFAULT_IOU = 0.45
DEFAULT_SIZE = 320


def try_open_source(src, max_index=4):
    """Try to open `src`. If src is a numeric index that fails, try 0..max_index-1 and return the first working index.
    Returns an integer index or the original string path if it opens as a file.
    Raises RuntimeError if nothing works.
    """
    if isinstance(src, str):
        try:
            cap = cv2.VideoCapture(src)
            if cap is not None and cap.isOpened():
                cap.release()
                return src
        except Exception:
            pass

    try:
        requested = int(src)
    except Exception:
        requested = None

    if requested is not None:
        indices = [requested] + [i for i in range(max_index) if i != requested]
        for idx in indices:
            try:
                cap = cv2.VideoCapture(idx)
                if cap is not None and cap.isOpened():
                    cap.release()
                    return idx
            except Exception:
                pass

    raise RuntimeError(f"Unable to open source '{src}'. Tested indices 0..{max_index-1} and path; ensure camera/file exists.")


def draw_boxes(frame, detections, color=(16, 200, 100)):
    for d in detections:
        x1, y1, x2, y2 = d['box']
        label = f"{d['name']} {d['conf']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - 18), (x1 + w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model (.pt/.onnx/.engine)")
    parser.add_argument("--source", default=0, help="Camera index or video file path")
    parser.add_argument("--device", default="cpu", help="'cpu' or CUDA index like '0'")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--display", action="store_true", help="Show annotated frames")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = run until q/EOF)")
    args = parser.parse_args()

    try:
        src = try_open_source(args.source)
    except RuntimeError as e:
        print("[ERROR]", e)
        return

    print(f"Loading model: {args.model} on device {args.device}")
    model = YOLO(args.model)

    # open capture
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Failed to open source {src}")
        return

    frame_count = 0
    t0 = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of stream or read failure")
                break

            # run inference
            results = model(frame, imgsz=args.imgsz, conf=args.conf, iou=DEFAULT_IOU, device=args.device)
            r = results[0]
            dets = []
            if hasattr(r, 'boxes') and r.boxes is not None:
                boxes = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes, 'xyxy') else np.array([])
                confs = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, 'conf') else np.array([])
                cls_idxs = r.boxes.cls.cpu().numpy().astype(int) if hasattr(r.boxes, 'cls') else np.array([])
                for (bb, conf, cls_idx) in zip(boxes, confs, cls_idxs):
                    name = model.model.names[int(cls_idx)] if hasattr(model, 'model') and hasattr(model.model, 'names') else str(cls_idx)
                    x1, y1, x2, y2 = map(int, bb.tolist())
                    dets.append({'name': name, 'conf': float(conf), 'box': [x1, y1, x2, y2]})

            # print detections
            if dets:
                ts = datetime.now().isoformat()
                print(f"[{ts}] Detections:")
                for d in dets:
                    print(f"  - {d['name']} conf={d['conf']:.2f} box={d['box']}")

            annotated = draw_boxes(frame.copy(), dets)

            frame_count += 1
            if args.display:
                cv2.imshow('infer_simple', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if args.max_frames and frame_count >= args.max_frames:
                print(f"Reached max frames {args.max_frames}")
                break

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        elapsed = time.time() - t0
        fps = frame_count / elapsed if elapsed > 0 else 0
        print(f"Processed {frame_count} frames in {elapsed:.2f}s ({fps:.2f} FPS)")


if __name__ == '__main__':
    main()
