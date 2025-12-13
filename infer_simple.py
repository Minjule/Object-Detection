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

DEFAULT_CONF = 0.40
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
        if d['name'] == "child":
            label = f"{d.get('name','')[:20]} {d['conf']:.2f}"
        else:
            label = f"{'Danger'[:20]} {d['conf']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - 18), (x1 + w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return frame


def iou(boxA, boxB):
    # boxes are [x1,y1,x2,y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union = boxAArea + boxBArea - interArea
    return interArea / union if union > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model (.pt/.onnx/.engine)")
    parser.add_argument("--model2", dest="model2", default=None,
                        help="Optional second model to load for comparison (path to .pt/.onnx).")
    parser.add_argument("--source", default=0, help="Camera index or video file path")
    parser.add_argument("--device", default="cpu", help="'cpu' or CUDA index like '0'")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--display", action="store_true", help="Show annotated frames")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = run until q/EOF)")
    parser.add_argument("--output", dest="output", default=None,
                        help="Path to save annotated video (e.g. output.mp4). If omitted, no file is written.")
    parser.add_argument("--overlap-threshold", dest="overlap_threshold", type=float, default=0.01,
                        help="IoU threshold above which overlapping detections trigger an alert (0-1).")
    parser.add_argument("--compare-same-class", dest="compare_same_class", action="store_true",
                        help="Only consider overlaps when both detections have the same class index.")
    parser.add_argument("--alert-text", dest="alert_text", default="ALERT: overlapping detections",
                        help="Text to display on the frame when overlapping detections are found.")
    args = parser.parse_args()

    try:
        src = try_open_source(args.source)
    except RuntimeError as e:
        print("[ERROR]", e)
        return

    print(f"Loading model: {args.model} on device {args.device}")
    model = YOLO(args.model)
    model2 = None
    if args.model2:
        print(f"Loading second model: {args.model2} on device {args.device}")
        model2 = YOLO(args.model2)

    # open capture
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Failed to open source {src}")
        return

    # Video writer (created on first frame when we know frame size)
    writer = None

    frame_count = 0
    t0 = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of stream or read failure")
                break

            # run inference model 1
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
                    dets.append({'name': name, 'conf': float(conf), 'box': [x1, y1, x2, y2], 'cls': int(cls_idx)})

            # run inference model 2 (optional)
            dets2 = []
            if model2 is not None:
                results2 = model2(frame, imgsz=args.imgsz, conf=args.conf, iou=DEFAULT_IOU, device=args.device)
                r2 = results2[0]
                if hasattr(r2, 'boxes') and r2.boxes is not None:
                    boxes2 = r2.boxes.xyxy.cpu().numpy() if hasattr(r2.boxes, 'xyxy') else np.array([])
                    confs2 = r2.boxes.conf.cpu().numpy() if hasattr(r2.boxes, 'conf') else np.array([])
                    cls_idxs2 = r2.boxes.cls.cpu().numpy().astype(int) if hasattr(r2.boxes, 'cls') else np.array([])
                    for (bb, conf, cls_idx) in zip(boxes2, confs2, cls_idxs2):
                        name = model2.model.names[int(cls_idx)] if hasattr(model2, 'model') and hasattr(model2.model, 'names') else str(cls_idx)
                        x1, y1, x2, y2 = map(int, bb.tolist())
                        dets2.append({'name': name, 'conf': float(conf), 'box': [x1, y1, x2, y2], 'cls': int(cls_idx)})

            # print detections (both models)
            if dets or dets2:
                ts = datetime.now().isoformat()
                print(f"[{ts}] Detections:")
                for d in dets:
                    print(f"  - M1 {d['name']} conf={d['conf']:.2f} box={d['box']}")
                for d in dets2:
                    print(f"  - M2 {d['name']} conf={d['conf']:.2f} box={d['box']}")

            annotated = frame.copy()

            # draw boxes from both models in different colors
            annotated = draw_boxes(annotated, dets, color=(16, 200, 100))  # green-ish for model1
            if dets2:
                annotated = draw_boxes(annotated, dets2, color=(100, 160, 255))  # blue-ish for model2

            # check overlaps between detections of model1 and model2
            alert = False
            if dets and dets2:
                for a in dets:
                    for b in dets2:
                        if args.compare_same_class and ('cls' in a and 'cls' in b) and a['cls'] != b['cls']:
                            continue
                        i = iou(a['box'], b['box'])
                        if i >= args.overlap_threshold:
                            alert = True
                            # draw intersection box / highlight
                            xa = max(a['box'][0], b['box'][0])
                            ya = max(a['box'][1], b['box'][1])
                            xb = min(a['box'][2], b['box'][2])
                            yb = min(a['box'][3], b['box'][3])
                            if xb > xa and yb > ya:
                                cv2.rectangle(annotated, (xa, ya), (xb, yb), (0, 0, 255), 2)
                            # draw connecting line between centers
                            ca = ((a['box'][0] + a['box'][2]) // 2, (a['box'][1] + a['box'][3]) // 2)
                            cb = ((b['box'][0] + b['box'][2]) // 2, (b['box'][1] + b['box'][3]) // 2)
                            cv2.line(annotated, ca, cb, (0, 0, 255), 1)

            # overlay alert text if needed
            if alert:
                txt = args.alert_text
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                cv2.rectangle(annotated, (10, 10), (10 + tw + 10, 10 + th + 10), (0, 0, 255), -1)
                cv2.putText(annotated, txt, (15, 10 + th), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            # initialize writer on first frame when output requested
            if args.output:
                if writer is None:
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    try:
                        fps = float(fps) if fps and fps > 0 else 30.0
                    except Exception:
                        fps = 30.0
                    h, w = annotated.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))
                    if not writer.isOpened():
                        print(f"Warning: failed to open VideoWriter for '{args.output}' — output disabled")
                        writer = None
                if writer is not None:
                    writer.write(annotated)

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
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        elapsed = time.time() - t0
        fps = frame_count / elapsed if elapsed > 0 else 0
        print(f"Processed {frame_count} frames in {elapsed:.2f}s ({fps:.2f} FPS)")


if __name__ == '__main__':
    main()
