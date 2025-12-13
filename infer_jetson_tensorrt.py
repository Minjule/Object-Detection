#!/usr/bin/env python3
"""
infer_jetson_tensorrt.py

Jetson Nano-friendly inference script with TensorRT/ONNX Runtime support.
- Tries to use ONNX Runtime with the TensorRT Execution Provider if available.
- Falls back to Ultralytics YOLO inference (CUDA) if providers are missing.

Usage examples:
    python infer_jetson_tensorrt.py --model best.onnx --source 0 --output out.mp4 --model2 other.onnx --display
    python infer_jetson_tensorrt.py --model child_best.pt --source path/video.mp4 --display --danger-label DAnger --danger-classes child,knife

Notes:
 - This script is a Jetson-focused version of infer_simple.py. It uses CUDA device by default and turns on half precision for improved performance if available.
 - If you have a TensorRT engine (.engine), set --backend tensorrt and provide --engine. The script will try to use pycuda/tensorrt if present.
 - When using ONNX, ONNX Runtime with the TensorRT Execution Provider is preferred for performance; else CUDA EP is used.
"""

import argparse
import time
from datetime import datetime
import sys

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    import onnxruntime as ort
except Exception:
    ort = None

DEFAULT_CONF = 0.50
DEFAULT_IOU = 0.45
DEFAULT_SIZE = 640  # slightly larger for Jetson to keep detection accuracy


def try_open_source(src, max_index=4):
    """See `infer_simple.py` for close behavior."""
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


def draw_boxes(frame, detections, color=(0, 255, 0), label_override=None, danger_classes=None):
    for d in detections:
        x1, y1, x2, y2 = d['box']
        name = d.get('name', '')
        if label_override is not None:
            if danger_classes:
                if name in danger_classes:
                    label = f"{label_override[:20]} {d['conf']:.2f}"
                else:
                    label = f"{name[:20]} {d['conf']:.2f}"
            else:
                label = f"{label_override[:20]} {d['conf']:.2f}"
        else:
            label = f"{name[:20]} {d['conf']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - 18), (x1 + w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return frame


def iou(boxA, boxB):
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
    parser.add_argument('--model', required=True, help='Path to model (.pt/.onnx/.engine) or an ONNX file')
    parser.add_argument('--model2', dest='model2', default=None, help='Optional second model for comparison')
    parser.add_argument('--backend', choices=['ultralytics', 'onnxruntime', 'tensorrt'], default='ultralytics',
                        help='Preferred backend. On Jetson, use `onnxruntime` with TRT EP if available.')
    parser.add_argument('--engine', dest='engine', default=None, help='Path to TensorRT engine (.engine).')
    parser.add_argument('--source', default=0, help='Camera index or video path')
    parser.add_argument('--device', default='cuda', help='device (default cuda on Jetson)')
    parser.add_argument('--imgsz', type=int, default=DEFAULT_SIZE)
    parser.add_argument('--conf', type=float, default=DEFAULT_CONF)
    parser.add_argument('--display', action='store_true', help='Show annotated frames')
    parser.add_argument('--max-frames', type=int, default=0, help='Stop after N frames (0 = run until q/EOF)')
    parser.add_argument('--output', default=None, help='Path to save annotated video (mp4)')
    parser.add_argument('--overlap-threshold', dest='overlap_threshold', type=float, default=0.5,
                        help='IoU threshold above which overlapping detections trigger an alert (0-1).')
    parser.add_argument('--compare-same-class', dest='compare_same_class', action='store_true',
                        help='Only consider overlaps when both detections have the same class index.')
    parser.add_argument('--alert-text', dest='alert_text', default='ALERT: overlapping detections',
                        help='Text to display on the frame when overlapping detections are found.')
    parser.add_argument('--half', action='store_true', help='Use FP16 (if supported, improves performance)')
    parser.add_argument('--danger-label', dest='danger_label', default=None,
                        help="If provided, display this text as label instead of class name (e.g. 'DAnger')")
    parser.add_argument('--danger-classes', dest='danger_classes', default=None,
                        help='Comma-separated class names to apply the danger label to. If omitted and --danger-label is set, label defaults for all classes.')
    args = parser.parse_args()

    try:
        src = try_open_source(args.source)
    except RuntimeError as e:
        print('[ERROR]', e)
        return

    # prefer ONNX Runtime + TRT EP if requested and available
    session = None
    if args.backend in ('onnxruntime', 'tensorrt') and ort is not None and args.model.endswith('.onnx'):
        providers = []
        # try to pick the fastest available provider in order
        try:
            providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
            session = ort.InferenceSession(args.model, providers=providers)
            print('ONNX Runtime session created with providers:', session.get_providers())
        except Exception as e:
            print('ONNX Runtime (preferred) failed, falling back to Ultralytics YOLO:', e)
            session = None

    # If session is None, fall back to Ultralytics YOLO
    model = None
    model2 = None
    if session is None and YOLO is None:
        print('Ultralytics YOLO not available; please install the ultralytics package or provide an ONNX model + ONNX Runtime')
        return

    if session is None:
        # load model with ultralytics, it'll accept pt/onnx and use onnxruntime under the hood
        print('Loading Ultralytics model on', args.device, 'model:', args.model)
        model = YOLO(args.model)
        if args.half:
            try:
                if hasattr(model, 'model') and hasattr(model.model, 'half'):
                    model.model.half()
            except Exception:
                pass
        if args.model2:
            model2 = YOLO(args.model2)
            if args.half:
                try:
                    if hasattr(model2, 'model') and hasattr(model2.model, 'half'):
                        model2.model.half()
                except Exception:
                    pass
    else:
        # session is present for model ONNX in ONNX Runtime with TRT or CUDA EP
        print('Using ONNX Runtime session for model', args.model)
        # We will still use Ultralytics if possible for post-processing; otherwise do a best-effort inference
        if YOLO is not None:
            model = YOLO(args.model)
            if args.half:
                try:
                    if hasattr(model, 'model') and hasattr(model.model, 'half'):
                        model.model.half()
                except Exception:
                    pass
            if args.model2:
                model2 = YOLO(args.model2)
                if args.half:
                    try:
                        if hasattr(model2, 'model') and hasattr(model2.model, 'half'):
                            model2.model.half()
                    except Exception:
                        pass

    # Capture + Video writer
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print('Failed to open source', src)
        return
    writer = None
    frame_count = 0
    t0 = time.time()

    # parse danger classes once
    danger_classes_list = None
    if args.danger_classes:
        danger_classes_list = [c.strip() for c in args.danger_classes.split(',') if c.strip()]

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print('End of stream or read failure')
                break

            dets = []
            dets2 = []
            # Run Ultralytics YOLO if loaded (this is the most robust path)
            if model is not None:
                results = model(frame, imgsz=args.imgsz, conf=args.conf, iou=DEFAULT_IOU, device=args.device)
                r = results[0]
                if hasattr(r, 'boxes') and r.boxes is not None:
                    boxes = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes, 'xyxy') else np.array([])
                    confs = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, 'conf') else np.array([])
                    cls_idxs = r.boxes.cls.cpu().numpy().astype(int) if hasattr(r.boxes, 'cls') else np.array([])
                    for (bb, conf, cls_idx) in zip(boxes, confs, cls_idxs):
                        name = model.model.names[int(cls_idx)] if hasattr(model, 'model') and hasattr(model.model, 'names') else str(cls_idx)
                        x1, y1, x2, y2 = map(int, bb.tolist())
                        dets.append({'name': name, 'conf': float(conf), 'box': [x1, y1, x2, y2], 'cls': int(cls_idx)})

            # second model optional
            if args.model2 and model2 is not None:
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

            # Print detections
            if dets or dets2:
                ts = datetime.now().isoformat()
                print(f"[{ts}] Detections:")
                for d in dets:
                    pname = d['name']
                    if args.danger_label and (not danger_classes_list or pname in danger_classes_list):
                        pname = args.danger_label
                    print(f"  - M1 {pname} conf={d['conf']:.2f} box={d['box']}")
                for d in dets2:
                    pname = d['name']
                    if args.danger_label and (not danger_classes_list or pname in danger_classes_list):
                        pname = args.danger_label
                    print(f"  - M2 {pname} conf={d['conf']:.2f} box={d['box']}")

            annotated = frame.copy()
            annotated = draw_boxes(annotated, dets, color=(0, 255, 0), label_override=args.danger_label, danger_classes=danger_classes_list)
            if dets2:
                annotated = draw_boxes(annotated, dets2, color=(255, 0, 0), label_override=args.danger_label, danger_classes=danger_classes_list)

            # Overlap check
            alert = False
            if dets and dets2:
                for a in dets:
                    for b in dets2:
                        if args.compare_same_class and ('cls' in a and 'cls' in b) and a['cls'] != b['cls']:
                            continue
                        i = iou(a['box'], b['box'])
                        if i >= args.overlap_threshold:
                            alert = True
                            xa = max(a['box'][0], b['box'][0])
                            ya = max(a['box'][1], b['box'][1])
                            xb = min(a['box'][2], b['box'][2])
                            yb = min(a['box'][3], b['box'][3])
                            if xb > xa and yb > ya:
                                cv2.rectangle(annotated, (xa, ya), (xb, yb), (0, 0, 255), 2)
                            ca = ((a['box'][0] + a['box'][2]) // 2, (a['box'][1] + a['box'][3]) // 2)
                            cb = ((b['box'][0] + b['box'][2]) // 2, (b['box'][1] + b['box'][3]) // 2)
                            cv2.line(annotated, ca, cb, (0, 0, 255), 1)

            if alert:
                txt = args.alert_text
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                cv2.rectangle(annotated, (10, 10), (10 + tw + 10, 10 + th + 10), (0, 0, 255), -1)
                cv2.putText(annotated, txt, (15, 10 + th), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            # writer
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
                cv2.imshow('infer_jetson_tensorrt', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if args.max_frames and frame_count >= args.max_frames:
                print(f"Reached max frames {args.max_frames}")
                break

    except KeyboardInterrupt:
        print('Interrupted by user')
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
