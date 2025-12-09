#!/usr/bin/env python3
"""
realtime_infer.py

Real-time threaded inference pipeline that:
 - captures frames from camera/video
 - runs detection with Ultralytics YOLO (supports .engine / .onnx / .pt)
 - draws boxes, labels, FPS
 - calls alert_callback(detections, frame, timestamp) when detection of interest occurs

Usage:
    python realtime_infer.py --model /home/ubuntu/HAR_dataset/runs/exp/weights/model.engine --size 320 --source 0

Author: adapted for HAR project
"""

import argparse
import time
import threading
import queue
from datetime import datetime

import cv2
from ultralytics import YOLO
import numpy as np
import os

CAP_QUEUE_MAX = 4
INF_QUEUE_MAX = 4

DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
DEFAULT_SIZE = 320  

CRITICAL_CLASSES = {'axe','knife','scissor','socket','window'}  


def gstreamer_pipeline(capture_width=1280, capture_height=720, display_width=640, display_height=360, framerate=30, flip_method=0):
    """
    Return a GStreamer pipeline string for Jetson CSI camera (nvarguscamerasrc).
    Use this string as cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    """
    return (
        "nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        "videoconvert ! video/x-raw, format=(string)BGR ! appsink"
    )


def validate_source(src, use_gstreamer=False, max_test_index=4):
    """
    Try to validate and resolve a video source. If `src` opens fine return it.
    If `src` is an integer index that fails, try fallback indices from 0..max_test_index-1
    and return the first working index. If none work, raise RuntimeError with a
    helpful message.
    """
    # If using GStreamer we assume the provided pipeline string is intentional
    if use_gstreamer:
        return src

    # If src is a filepath (string) and exists, return it
    try:
        if isinstance(src, str) and os.path.exists(src):
            return src
    except Exception:
        pass

    # Try opening the provided source directly
    try:
        cap = cv2.VideoCapture(src)
        if cap is not None and cap.isOpened():
            cap.release()
            return src
    except Exception:
        pass

    # If src looks numeric (string or int), try integer indices and fallbacks
    try:
        requested = int(src)
    except Exception:
        requested = None

    if requested is not None:
        # try the requested index first, then 0..max_test_index-1
        indices = [requested] + [i for i in range(max_test_index) if i != requested]
        for idx in indices:
            try:
                cap = cv2.VideoCapture(idx)
                if cap is not None and cap.isOpened():
                    cap.release()
                    return idx
            except Exception:
                pass

    # nothing worked
    tested = f"numeric indices 0..{max_test_index-1}" if requested is None else f"requested {requested} and 0..{max_test_index-1}"
    raise RuntimeError(f"Unable to open video source '{src}'. Tested: {tested}.\n" \
                       "Check that the camera is connected, the index is correct, or pass a video file path.")


def alert_callback(detections, frame, timestamp):
    """
    detections: list of dicts: [{'name': str, 'conf': float, 'box': [x1,y1,x2,y2]}, ...]
    frame: BGR numpy array of current frame
    timestamp: datetime
    Customize this to send SMS / HTTP / MQTT / GPIO signal.
    """
    #
    critical = [d for d in detections if d['name'] in CRITICAL_CLASSES and d['conf'] >= 0.5]
    if critical:
        print(f"[ALERT {timestamp.isoformat()}] Critical detections:", critical)
        # Save evidence image (optional)
        fn = f"alert_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(fn, frame)
        # TODO: add HTTP POST or MQTT publish to your backend here
        # Example (pseudo):
        # requests.post("https://your-server/events", json=payload, timeout=2)


class CaptureThread(threading.Thread):
    def __init__(self, src, cap_queue, width=None, height=None, use_gstreamer=False):
        super().__init__(daemon=True)
        self.src = src
        self.cap_queue = cap_queue
        self.running = True
        self.width = width
        self.height = height
        self.use_gstreamer = use_gstreamer
        self.cap = None

    def run(self):
        if self.use_gstreamer and isinstance(self.src, str):
            cap = cv2.VideoCapture(self.src, cv2.CAP_GSTREAMER)
        else:
            cap = cv2.VideoCapture(self.src)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video source {self.src}")
        if self.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self.cap = cap
        while self.running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            try:
                self.cap_queue.put_nowait((frame, time.time()))
            except queue.Full:
                try:
                    _ = self.cap_queue.get_nowait()
                    self.cap_queue.put_nowait((frame, time.time()))
                except queue.Empty:
                    pass
        cap.release()

    def stop(self):
        self.running = False


class InferenceThread(threading.Thread):
    def __init__(self, model_path, cap_queue, out_queue, conf_thres=DEFAULT_CONF, iou_thres=DEFAULT_IOU, imgsz=DEFAULT_SIZE, device='cpu'):
        super().__init__(daemon=True)
        self.cap_queue = cap_queue
        self.out_queue = out_queue
        self.running = True
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.imgsz = imgsz
        self.model_path = model_path
        self.device = device
        self.model = None

    def _init_model(self):
        self.model = YOLO(self.model_path)
        # set model conf / iou defaults if supported
        # Ultralytics model(...) call supports conf and iou via call params

    def run(self):
        self._init_model()
        while self.running:
            try:
                frame, ts = self.cap_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            # Run inference using Ultralytics API (handles .engine/onnx/pt)
            # return value is Results object(s); calling with imgsz and device is optional
            results = self.model(frame, imgsz=self.imgsz, conf=self.conf_thres, iou=self.iou_thres, device=self.device)

            # parse detections
            dets = []
            # results may be a list; take first
            r = results[0]
            if hasattr(r, 'boxes') and r.boxes is not None:
                boxes = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes, 'xyxy') else np.array([])
                confs = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, 'conf') else np.array([])
                cls_idxs = r.boxes.cls.cpu().numpy().astype(int) if hasattr(r.boxes, 'cls') else np.array([])
                for (bb, conf, cls_idx) in zip(boxes, confs, cls_idxs):
                    name = self.model.model.names[int(cls_idx)] if hasattr(self.model, 'model') and hasattr(self.model.model, 'names') else str(cls_idx)
                    x1, y1, x2, y2 = map(int, bb.tolist())
                    dets.append({'name': name, 'conf': float(conf), 'box': [x1, y1, x2, y2]})

            # push annotated info
            try:
                self.out_queue.put_nowait((frame, ts, dets))
            except queue.Full:
                # drop if congested
                pass

    def stop(self):
        self.running = False


def draw_detections(frame, detections):
    for d in detections:
        x1, y1, x2, y2 = d['box']
        label = f"{d['name']} {d['conf']:.2f}"
        # bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (16, 200, 100), 2)
        # label background
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - 18), (x1 + w, y1), (16, 200, 100), -1)
        cv2.putText(frame, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return frame

def main(args):
    # Queues
    cap_q = queue.Queue(maxsize=CAP_QUEUE_MAX)
    out_q = queue.Queue(maxsize=INF_QUEUE_MAX)

    # Camera source: if user supplies "gstreamer" flag, build pipeline
    use_gst = False
    src = args.source
    gst_pipeline = None
    if args.use_gstreamer:
        use_gst = True
        gst_pipeline = gstreamer_pipeline(capture_width=args.cap_w, capture_height=args.cap_h,
                                         display_width=args.display_w, display_height=args.display_h,
                                         framerate=args.fps, flip_method=args.flip)
        src = gst_pipeline
        print("Using GStreamer pipeline:", gst_pipeline)

    # Start capture thread
    # Validate the source and possibly get a fallback index before creating capture thread
    try:
        src = validate_source(src, use_gstreamer=use_gst)
    except RuntimeError as e:
        print("[ERROR]", e)
        return

    cap_thread = CaptureThread(src=src, cap_queue=cap_q, width=args.cap_w if not use_gst else None,
                               height=args.cap_h if not use_gst else None, use_gstreamer=use_gst)
    cap_thread.start()

    # Start inference thread
    inf_thread = InferenceThread(model_path=args.model, cap_queue=cap_q, out_queue=out_q,
                                 conf_thres=args.conf, iou_thres=args.iou, imgsz=args.size)
    inf_thread.start()

    # Main visualization loop
    fps_smooth = None
    frame_count = 0
    t0 = time.time()
    try:
        while True:
            try:
                frame, ts, dets = out_q.get(timeout=1.0)
            except queue.Empty:
                continue

            # trigger alert callback (user can customize)
            if dets:
                alert_callback(dets, frame.copy(), datetime.fromtimestamp(ts))

            # draw detections and FPS
            annotated = draw_detections(frame, dets)

            # compute FPS
            frame_count += 1
            elapsed = time.time() - t0
            fps = frame_count / elapsed if elapsed > 0 else 0.0
            fps_smooth = fps if fps_smooth is None else (0.9 * fps_smooth + 0.1 * fps)
            cv2.putText(annotated, f"FPS {fps_smooth:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            # show
            if args.display:
                cv2.imshow("Realtime Inference", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # headless: optionally save frames or do nothing
                pass

    except KeyboardInterrupt:
        print("Stopping...")

    # cleanup
    cap_thread.stop()
    inf_thread.stop()
    time.sleep(0.5)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model (.engine, .onnx, .pt).")
    parser.add_argument("--source", default=0, help="Camera index, video path, or '0' for default webcam. Use integer for USB cam.")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE, help="Inference image size (model input).")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU, help="NMS IoU threshold.")
    parser.add_argument("--display", action="store_true", help="Show GUI window (cv2.imshow).")
    parser.add_argument("--device", default="cpu", help="Computation device: 'cpu' or CUDA index like '0' or '0,1'.")
    # Jetson-specific GST options
    parser.add_argument("--use-gstreamer", action="store_true", help="Use GStreamer nvarguscamera pipeline (Jetson CSI).")
    parser.add_argument("--cap-w", type=int, default=1280)
    parser.add_argument("--cap-h", type=int, default=720)
    parser.add_argument("--display-w", type=int, default=640)
    parser.add_argument("--display-h", type=int, default=360)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--flip", type=int, default=0)
    args = parser.parse_args()

    # Normalize source type (int if numeric)
    try:
        args.source = int(args.source)
    except Exception:
        args.source = args.source

    main(args)
