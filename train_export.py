# train_export.py
import argparse
from ultralytics import YOLO
import os
import shutil
import yaml

def train_and_export(data_yaml, model='yolov8n.pt', epochs=50, imgsz=640, batch=16, device='0'):
    """
    Trains a YOLO model and exports to ONNX and TensorRT engine.
    Uses Ultralytics YOLO (v8+). For YOLOv5, use the yolov5 repo workflow.
    """
    print("Training:", model)
    model_obj = YOLO(model) 
    result = model_obj.train(data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch, device=device)
    run_dir = None
    for f in result.files:
        if 'weights' in f or f.endswith('.pt'):
            run_dir = os.path.dirname(f)
            break
    if run_dir is None:
        run_dir = 'runs/detect'
    p = os.path.join(run_dir, 'weights', 'best.pt')
    if not os.path.exists(p):
        p = os.path.join(run_dir, 'weights', 'last.pt')
    if not os.path.exists(p):
        raise FileNotFoundError("Couldn't find trained weights. Check training output.")
    print("Trained weights:", p)

    print("Exporting to onnx and TensorRT (engine)...")
    y = YOLO(p)
    y.export(format='onnx', imgsz=imgsz)
    # export TensorRT engine if environment supports it (on Jetson with TensorRT installed)
    # This will create a .engine file in same folder
    try:
        y.export(format='engine', imgsz=imgsz)  # creates .engine using TensorRT
    except Exception as e:
        print("Engine export failed here (likely not on Jetson or missing TensorRT).", e)
        print("You can convert ONNX->TensorRT on Jetson using trtexec or ultralytics export on the device.")
    print("Export complete. Check the run folder for .onnx and .engine files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="path to har_dataset.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="pretrained model (yolov8n.pt recommended)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--img", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu", help="cuda device or 'cpu'")
    args = parser.parse_args()
    train_and_export(args.data, model=args.model, epochs=args.epochs, imgsz=args.img, batch=args.batch, device=args.device)
