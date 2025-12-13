import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time

ENGINE_PATH = "object_best.engine"
INPUT_W = 640
INPUT_H = 640
CONF_THRESH = 0.25
IOU_THRESH = 0.45

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

# ----------------------------
# Load TensorRT Engine
# ----------------------------
def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

engine = load_engine(ENGINE_PATH)
context = engine.create_execution_context()

# Allocate buffers
bindings = []
inputs = []
outputs = []
stream = cuda.Stream()

for binding in engine:
    size = trt.volume(engine.get_binding_shape(binding))
    dtype = trt.nptype(engine.get_binding_dtype(binding))
    host_mem = cuda.pagelocked_empty(size, dtype)
    device_mem = cuda.mem_alloc(host_mem.nbytes)
    bindings.append(int(device_mem))
    if engine.binding_is_input(binding):
        input_host = host_mem
        input_device = device_mem
    else:
        output_host = host_mem
        output_device = device_mem

# ----------------------------
# Preprocess
# ----------------------------
def preprocess(img):
    img_resized = cv2.resize(img, (INPUT_W, INPUT_H))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_chw = np.transpose(img_norm, (2, 0, 1))
    return np.expand_dims(img_chw, axis=0).ravel()

# ----------------------------
# Postprocess (YOLOv5)
# ----------------------------
def postprocess(output, img_shape):
    boxes = []
    scores = []
    class_ids = []

    output = output.reshape(-1, 85)  # YOLOv5 output

    for det in output:
        conf = det[4]
        if conf < CONF_THRESH:
            continue
        class_id = np.argmax(det[5:])
        score = conf * det[5 + class_id]

        cx, cy, w, h = det[0:4]
        x1 = int((cx - w / 2) * img_shape[1] / INPUT_W)
        y1 = int((cy - h / 2) * img_shape[0] / INPUT_H)
        x2 = int((cx + w / 2) * img_shape[1] / INPUT_W)
        y2 = int((cy + h / 2) * img_shape[0] / INPUT_H)

        boxes.append([x1, y1, x2, y2])
        scores.append(float(score))
        class_ids.append(class_id)

    return boxes, scores, class_ids

# ----------------------------
# Camera Loop
# ----------------------------
cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Camera not found"

fps_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_data = preprocess(frame)
    np.copyto(input_host, input_data)

    cuda.memcpy_htod_async(input_device, input_host, stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(output_host, output_device, stream)
    stream.synchronize()

    boxes, scores, class_ids = postprocess(output_host, frame.shape)

    for box, score, cls_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID:{cls_id} {score:.2f}",
                    (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1)

    fps = 1.0 / (time.time() - fps_time)
    fps_time = time.time()
    cv2.putText(frame, f"FPS:{fps:.1f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("YOLOv5 TensorRT", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
