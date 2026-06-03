# ONNX inference engine distilled from notebook4
import numpy as np
import cv2
import onnxruntime as ort
import yaml
from pathlib import Path

# load class names from dat.yaml file
with open('data.yaml') as f:
    class_names = yaml.safe_load(f)['names']
# Map class index → violation label for the two classes we care about
# Check YOUR data.yaml — indices may differ
VIOLATION_CLASS_INDICES = {i for i, n in enumerate(class_names) if 'NO-' in n and 'Safety' in n or n=='NO-Hardhat'}

class ONNXEngine:
    def __init__(self, model_path: str):
        # Use GPU if available, else CPU
        self.session = ort.InferenceSession(model_path,
        providers=['CUDAExecutionProvider','CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.img_size = 640
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        img = cv2.resize(frame, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1)) # HWC -> CHW
        return np.expand_dims(img, 0) # add batch dim -> BCHW
    
    def postprocess(self, outputs, orig_shape, conf_thresh=0.4, iou_thresh=0.5):
        # YOLOv8 ONNX output: [1, num_classes+4, num_anchors]
        output = outputs[0][0] # shape: [84, 8400] for 80-class; [14, 8400] for 10-class
        boxes_raw = output[:4].T # [anchors, 4] — cx, cy, w, h normalized
        scores = output[4:].T # [anchors, num_classes]
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        mask = confidences > conf_thresh
        boxes_raw = boxes_raw[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]
        # Scale back to original image size
        h, w = orig_shape[:2]
        sx, sy = w / self.img_size, h / self.img_size
        x1 = (boxes_raw[:,0] - boxes_raw[:,2]/2) * sx
        y1 = (boxes_raw[:,1] - boxes_raw[:,3]/2) * sy
        x2 = (boxes_raw[:,0] + boxes_raw[:,2]/2) * sx
        y2 = (boxes_raw[:,1] + boxes_raw[:,3]/2) * sy
        # NMS — remove overlapping boxes for the same object
        xywh = list(zip(x1.tolist(), y1.tolist(), (x2-x1).tolist(), (y2-y1).tolist()))
        indices = cv2.dnn.NMSBoxes(xywh, confidences.tolist(), conf_thresh, iou_thresh)
        detections = []
        for i in (indices if len(indices) else []):
            detections.append({
            'class_id': int(class_ids[i]),
            'class_name': class_names[int(class_ids[i])],
            'confidence': float(confidences[i]),
            'bbox': [float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])], 'is_violation': int(class_ids[i]) in VIOLATION_CLASS_INDICES})
        return detections

# Module-level singleton — loaded once when the server starts
engine = ONNXEngine('model/best.onnx')
