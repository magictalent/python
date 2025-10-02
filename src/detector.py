# src/detector.py
from ultralytics import YOLO
import numpy as np

class PlayerDetector:
    def __init__(self, model_path="models/yolov8n.pt", device="cpu", conf_thresh=0.4):
        self.model = YOLO(model_path)
        self.device = device
        self.conf_thresh = conf_thresh

    def detect(self, frame):
        """
        Run YOLOv8 on 'frame' and return detections in format:
        [ [x1,y1,x2,y2,confidence, class_id], ... ]
        Only returns class_id == 0 (person) detections.
        """
        # ultralytics expects BGR frames as numpy arrays
        results = self.model(frame, device=self.device)[0]
        dets = []
        if results is None or len(results.boxes) == 0:
            return dets
        for box, conf, cls in zip(results.boxes.xyxy.cpu().numpy(),
                                  results.boxes.conf.cpu().numpy(),
                                  results.boxes.cls.cpu().numpy()):
            # Only person class (COCO id 0)
            if int(cls) != 0:
                continue
            x1, y1, x2, y2 = box.astype(int).tolist()
            if conf < self.conf_thresh:
                continue
            dets.append([x1, y1, x2, y2, float(conf), int(cls)])
        return dets
