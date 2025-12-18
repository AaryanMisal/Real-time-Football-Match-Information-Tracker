from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np
from ultralytics import YOLO

@dataclass
class Detection:
    xyxy: np.ndarray  
    conf: float
    cls_id: int

class YOLODetector:
    def __init__(self, weights: str, device: str = "cpu", conf: float = 0.25, iou: float = 0.45, imgsz: int = 960):
        self.model = YOLO(weights)
        self.device = device
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz

    def infer(self, frame_bgr) -> List[Detection]:
        results = self.model.predict(
            source=frame_bgr,
            device=self.device,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False
        )
        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return []
        xyxy = r0.boxes.xyxy.cpu().numpy().astype(np.float32)
        conf = r0.boxes.conf.cpu().numpy().astype(np.float32)
        cls = r0.boxes.cls.cpu().numpy().astype(np.int32)
        return [Detection(xyxy=xyxy[i], conf=float(conf[i]), cls_id=int(cls[i])) for i in range(len(xyxy))]
