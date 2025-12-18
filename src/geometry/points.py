from __future__ import annotations
import numpy as np

def bbox_center_xy(bbox_xyxy: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = bbox_xyxy.astype(np.float32)
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)

def bbox_foot_xy(bbox_xyxy: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = bbox_xyxy.astype(np.float32)
    return np.array([(x1 + x2) / 2.0, y2], dtype=np.float32)

def clip_point(p: np.ndarray, w: int, h: int) -> np.ndarray:
    return np.array([float(np.clip(p[0], 0, w - 1)), float(np.clip(p[1], 0, h - 1))], dtype=np.float32)
