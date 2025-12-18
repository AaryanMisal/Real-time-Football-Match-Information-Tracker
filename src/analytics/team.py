from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
from sklearn.cluster import KMeans

@dataclass
class TeamColorModel:
    centroids: np.ndarray  

def _roi_from_bbox(frame_bgr, bbox_xyxy, roi_rel):
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy.astype(int)
    x1 = max(0, min(w - 1, x1)); x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1)); y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    ch, cw = crop.shape[:2]
    rx1 = int(roi_rel["x1"] * cw); rx2 = int(roi_rel["x2"] * cw)
    ry1 = int(roi_rel["y1"] * ch); ry2 = int(roi_rel["y2"] * ch)
    rx1 = max(0, min(cw - 1, rx1)); rx2 = max(0, min(cw, rx2))
    ry1 = max(0, min(ch - 1, ry1)); ry2 = max(0, min(ch, ry2))
    if rx2 <= rx1 or ry2 <= ry1:
        return None
    return crop[ry1:ry2, rx1:rx2]

def _mean_color(patch, color_space: str) -> Optional[np.ndarray]:
    if patch is None or patch.size == 0:
        return None
    if color_space.upper() == "HSV":
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        return hsv.reshape(-1, 3).mean(axis=0).astype(np.float32)
    return patch.reshape(-1, 3).mean(axis=0).astype(np.float32)

def fit_team_color_model(samples: List[np.ndarray], k: int = 2) -> TeamColorModel:
    X = np.stack(samples, axis=0).astype(np.float32)
    km = KMeans(n_clusters=int(k), n_init="auto", random_state=7)
    km.fit(X)
    return TeamColorModel(centroids=km.cluster_centers_.astype(np.float32))

def assign_team_id(color_vec: np.ndarray, model: TeamColorModel) -> Tuple[int, float]:
    d = np.linalg.norm(model.centroids - color_vec[None, :], axis=1)
    team = int(np.argmin(d))
    sd = np.sort(d)
    conf = float(sd[1] - sd[0]) / float(sd[1] + 1e-6) if len(sd) > 1 else 1.0
    return team, conf

class TeamAssigner:
    def __init__(self, roi_rel: dict, kmeans_k: int, color_space: str = "HSV", smooth_alpha: float = 0.85):
        self.roi_rel = roi_rel
        self.k = int(kmeans_k)
        self.color_space = str(color_space)
        self.smooth_alpha = float(smooth_alpha)
        self.model: Optional[TeamColorModel] = None
        self.player_team_probs: Dict[int, np.ndarray] = {}

    def collect_sample(self, frame_bgr, bbox_xyxy) -> Optional[np.ndarray]:
        patch = _roi_from_bbox(frame_bgr, bbox_xyxy, self.roi_rel)
        return _mean_color(patch, self.color_space)

    def fit(self, samples: List[np.ndarray]):
        self.model = fit_team_color_model(samples, k=self.k)

    def update_player_team(self, player_id: int, color_vec: np.ndarray) -> Tuple[int, float]:
        assert self.model is not None
        team, _ = assign_team_id(color_vec, self.model)
        probs = np.zeros((self.k,), dtype=np.float32)
        probs[team] = 1.0
        if player_id not in self.player_team_probs:
            self.player_team_probs[player_id] = probs
        else:
            self.player_team_probs[player_id] = self.smooth_alpha * self.player_team_probs[player_id] + (1.0 - self.smooth_alpha) * probs
        p = self.player_team_probs[player_id]
        team_out = int(np.argmax(p))
        return team_out, float(p[team_out])
