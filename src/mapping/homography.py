from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np
import cv2

@dataclass
class Homography:
    H: np.ndarray  
    pitch_length_m: float
    pitch_width_m: float

    def img_to_world(self, p_xy: np.ndarray) -> np.ndarray:
        pt = np.array([[p_xy]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self.H)
        return out[0, 0].astype(np.float32)

def load_or_compute_homography(
    homography_matrix_path: Optional[str],
    image_points_px: Sequence[Sequence[float]],
    pitch_points_m: Sequence[Sequence[float]],
    pitch_length_m: float,
    pitch_width_m: float,
) -> Homography:
    if homography_matrix_path:
        H = np.load(str(homography_matrix_path)).astype(np.float32)
        if H.shape != (3, 3):
            raise ValueError(f"Homography matrix must be 3x3, got {H.shape}")
        return Homography(H=H, pitch_length_m=float(pitch_length_m), pitch_width_m=float(pitch_width_m))

    if len(image_points_px) < 4 or len(pitch_points_m) < 4:
        raise ValueError("Need at least 4 point correspondences to compute homography.")
    src = np.array(image_points_px, dtype=np.float32)
    dst = np.array(pitch_points_m, dtype=np.float32)

    if len(src) > 4:
        H, _mask = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    else:
        H = cv2.getPerspectiveTransform(src, dst)

    if H is None:
        raise RuntimeError("Failed to compute homography. Check your point correspondences.")
    return Homography(H=H.astype(np.float32), pitch_length_m=float(pitch_length_m), pitch_width_m=float(pitch_width_m))
