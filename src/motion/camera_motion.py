from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cv2

@dataclass
class CameraMotion:
    dx: float
    dy: float
    cum_dx: float
    cum_dy: float

class CameraMotionEstimator:
    def __init__(
        self,
        first_frame_bgr,
        max_features: int = 400,
        min_quality: float = 0.01,
        min_distance: int = 12,
        lk_win_size: int = 21,
        lk_max_level: int = 3,
        smooth_alpha: float = 0.6,
        ransac: bool = False,
    ):
        self.prev_gray = cv2.cvtColor(first_frame_bgr, cv2.COLOR_BGR2GRAY)
        self.max_features = int(max_features)
        self.min_quality = float(min_quality)
        self.min_distance = int(min_distance)
        self.lk_params = dict(
            winSize=(int(lk_win_size), int(lk_win_size)),
            maxLevel=int(lk_max_level),
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        self.smooth_alpha = float(smooth_alpha)
        self.ransac = bool(ransac)
        self._ema_dx = 0.0
        self._ema_dy = 0.0
        self._cum_dx = 0.0
        self._cum_dy = 0.0

    def _good_features(self, gray):
        return cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_features,
            qualityLevel=self.min_quality,
            minDistance=self.min_distance,
            blockSize=7,
        )

    def step(self, frame_bgr) -> CameraMotion:
        curr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        p0 = self._good_features(self.prev_gray)

        dx, dy = 0.0, 0.0
        if p0 is not None and len(p0) >= 10:
            p1, st, _err = cv2.calcOpticalFlowPyrLK(self.prev_gray, curr_gray, p0, None, **self.lk_params)
            if p1 is not None and st is not None:
                good0 = p0[st.squeeze() == 1]
                good1 = p1[st.squeeze() == 1]
                if len(good0) >= 10:
                    if self.ransac and len(good0) >= 12:
                        m, _inliers = cv2.estimateAffinePartial2D(good0, good1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
                        if m is not None:
                            dx, dy = float(m[0, 2]), float(m[1, 2])
                        else:
                            d = (good1 - good0).reshape(-1, 2)
                            dx, dy = float(np.median(d[:, 0])), float(np.median(d[:, 1]))
                    else:
                        d = (good1 - good0).reshape(-1, 2)
                        dx, dy = float(np.median(d[:, 0])), float(np.median(d[:, 1]))

        self._ema_dx = self.smooth_alpha * self._ema_dx + (1.0 - self.smooth_alpha) * dx
        self._ema_dy = self.smooth_alpha * self._ema_dy + (1.0 - self.smooth_alpha) * dy
        self._cum_dx += self._ema_dx
        self._cum_dy += self._ema_dy
        self.prev_gray = curr_gray
        return CameraMotion(dx=self._ema_dx, dy=self._ema_dy, cum_dx=self._cum_dx, cum_dy=self._cum_dy)

    def compensate_point(self, p_xy: np.ndarray) -> np.ndarray:
        return (p_xy - np.array([self._cum_dx, self._cum_dy], dtype=np.float32)).astype(np.float32)
