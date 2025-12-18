from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from ..utils.math import rolling_mean

@dataclass
class BallSeries:
    frame_idx: np.ndarray
    p: np.ndarray
    speed_kmh: np.ndarray
    cumulative_dist_m: np.ndarray

def _interpolate_xy(frames: np.ndarray, xs: np.ndarray, ys: np.ndarray, max_gap: int) -> Tuple[np.ndarray, np.ndarray]:
    s_x = pd.Series(xs, index=frames, dtype="float32")
    s_y = pd.Series(ys, index=frames, dtype="float32")
    s_xi = s_x.interpolate(method="linear", limit=max_gap, limit_area="inside")
    s_yi = s_y.interpolate(method="linear", limit=max_gap, limit_area="inside")
    return s_xi.values.astype(np.float32), s_yi.values.astype(np.float32)

def clean_ball_track(frame_indices_all: np.ndarray, p_list: List[Optional[np.ndarray]], fps: float, max_gap_frames: int = 12,
                     smoothing_window: int = 7, outlier_max_speed_kmh: float = 120.0) -> BallSeries:
    N = len(p_list)
    assert N == len(frame_indices_all)
    xs = np.array([p[0] if p is not None else np.nan for p in p_list], dtype=np.float32)
    ys = np.array([p[1] if p is not None else np.nan for p in p_list], dtype=np.float32)

    xi, yi = _interpolate_xy(frame_indices_all, xs, ys, max_gap=max_gap_frames)

    dt = 1.0 / float(fps)
    dx = np.diff(xi)
    dy = np.diff(yi)
    dist = np.sqrt(dx * dx + dy * dy)
    speed_kmh = np.concatenate([[0.0], (dist / dt) * 3.6]).astype(np.float32)

    bad = speed_kmh > float(outlier_max_speed_kmh)
    xi2 = xi.copy(); yi2 = yi.copy()
    xi2[bad] = np.nan; yi2[bad] = np.nan
    xi2, yi2 = _interpolate_xy(frame_indices_all, xi2, yi2, max_gap=max_gap_frames)

    xi_sm = rolling_mean(xi2.astype(np.float32), smoothing_window).astype(np.float32)
    yi_sm = rolling_mean(yi2.astype(np.float32), smoothing_window).astype(np.float32)

    dx = np.diff(xi_sm)
    dy = np.diff(yi_sm)
    dist = np.sqrt(dx * dx + dy * dy).astype(np.float32)
    speed_kmh = np.concatenate([[0.0], (dist / dt) * 3.6]).astype(np.float32)
    cumdist = np.concatenate([[0.0], np.cumsum(dist)]).astype(np.float32)

    p = np.stack([xi_sm, yi_sm], axis=1).astype(np.float32)
    return BallSeries(frame_idx=frame_indices_all.astype(np.int32), p=p, speed_kmh=speed_kmh, cumulative_dist_m=cumdist)
