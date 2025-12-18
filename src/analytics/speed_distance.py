from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class Kinematics:
    speed_kmh: float
    accel_ms2: float
    distance_m: float

def compute_kinematics(prev_p: Optional[np.ndarray], prev_speed_ms: Optional[float], curr_p: Optional[np.ndarray], dt_s: float) -> Optional[Kinematics]:
    if prev_p is None or curr_p is None or dt_s <= 1e-6:
        return None
    dp = curr_p - prev_p
    dist = float(np.linalg.norm(dp))
    speed_ms = dist / dt_s
    speed_kmh = speed_ms * 3.6
    accel = 0.0 if prev_speed_ms is None else (speed_ms - prev_speed_ms) / dt_s
    return Kinematics(speed_kmh=float(speed_kmh), accel_ms2=float(accel), distance_m=float(dist))

def speed_zones(speed_kmh: float, jog_kmh: float, high_intensity_kmh: float, sprint_kmh: float) -> str:
    if speed_kmh >= sprint_kmh:
        return "sprint"
    if speed_kmh >= high_intensity_kmh:
        return "high_intensity"
    if speed_kmh >= jog_kmh:
        return "jog"
    return "walk/stand"
