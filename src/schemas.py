from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np

@dataclass
class ObjState:
    track_id: int
    cls: str
    conf: float
    bbox_xyxy: np.ndarray 
    p_img: Optional[np.ndarray] = None
    p_img_mc: Optional[np.ndarray] = None
    p_world: Optional[np.ndarray] = None
    speed_kmh: Optional[float] = None
    accel_ms2: Optional[float] = None
    distance_m: Optional[float] = None
    team_id: Optional[int] = None
    team_conf: Optional[float] = None
    has_ball: bool = False

@dataclass
class FrameState:
    frame_idx: int
    timestamp_s: float
    players: Dict[int, ObjState] = field(default_factory=dict)
    referees: Dict[int, ObjState] = field(default_factory=dict)
    ball_xyxy: Optional[np.ndarray] = None
    ball_conf: Optional[float] = None
    ball_p_img: Optional[np.ndarray] = None
    ball_p_img_mc: Optional[np.ndarray] = None
    ball_p_world: Optional[np.ndarray] = None
    ball_speed_kmh: Optional[float] = None
    ball_distance_m: Optional[float] = None
    team_in_possession: Optional[int] = None
    player_in_possession: Optional[int] = None
    cam_dx: float = 0.0
    cam_dy: float = 0.0
