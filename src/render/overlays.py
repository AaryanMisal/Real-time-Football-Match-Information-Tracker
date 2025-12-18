from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
from ..schemas import FrameState, ObjState

@dataclass
class RenderConfig:
    show_ids: bool = True
    show_speed: bool = True
    show_distance: bool = True
    show_team: bool = True
    show_possession_bar: bool = True
    show_ball_speed: bool = True
    thickness: int = 2
    font_scale: float = 0.6

def _team_color(team_id: Optional[int]) -> Tuple[int, int, int]:
    if team_id == 0:
        return (255, 80, 80)
    if team_id == 1:
        return (80, 80, 255)
    return (200, 200, 200)

class OverlayRenderer:
    def __init__(self, cfg: RenderConfig):
        self.cfg = cfg

    def _label(self, o: ObjState) -> str:
        parts = []
        if self.cfg.show_ids:
            parts.append(f"#{o.track_id}")
        if self.cfg.show_team and o.team_id is not None:
            parts.append(f"T{o.team_id}")
        if self.cfg.show_speed and o.speed_kmh is not None:
            parts.append(f"{o.speed_kmh:.1f} km/h")
        if self.cfg.show_distance and o.distance_m is not None:
            parts.append(f"{o.distance_m:.1f} m")
        if o.has_ball:
            parts.append("⚽")
        return " | ".join(parts)

    def draw(self, frame_bgr, state: FrameState):
        for o in list(state.players.values()) + list(state.referees.values()):
            x1, y1, x2, y2 = o.bbox_xyxy.astype(int)
            c = _team_color(o.team_id)
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), c, thickness=self.cfg.thickness)
            lab = self._label(o)
            if lab:
                cv2.putText(frame_bgr, lab, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                            self.cfg.font_scale, c, thickness=self.cfg.thickness, lineType=cv2.LINE_AA)

        if state.ball_p_img is not None:
            bx, by = state.ball_p_img.astype(int)
            cv2.circle(frame_bgr, (bx, by), 6, (0, 255, 255), -1)
            cv2.circle(frame_bgr, (bx, by), 10, (0, 0, 0), 2)
            if self.cfg.show_ball_speed and state.ball_speed_kmh is not None:
                cv2.putText(frame_bgr, f"ball {state.ball_speed_kmh:.1f} km/h", (bx + 12, by - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, self.cfg.font_scale, (0, 255, 255), self.cfg.thickness, cv2.LINE_AA)

        if self.cfg.show_possession_bar and state.team_in_possession is not None:
            cv2.putText(frame_bgr, f"Possession: Team {state.team_in_possession}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, _team_color(state.team_in_possession), 2, cv2.LINE_AA)
        return frame_bgr
