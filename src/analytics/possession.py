from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

@dataclass
class PossessionState:
    team_id: Optional[int] = None
    player_id: Optional[int] = None
    hold: int = 0

def assign_possession(ball_p, players: Dict[int, np.ndarray], player_team: Dict[int, int], prev: PossessionState,
                     max_ball_to_player_m: float, hysteresis_m: float, hold_frames: int) -> PossessionState:
    if ball_p is None or len(players) == 0:
        if prev.hold > 0:
            return PossessionState(team_id=prev.team_id, player_id=prev.player_id, hold=prev.hold - 1)
        return PossessionState(team_id=prev.team_id, player_id=prev.player_id, hold=0)

    ids = np.array(list(players.keys()), dtype=np.int32)
    P = np.stack([players[i] for i in ids], axis=0).astype(np.float32)
    d = np.linalg.norm(P - ball_p[None, :], axis=1)
    j = int(np.argmin(d))
    best_id = int(ids[j]); best_d = float(d[j])

    if best_d > max_ball_to_player_m:
        if prev.hold > 0:
            return PossessionState(team_id=prev.team_id, player_id=prev.player_id, hold=prev.hold - 1)
        return PossessionState(team_id=prev.team_id, player_id=prev.player_id, hold=0)

    if prev.player_id is not None and prev.player_id in players and prev.player_id != best_id:
        prev_d = float(np.linalg.norm(players[prev.player_id] - ball_p))
        if prev_d <= best_d + hysteresis_m and prev.hold > 0:
            return PossessionState(team_id=prev.team_id, player_id=prev.player_id, hold=prev.hold - 1)

    team_id = player_team.get(best_id, None)
    return PossessionState(team_id=team_id, player_id=best_id, hold=hold_frames)
