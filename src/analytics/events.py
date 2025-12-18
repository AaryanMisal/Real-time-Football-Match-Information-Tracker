from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Event:
    frame_idx: int
    t_s: float
    type: str
    team_id: Optional[int]
    from_player: Optional[int]
    to_player: Optional[int]
    note: str = ""

def infer_events_from_possession(frame_idx: int, t_s: float, prev_team: Optional[int], prev_player: Optional[int],
                                 curr_team: Optional[int], curr_player: Optional[int]) -> List[Event]:
    ev: List[Event] = []
    if prev_team is None and curr_team is not None:
        ev.append(Event(frame_idx, t_s, "possession_start", curr_team, None, curr_player))
    if prev_team is not None and curr_team is None:
        ev.append(Event(frame_idx, t_s, "possession_end", prev_team, prev_player, None))

    if prev_team is None or curr_team is None:
        if prev_player is not None and curr_player == prev_player:
            ev.append(Event(frame_idx, t_s, "touch", curr_team, curr_player, curr_player))
        return ev

    if prev_team == curr_team and prev_player is not None and curr_player is not None and prev_player != curr_player:
        ev.append(Event(frame_idx, t_s, "pass", curr_team, prev_player, curr_player))
    elif prev_team != curr_team:
        ev.append(Event(frame_idx, t_s, "turnover", curr_team, prev_player, curr_player, note=f"{prev_team}->{curr_team}"))
    else:
        ev.append(Event(frame_idx, t_s, "touch", curr_team, curr_player, curr_player))
    return ev
