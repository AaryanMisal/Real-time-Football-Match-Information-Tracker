from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple
import cv2

@dataclass
class VideoMeta:
    fps: float
    width: int
    height: int
    frame_count: int

def get_video_meta(video_path: str) -> VideoMeta:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return VideoMeta(fps=float(fps), width=width, height=height, frame_count=frame_count)

def iter_frames(video_path: str, start_frame: int = 0, end_frame: Optional[int] = None) -> Iterator[Tuple[int, "cv2.Mat"]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = max(0, int(start_frame))
    end = int(end_frame) if end_frame is not None else (total - 1)
    end = min(end, total - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    idx = start
    while idx <= end:
        ok, frame = cap.read()
        if not ok:
            break
        yield idx, frame
        idx += 1
    cap.release()
