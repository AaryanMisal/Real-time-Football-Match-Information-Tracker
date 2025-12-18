from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import inspect
import numpy as np
import supervision as sv

from ..detection.yolo_detector import Detection


@dataclass
class TrackDet:
    xyxy: np.ndarray
    conf: float
    cls_id: int
    track_id: int


class ByteTrackWrapper:
    def __init__(
        self,
        track_thresh: float = 0.25,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        frame_rate: Optional[float] = None,
    ):
        # Build kwargs only if supported by the installed supervision version
        desired_kwargs = dict(
            track_thresh=float(track_thresh),
            track_buffer=int(track_buffer),
            match_thresh=float(match_thresh),
            frame_rate=float(frame_rate) if frame_rate is not None else 30.0,
        )

        try:
            sig = inspect.signature(sv.ByteTrack)
            supported = set(sig.parameters.keys())
            filtered_kwargs = {k: v for k, v in desired_kwargs.items() if k in supported}

            if len(filtered_kwargs) == 0 and len(supported) <= 1:
                raise RuntimeError(
                    "Your installed supervision.ByteTrack does not accept tracking kwargs "
                    "(track_thresh/track_buffer/match_thresh/frame_rate). "
                    "Pin/upgrade supervision to a version that supports these args."
                )

            self.tracker = sv.ByteTrack(**filtered_kwargs) if len(filtered_kwargs) else sv.ByteTrack()

        except Exception as e:
            raise RuntimeError(f"Failed to construct ByteTrack with your supervision version: {e}")

    def update(self, detections: List[Detection]) -> List[TrackDet]:
        if len(detections) == 0:
            # Some supervision versions have Detections.empty() keep it safe
            dets = sv.Detections(xyxy=np.zeros((0, 4), dtype=np.float32),
                                 confidence=np.zeros((0,), dtype=np.float32),
                                 class_id=np.zeros((0,), dtype=np.int32))
        else:
            xyxy = np.stack([d.xyxy for d in detections], axis=0).astype(np.float32)
            conf = np.array([d.conf for d in detections], dtype=np.float32)
            cls_id = np.array([d.cls_id for d in detections], dtype=np.int32)
            dets = sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls_id)

        tracked = self.tracker.update_with_detections(dets)
        if tracked is None or len(tracked) == 0:
            return []

        # tracker id attribute name differs across versions
        tracker_ids = getattr(tracked, "tracker_id", None)
        if tracker_ids is None:
            tracker_ids = getattr(tracked, "tracker_ids", None)
        if tracker_ids is None:
            return []

        out: List[TrackDet] = []
        for xyxy, conf, cls_id, tid in zip(tracked.xyxy, tracked.confidence, tracked.class_id, tracker_ids):
            out.append(
                TrackDet(
                    xyxy=np.array(xyxy, dtype=np.float32),
                    conf=float(conf),
                    cls_id=int(cls_id),
                    track_id=int(tid),
                )
            )
        return out
