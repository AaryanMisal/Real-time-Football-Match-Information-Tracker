from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from .analytics.ball import clean_ball_track
from .analytics.events import Event, infer_events_from_possession
from .analytics.possession import PossessionState, assign_possession
from .analytics.speed_distance import compute_kinematics, speed_zones
from .analytics.team import TeamAssigner
from .detection.yolo_detector import Detection, YOLODetector
from .geometry.points import bbox_center_xy, bbox_foot_xy, clip_point
from .io.video import get_video_meta, iter_frames
from .mapping.homography import Homography, load_or_compute_homography
from .motion.camera_motion import CameraMotionEstimator
from .schemas import FrameState, ObjState
from .tracking.bytetrack import ByteTrackWrapper
from .utils.config import ensure_dir
from .utils.time import run_id_now


@dataclass
class PipelineOutputs:
    run_dir: Path
    annotated_video_path: Optional[Path]
    frames_csv: Path
    players_summary_csv: Path
    teams_summary_csv: Path
    events_csv: Path
    heatmaps_players_csv: Optional[Path]
    ball_summary_csv: Path
    possession_thirds_csv: Optional[Path]


def _as_bool(x) -> bool:
    return bool(x) if x is not None else False


def _nms_detections(dets: List[Detection], iou_thr: float = 0.6) -> List[Detection]:
    if len(dets) <= 1:
        return dets

    boxes = np.array([d.xyxy for d in dets], dtype=np.float32)
    scores = np.array([d.conf for d in dets], dtype=np.float32)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
    order = scores.argsort()[::-1]

    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])

        w = np.maximum(0.0, xx2 - xx1 + 1.0)
        h = np.maximum(0.0, yy2 - yy1 + 1.0)
        inter = w * h
        iou = inter / (areas[i] + areas[rest] - inter + 1e-6)

        inds = np.where(iou <= float(iou_thr))[0]
        order = rest[inds]

    return [dets[i] for i in keep]


class FootballAnalyticsPipeline:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.meta = get_video_meta(cfg["run"]["input_video"])
        self.fps = float(self.meta.fps)
        self.dt = 1.0 / self.fps

        m = cfg["model"]
        self.detector = YOLODetector(
            weights=m["weights"],
            device=cfg["run"].get("device", "cpu"),
            conf=m.get("conf", 0.25),
            iou=m.get("iou", 0.45),
            imgsz=m.get("imgsz", 960),
        )

        self.class_ids = cfg["classes"]
        self.track_cfg = cfg["tracking"]["bytetrack"]

        frame_rate = self.track_cfg.get("frame_rate") or self.fps

        # Two trackers: players/keepers vs referees 
        self.tracker_players = ByteTrackWrapper(
            track_thresh=self.track_cfg.get("track_thresh", 0.25),
            track_buffer=self.track_cfg.get("track_buffer", 30),
            match_thresh=self.track_cfg.get("match_thresh", 0.8),
            frame_rate=frame_rate,
        )
        self.tracker_refs = ByteTrackWrapper(
            track_thresh=self.track_cfg.get("track_thresh", 0.25),
            track_buffer=self.track_cfg.get("track_buffer", 30),
            match_thresh=self.track_cfg.get("match_thresh", 0.8),
            frame_rate=frame_rate,
        )

        self.hmg: Optional[Homography] = None
        if _as_bool(cfg.get("mapping", {}).get("enable", False)):
            mp = cfg["mapping"]
            self.hmg = load_or_compute_homography(
                homography_matrix_path=mp.get("homography_matrix_path"),
                image_points_px=mp.get("image_points_px", []),
                pitch_points_m=mp.get("pitch_points_m", []),
                pitch_length_m=mp.get("pitch_length_m", 105.0),
                pitch_width_m=mp.get("pitch_width_m", 68.0),
            )

        self.motion_estimator: Optional[CameraMotionEstimator] = None

        tcfg = cfg["teams"]
        self.team_assigner = TeamAssigner(
            roi_rel=tcfg["jersey_roi"],
            kmeans_k=tcfg.get("kmeans_k", 2),
            color_space=tcfg.get("color_space", "HSV"),
            smooth_alpha=tcfg.get("smooth_alpha", 0.85),
        )
        self.team_model_ready = False

    def run(self) -> PipelineOutputs:
        run_cfg = self.cfg["run"]
        in_path = run_cfg["input_video"]
        start = int(run_cfg.get("start_frame", 0) or 0)
        end = run_cfg.get("end_frame", None)
        if end is not None:
            end = int(end)

        out_root = ensure_dir(run_cfg.get("output_dir", "outputs"))
        run_id = run_cfg.get("run_id") or run_id_now()
        run_dir = ensure_dir(out_root / run_id)

        # Video writer
        save_video = _as_bool(run_cfg.get("save_annotated_video", True)) and _as_bool(self.cfg.get("render", {}).get("enable", True))
        video_writer = None
        annotated_path = None
        if save_video:
            annotated_path = run_dir / "annotated.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                str(annotated_path),
                fourcc,
                float(self.fps),
                (self.meta.width, self.meta.height),
            )

        renderer = None
        if _as_bool(self.cfg.get("render", {}).get("enable", True)):
            from .render.overlays import OverlayRenderer, RenderConfig

            rcfg = self.cfg["render"]
            renderer = OverlayRenderer(
                RenderConfig(
                    show_ids=rcfg.get("show_ids", True),
                    show_speed=rcfg.get("show_speed", True),
                    show_distance=rcfg.get("show_distance", True),
                    show_team=rcfg.get("show_team", True),
                    show_possession_bar=rcfg.get("show_possession_bar", True),
                    show_ball_speed=rcfg.get("show_ball_speed", True),
                    thickness=rcfg.get("thickness", 2),
                    font_scale=rcfg.get("font_scale", 0.6),
                )
            )

        frame_states: List[FrameState] = []
        frame_indices: List[int] = []
        ball_p_world_raw: List[Optional[np.ndarray]] = []
        ball_p_img_mc_raw: List[Optional[np.ndarray]] = []

        team_samples: List[np.ndarray] = []
        sample_frames = int(self.cfg["teams"].get("sample_frames", 20))
        min_players_for_clustering = int(self.cfg["teams"].get("min_players_for_clustering", 6))

        possession_prev = PossessionState()

        prev_player_pos: Dict[int, np.ndarray] = {}
        prev_player_speed_ms: Dict[int, float] = {}
        cum_player_distance: Dict[int, float] = {}

        events: List[Event] = []

        frames_iter = list(iter_frames(in_path, start_frame=start, end_frame=end))
        if len(frames_iter) == 0:
            raise RuntimeError("No frames read. Check start/end frame and video path.")

        # Track confirmation 
        track_hits_players: Dict[int, int] = {}
        track_hits_refs: Dict[int, int] = {}
        min_hits = int(self.track_cfg.get("min_hits", 10)) 

        # Camera motion estimator
        if _as_bool(self.cfg.get("camera_motion", {}).get("enable", False)):
            cm = self.cfg["camera_motion"]
            self.motion_estimator = CameraMotionEstimator(
                first_frame_bgr=frames_iter[0][1],
                max_features=cm.get("max_features", 400),
                min_quality=cm.get("min_feature_quality", 0.01),
                min_distance=cm.get("min_feature_distance", 12),
                lk_win_size=cm.get("lk_win_size", 21),
                lk_max_level=cm.get("lk_max_level", 3),
                smooth_alpha=cm.get("smooth_alpha", 0.6),
                ransac=cm.get("ransac", False),
            )

        player_id = int(self.class_ids["player"])
        keeper_id = int(self.class_ids["goalkeeper"])
        ref_id = int(self.class_ids["referee"])
        ball_id = int(self.class_ids["ball"])

        # Pass 1: detect + track
        for local_i, (frame_idx, frame_bgr) in enumerate(tqdm(frames_iter, desc="Pass 1: detect/track")):
            t_s = (frame_idx - start) / float(self.fps)

            cam_dx = cam_dy = 0.0
            if self.motion_estimator is not None and local_i > 0:
                cm = self.motion_estimator.step(frame_bgr)
                cam_dx, cam_dy = cm.dx, cm.dy

            detections = self.detector.infer(frame_bgr)

            # Split dets by class for separate trackers
            det_players: List[Detection] = []
            det_refs: List[Detection] = []
            det_ball: List[Detection] = []

            for d in detections:
                if d.cls_id == ball_id:
                    det_ball.append(d)
                elif d.cls_id == keeper_id:
                    # unify keepers into player class for tracking
                    det_players.append(Detection(xyxy=d.xyxy, conf=d.conf, cls_id=player_id))
                elif d.cls_id == player_id:
                    det_players.append(d)
                elif d.cls_id == ref_id:
                    det_refs.append(d)

            # NMS to remove duplicate detections that cause duplicate track IDs
            det_players = _nms_detections(det_players, iou_thr=0.6)
            det_refs = _nms_detections(det_refs, iou_thr=0.6)

            tracking_enabled = _as_bool(self.cfg.get("tracking", {}).get("enable", True))
            tracked_players = self.tracker_players.update(det_players) if tracking_enabled else []
            tracked_refs = self.tracker_refs.update(det_refs) if tracking_enabled else []

            best_ball = max(det_ball, key=lambda x: x.conf) if len(det_ball) else None

            fs = FrameState(frame_idx=frame_idx, timestamp_s=float(t_s), cam_dx=float(cam_dx), cam_dy=float(cam_dy))

            # players
            for td in tracked_players:
                tid = int(td.track_id)
                track_hits_players[tid] = track_hits_players.get(tid, 0) + 1
                if track_hits_players[tid] < min_hits:
                    continue

                p_img = bbox_foot_xy(td.xyxy)
                p_img = clip_point(p_img, self.meta.width, self.meta.height)
                p_mc = self.motion_estimator.compensate_point(p_img) if self.motion_estimator is not None else p_img
                p_world = self.hmg.img_to_world(p_mc) if self.hmg is not None else None

                o = ObjState(
                    track_id=tid,
                    cls="player",
                    conf=float(td.conf),
                    bbox_xyxy=td.xyxy.astype(np.float32),
                    p_img=p_img.astype(np.float32),
                    p_img_mc=p_mc.astype(np.float32),
                    p_world=p_world.astype(np.float32) if p_world is not None else None,
                )

                fs.players[o.track_id] = o

                if _as_bool(self.cfg["teams"].get("enable", True)) and local_i < sample_frames:
                    c = self.team_assigner.collect_sample(frame_bgr, o.bbox_xyxy)
                    if c is not None:
                        team_samples.append(c)

            # referees
            for td in tracked_refs:
                tid = int(td.track_id)
                track_hits_refs[tid] = track_hits_refs.get(tid, 0) + 1
                if track_hits_refs[tid] < min_hits:
                    continue

                p_img = bbox_foot_xy(td.xyxy)
                p_img = clip_point(p_img, self.meta.width, self.meta.height)
                p_mc = self.motion_estimator.compensate_point(p_img) if self.motion_estimator is not None else p_img
                p_world = self.hmg.img_to_world(p_mc) if self.hmg is not None else None

                o = ObjState(
                    track_id=tid,
                    cls="referee",
                    conf=float(td.conf),
                    bbox_xyxy=td.xyxy.astype(np.float32),
                    p_img=p_img.astype(np.float32),
                    p_img_mc=p_mc.astype(np.float32),
                    p_world=p_world.astype(np.float32) if p_world is not None else None,
                )
                fs.referees[o.track_id] = o

            if best_ball is not None:
                bp = bbox_center_xy(best_ball.xyxy)
                bp = clip_point(bp, self.meta.width, self.meta.height)
                bp_mc = self.motion_estimator.compensate_point(bp) if self.motion_estimator is not None else bp
                bp_w = self.hmg.img_to_world(bp_mc) if self.hmg is not None else None
                fs.ball_xyxy = best_ball.xyxy.astype(np.float32)
                fs.ball_conf = float(best_ball.conf)
                fs.ball_p_img = bp.astype(np.float32)
                fs.ball_p_img_mc = bp_mc.astype(np.float32)
                fs.ball_p_world = bp_w.astype(np.float32) if bp_w is not None else None

            frame_states.append(fs)
            frame_indices.append(frame_idx)
            ball_p_world_raw.append(fs.ball_p_world)
            ball_p_img_mc_raw.append(fs.ball_p_img_mc)

        # Fit team colors
        if _as_bool(self.cfg["teams"].get("enable", True)) and len(team_samples) >= min_players_for_clustering:
            self.team_assigner.fit(team_samples)
            self.team_model_ready = True

        # Clean ball track
        ball_cfg = self.cfg.get("ball", {})
        use_world = _as_bool(ball_cfg.get("world_use", True)) and (self.hmg is not None)

        ball_series = clean_ball_track(
            frame_indices_all=np.array(frame_indices, dtype=np.int32),
            p_list=ball_p_world_raw if use_world else ball_p_img_mc_raw,
            fps=float(self.fps),
            max_gap_frames=int(ball_cfg.get("max_gap_frames_to_interpolate", 12)),
            smoothing_window=int(ball_cfg.get("smoothing_window", 7)),
            outlier_max_speed_kmh=float(ball_cfg.get("outlier_max_speed_kmh", 120.0)),
        )

        for i, fs in enumerate(frame_states):
            bp = ball_series.p[i]
            if np.any(np.isnan(bp)):
                continue
            if use_world:
                fs.ball_p_world = bp.astype(np.float32)
            else:
                fs.ball_p_img_mc = bp.astype(np.float32)
                if self.hmg is not None:
                    fs.ball_p_world = self.hmg.img_to_world(fs.ball_p_img_mc).astype(np.float32)
            fs.ball_speed_kmh = float(ball_series.speed_kmh[i])
            fs.ball_distance_m = float(ball_series.cumulative_dist_m[i])

        # Config for analytics
        analytics_cfg = self.cfg.get("analytics", {})
        zones_cfg = analytics_cfg.get("zones", {"jog_kmh": 10.0, "high_intensity_kmh": 18.0, "sprint_kmh": 25.0})
        speed_cfg = analytics_cfg.get("speed", {"min_dt_s": 0.04})
        possession_cfg = self.cfg.get("possession", {})

        # Extra analytics accumulators
        player_pos_frames: Dict[int, int] = {}
        team_pos_frames: Dict[int, int] = {}

        # Ball thirds (requires world mapping)
        thirds_enable = _as_bool(analytics_cfg.get("thirds", {}).get("enable", False)) and (self.hmg is not None)
        thirds_rows: List[dict] = []
        pitch_len = float(self.hmg.pitch_length_m) if self.hmg is not None else 105.0
        third_edges = (pitch_len / 3.0, 2.0 * pitch_len / 3.0)

        # Heatmaps (requires world mapping)
        heat_cfg = analytics_cfg.get("heatmap", {})
        heat_enable = _as_bool(heat_cfg.get("enable", False)) and (self.hmg is not None)
        heat_grid_x = int(heat_cfg.get("grid_x", 12))
        heat_grid_y = int(heat_cfg.get("grid_y", 8))
        player_heat: Dict[int, np.ndarray] = {}

        prev_team = None
        prev_player = None

        # analytics + render
        for i, fs in enumerate(tqdm(frame_states, desc="Pass 2: analytics/render")):
            frame_bgr = frames_iter[i][1]

            # Team assignment per player
            for pid, p in fs.players.items():
                if self.team_model_ready and _as_bool(self.cfg["teams"].get("enable", True)):
                    c = self.team_assigner.collect_sample(frame_bgr, p.bbox_xyxy)
                    if c is not None:
                        tid, conf = self.team_assigner.update_player_team(pid, c)
                        p.team_id = tid
                        p.team_conf = conf

            # Kinematics
            for pid, p in fs.players.items():
                curr = p.p_world if p.p_world is not None else p.p_img_mc
                prev = prev_player_pos.get(pid, None)

                dt_eff = max(self.dt, float(speed_cfg.get("min_dt_s", 0.04)))
                kin = compute_kinematics(prev, prev_player_speed_ms.get(pid, None), curr, dt_s=dt_eff)

                if kin is not None:
                    p.speed_kmh = kin.speed_kmh
                    p.accel_ms2 = kin.accel_ms2
                    cum_player_distance[pid] = cum_player_distance.get(pid, 0.0) + kin.distance_m
                    p.distance_m = cum_player_distance[pid]
                    prev_player_speed_ms[pid] = kin.speed_kmh / 3.6

                if curr is not None:
                    prev_player_pos[pid] = curr

            # Possession assignment
            ball_p = fs.ball_p_world if (_as_bool(possession_cfg.get("use_world_distance", True)) and fs.ball_p_world is not None) else fs.ball_p_img_mc

            players_pos: Dict[int, np.ndarray] = {}
            teams_map: Dict[int, int] = {}
            for pid, p in fs.players.items():
                pp = p.p_world if ball_p is fs.ball_p_world else p.p_img_mc
                if pp is not None:
                    players_pos[pid] = pp.astype(np.float32)
                    if p.team_id is not None:
                        teams_map[pid] = int(p.team_id)

            possession_prev = assign_possession(
                ball_p=ball_p.astype(np.float32) if ball_p is not None else None,
                players=players_pos,
                player_team=teams_map,
                prev=possession_prev,
                max_ball_to_player_m=float(possession_cfg.get("max_ball_to_player_m", 2.2)),
                hysteresis_m=float(possession_cfg.get("hysteresis_m", 0.75)),
                hold_frames=int(possession_cfg.get("hold_frames", 6)),
            )

            fs.team_in_possession = possession_prev.team_id
            fs.player_in_possession = possession_prev.player_id

            # Mark has_ball
            if fs.player_in_possession is not None and fs.player_in_possession in fs.players:
                fs.players[fs.player_in_possession].has_ball = True

            # Possession time counters
            if fs.player_in_possession is not None:
                player_pos_frames[fs.player_in_possession] = player_pos_frames.get(fs.player_in_possession, 0) + 1
            if fs.team_in_possession is not None:
                team_pos_frames[fs.team_in_possession] = team_pos_frames.get(fs.team_in_possession, 0) + 1

            # Ball thirds
            if thirds_enable and fs.ball_p_world is not None:
                bx = float(fs.ball_p_world[0])
                if bx < third_edges[0]:
                    third = "defensive"
                elif bx < third_edges[1]:
                    third = "middle"
                else:
                    third = "attacking"
                thirds_rows.append(
                    {
                        "frame": fs.frame_idx,
                        "t_s": fs.timestamp_s,
                        "team_in_possession": fs.team_in_possession,
                        "ball_third": third,
                    }
                )

            # Heatmap update
            if heat_enable and fs.players:
                for pid, p in fs.players.items():
                    if p.p_world is None:
                        continue
                    if pid not in player_heat:
                        player_heat[pid] = np.zeros((heat_grid_y, heat_grid_x), dtype=np.int32)

                    X, Y = float(p.p_world[0]), float(p.p_world[1])
                    X = max(0.0, min(self.hmg.pitch_length_m, X))
                    Y = max(0.0, min(self.hmg.pitch_width_m, Y))

                    gx = min(heat_grid_x - 1, int((X / self.hmg.pitch_length_m) * heat_grid_x))
                    gy = min(heat_grid_y - 1, int((Y / self.hmg.pitch_width_m) * heat_grid_y))
                    player_heat[pid][gy, gx] += 1

            # Events
            events.extend(
                infer_events_from_possession(
                    fs.frame_idx,
                    fs.timestamp_s,
                    prev_team,
                    prev_player,
                    fs.team_in_possession,
                    fs.player_in_possession,
                )
            )
            prev_team = fs.team_in_possession
            prev_player = fs.player_in_possession

            # Render
            if video_writer is not None and renderer is not None:
                out_frame = renderer.draw(frame_bgr.copy(), fs)
                video_writer.write(out_frame)

        if video_writer is not None:
            video_writer.release()

        # Export CSVs
        frames_csv = run_dir / "frames.csv"
        players_summary_csv = run_dir / "players_summary.csv"
        teams_summary_csv = run_dir / "teams_summary.csv"
        events_csv = run_dir / "events.csv"
        ball_summary_csv = run_dir / "ball_summary.csv"

        # per-frame rows
        rows: List[dict] = []
        for fs in frame_states:
            for pid, p in fs.players.items():
                rows.append(
                    {
                        "frame": fs.frame_idx,
                        "t_s": fs.timestamp_s,
                        "type": "player",
                        "id": pid,
                        "team": p.team_id,
                        "team_conf": p.team_conf,
                        "has_ball": int(p.has_ball),
                        "x_img": None if p.p_img is None else float(p.p_img[0]),
                        "y_img": None if p.p_img is None else float(p.p_img[1]),
                        "x_world": None if p.p_world is None else float(p.p_world[0]),
                        "y_world": None if p.p_world is None else float(p.p_world[1]),
                        "speed_kmh": p.speed_kmh,
                        "accel_ms2": p.accel_ms2,
                        "distance_m": p.distance_m,
                        "team_in_possession": fs.team_in_possession,
                        "player_in_possession": fs.player_in_possession,
                    }
                )

            # ball row
            if fs.ball_p_world is not None or fs.ball_p_img is not None:
                rows.append(
                    {
                        "frame": fs.frame_idx,
                        "t_s": fs.timestamp_s,
                        "type": "ball",
                        "id": -1,
                        "team": fs.team_in_possession,
                        "team_conf": None,
                        "has_ball": None,
                        "x_img": None if fs.ball_p_img is None else float(fs.ball_p_img[0]),
                        "y_img": None if fs.ball_p_img is None else float(fs.ball_p_img[1]),
                        "x_world": None if fs.ball_p_world is None else float(fs.ball_p_world[0]),
                        "y_world": None if fs.ball_p_world is None else float(fs.ball_p_world[1]),
                        "speed_kmh": fs.ball_speed_kmh,
                        "accel_ms2": None,
                        "distance_m": fs.ball_distance_m,
                        "team_in_possession": fs.team_in_possession,
                        "player_in_possession": fs.player_in_possession,
                    }
                )

        df_frames = pd.DataFrame(rows)
        df_frames.to_csv(frames_csv, index=False)

        # Player summary + team summary
        heatmaps_players_csv: Optional[Path] = None
        possession_thirds_csv: Optional[Path] = None

        if len(df_frames) > 0:
            df_players = df_frames[df_frames["type"] == "player"].copy()

            agg = (
                df_players.groupby("id")
                .agg(
                    frames_seen=("frame", "count"),
                    team=("team", lambda x: int(x.dropna().mode().iloc[0]) if len(x.dropna()) else None),
                    total_distance_m=("distance_m", "max"),
                    avg_speed_kmh=("speed_kmh", "mean"),
                    max_speed_kmh=("speed_kmh", "max"),
                    avg_accel_ms2=("accel_ms2", "mean"),
                    touches=("has_ball", "sum"),
                )
                .reset_index()
                .rename(columns={"id": "player_id"})
            )

            df_players["speed_zone"] = df_players["speed_kmh"].fillna(0.0).apply(
                lambda s: speed_zones(
                    float(s),
                    float(self.cfg.get("analytics", {}).get("zones", {}).get("jog_kmh", 10.0)),
                    float(self.cfg.get("analytics", {}).get("zones", {}).get("high_intensity_kmh", 18.0)),
                    float(self.cfg.get("analytics", {}).get("zones", {}).get("sprint_kmh", 25.0)),
                )
            )
            zone_dist = df_players.groupby(["id", "speed_zone"])["distance_m"].max().unstack(fill_value=0.0)
            zone_dist = zone_dist.reset_index().rename(columns={"id": "player_id"})

            summary = agg.merge(zone_dist, on="player_id", how="left")
            summary.to_csv(players_summary_csv, index=False)

            # team possession %
            pos = df_frames[df_frames["type"] == "ball"][["frame", "team_in_possession"]].dropna()
            total_pos_frames = len(pos)
            team_pos = pos["team_in_possession"].value_counts().to_dict()
            team_rows = []
            for tid, cnt in team_pos.items():
                team_rows.append(
                    {
                        "team_id": int(tid),
                        "possession_frames": int(cnt),
                        "possession_pct": float(cnt) / float(total_pos_frames + 1e-6) * 100.0,
                    }
                )
            pd.DataFrame(team_rows).sort_values("team_id").to_csv(teams_summary_csv, index=False)
        else:
            pd.DataFrame(columns=["player_id"]).to_csv(players_summary_csv, index=False)
            pd.DataFrame(columns=["team_id"]).to_csv(teams_summary_csv, index=False)

        # events.csv
        pd.DataFrame([e.__dict__ for e in events]).to_csv(events_csv, index=False)

        # ball_summary.csv
        pd.DataFrame(
            {
                "frame": ball_series.frame_idx,
                "x": ball_series.p[:, 0],
                "y": ball_series.p[:, 1],
                "speed_kmh": ball_series.speed_kmh,
                "cumulative_dist_m": ball_series.cumulative_dist_m,
            }
        ).to_csv(ball_summary_csv, index=False)

        # Extra exports + enrich summaries
        possession_thirds_csv = run_dir / "possession_by_third.csv" if thirds_enable and len(thirds_rows) > 0 else None
        if possession_thirds_csv is not None:
            pd.DataFrame(thirds_rows).to_csv(possession_thirds_csv, index=False)

        # Heatmaps export
        heatmaps_players_csv = run_dir / "heatmaps_players.csv" if heat_enable else None
        if heat_enable and heatmaps_players_csv is not None:
            hm_rows = []
            for pid, grid in player_heat.items():
                for gy in range(grid.shape[0]):
                    for gx in range(grid.shape[1]):
                        hm_rows.append({"player_id": int(pid), "gy": int(gy), "gx": int(gx), "count": int(grid[gy, gx])})
            pd.DataFrame(hm_rows).to_csv(heatmaps_players_csv, index=False)

        # Enrich players_summary.csv + teams_summary.csv with possession time, pass/turnover counts, sprint counts
        if players_summary_csv.exists():
            try:
                df_sum = pd.read_csv(players_summary_csv)
                df_events = pd.DataFrame([e.__dict__ for e in events]) if len(events) else pd.DataFrame(
                    columns=["type", "team_id", "from_player", "to_player"]
                )

                # possession time (seconds)
                player_pos_frames: Dict[int, int] = {}  # kept for backward compatibility if you rely on this later

                # This block expects player_pos_frames & df_frames in scope; they are.
                # possession time
                pos_time = {pid: cnt * self.dt for pid, cnt in locals().get("player_pos_frames", {}).items()}
                df_sum["possession_time_s"] = df_sum["player_id"].apply(lambda pid: float(pos_time.get(int(pid), 0.0)))

                # passes/turnovers per player
                passes = df_events[df_events["type"] == "pass"] if len(df_events) else df_events
                turnovers = df_events[df_events["type"] == "turnover"] if len(df_events) else df_events

                passes_made = passes["from_player"].value_counts().to_dict() if len(passes) else {}
                passes_received = passes["to_player"].value_counts().to_dict() if len(passes) else {}
                turnovers_won = turnovers["to_player"].value_counts().to_dict() if len(turnovers) else {}
                turnovers_lost = turnovers["from_player"].value_counts().to_dict() if len(turnovers) else {}

                df_sum["passes_made"] = df_sum["player_id"].apply(lambda pid: int(passes_made.get(int(pid), 0)))
                df_sum["passes_received"] = df_sum["player_id"].apply(lambda pid: int(passes_received.get(int(pid), 0)))
                df_sum["turnovers_won"] = df_sum["player_id"].apply(lambda pid: int(turnovers_won.get(int(pid), 0)))
                df_sum["turnovers_lost"] = df_sum["player_id"].apply(lambda pid: int(turnovers_lost.get(int(pid), 0)))

                # sprint count (>= sprint threshold for >=5 consecutive frames)
                sprint_thr = float(zones_cfg.get("sprint_kmh", 25.0))
                min_sprint_frames = 5
                sprint_counts: Dict[int, int] = {}

                dfp = df_frames[df_frames["type"] == "player"][["id", "frame", "speed_kmh"]].copy() if "df_frames" in locals() else pd.DataFrame()
                if len(dfp):
                    for pid, g in dfp.groupby("id"):
                        s = g.sort_values("frame")["speed_kmh"].fillna(0.0).to_numpy()
                        cnt = 0
                        run = 0
                        for v in s:
                            if float(v) >= sprint_thr:
                                run += 1
                            else:
                                if run >= min_sprint_frames:
                                    cnt += 1
                                run = 0
                        if run >= min_sprint_frames:
                            cnt += 1
                        sprint_counts[int(pid)] = int(cnt)

                df_sum["sprint_count"] = df_sum["player_id"].apply(lambda pid: int(sprint_counts.get(int(pid), 0)))
                df_sum.to_csv(players_summary_csv, index=False)
            except Exception:
                # Keep pipeline running even if enrichment fails
                pass

        if teams_summary_csv.exists():
            try:
                df_team = pd.read_csv(teams_summary_csv)
                df_events = pd.DataFrame([e.__dict__ for e in events]) if len(events) else pd.DataFrame(
                    columns=["type", "team_id", "from_player", "to_player"]
                )

                team_pos_time = {tid: cnt * self.dt for tid, cnt in locals().get("team_pos_frames", {}).items()}
                if len(df_team):
                    df_team["possession_time_s"] = df_team["team_id"].apply(lambda tid: float(team_pos_time.get(int(tid), 0.0)))
                    if len(df_events):
                        df_team["passes"] = df_team["team_id"].apply(
                            lambda tid: int((df_events["type"].eq("pass") & df_events["team_id"].eq(int(tid))).sum())
                        )
                        df_team["turnovers"] = df_team["team_id"].apply(
                            lambda tid: int((df_events["type"].eq("turnover") & df_events["team_id"].eq(int(tid))).sum())
                        )
                    df_team.to_csv(teams_summary_csv, index=False)
            except Exception:
                pass

        return PipelineOutputs(
            run_dir=run_dir,
            annotated_video_path=annotated_path,
            frames_csv=frames_csv,
            players_summary_csv=players_summary_csv,
            teams_summary_csv=teams_summary_csv,
            events_csv=events_csv,
            heatmaps_players_csv=heatmaps_players_csv,
            ball_summary_csv=ball_summary_csv,
            possession_thirds_csv=possession_thirds_csv,
        )
