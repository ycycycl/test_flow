#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

try:
    import imageio.v2 as imageio
except ModuleNotFoundError:
    imageio = None

EPS = 1e-12
HEADER_H = 176

METRIC_COLUMNS = {
    "score": "score",
    "collision": "no_ego_at_fault_collisions",
    "offroad": "drivable_area_compliance",
    "ttc": "time_to_collision_within_bound",
    "progress_gate": "ego_is_making_progress",
    "progress_route": "ego_progress_along_expert_route",
    "comfort": "ego_is_comfortable",
    "speed_limit": "speed_limit_compliance",
    "direction": "driving_direction_compliance",
}

SHORT_METRICS = [
    ("score", "score"),
    ("coll", "collision"),
    ("off", "offroad"),
    ("ttc", "ttc"),
    ("prog", "progress"),
    ("comfort", "comfort"),
    ("speed", "speed_limit"),
]


def require_cv2() -> Any:
    if cv2 is None:
        raise ModuleNotFoundError("OpenCV is required for rendering. Install opencv-python or run in the nuPlan environment.")
    return cv2


def require_imageio() -> Any:
    if imageio is None:
        raise ModuleNotFoundError("imageio is required for video writing. Install imageio or run in the nuPlan environment.")
    return imageio


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return None if math.isnan(out) else out
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def safe_filename(value: str, max_len: int = 120) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)[:max_len]


def float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if math.isnan(out):
        return None
    return out


def fmt_metric(value: Any, digits: int = 3) -> str:
    val = float_or_none(value)
    if val is None:
        return "NA"
    if abs(val - round(val)) < 1e-9 and 0.0 <= val <= 1.0:
        return str(int(round(val)))
    return f"{val:.{digits}f}"


def pct_metric(value: Any) -> str:
    val = float_or_none(value)
    if val is None:
        return "NA"
    return f"{100.0 * val:.0f}"


def load_simulation_support() -> tuple[Any, Any, Any, Any]:
    import torch
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from nuplan.planning.simulation.simulation_log import SimulationLog

    return torch, SimulationLog, SemanticMapLayer, Point2D


def load_simulation_log_cpu_safe(path: Path) -> Any:
    torch, SimulationLog, _, _ = load_simulation_support()
    original = torch.storage._load_from_bytes

    def _cpu_load(buffer: bytes) -> Any:
        return torch.load(io.BytesIO(buffer), map_location="cpu")

    torch.storage._load_from_bytes = _cpu_load
    try:
        return SimulationLog.load_data(path)
    finally:
        torch.storage._load_from_bytes = original


def state_xy_heading(state: Any) -> tuple[float, float, float]:
    rear = state.rear_axle
    return float(rear.x), float(rear.y), float(rear.heading)


def trajectory_to_world_xy(trajectory: Any, max_points: int = 80) -> np.ndarray:
    sampled = trajectory.get_sampled_trajectory()
    points = []
    for state in sampled:
        x, y, _ = state_xy_heading(state)
        points.append((x, y))
    if not points:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(points[:max_points], dtype=np.float64)


def world_to_local_xy(world_xy: Any, ego_state: Any) -> np.ndarray:
    arr = np.asarray(world_xy, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    arr = arr.reshape(-1, arr.shape[-1])[:, :2]
    origin = np.asarray(ego_state.rear_axle.array[:2], dtype=np.float64)
    angle = float(ego_state.rear_axle.heading)
    rot = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float64,
    )
    return np.matmul(arr - origin, rot)


def local_to_pixel(local_xy: Any, width: int, height: int, bounds: float, offset: float) -> np.ndarray:
    arr = np.asarray(local_xy, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    arr = arr.reshape(-1, arr.shape[-1])[:, :2]
    xmin, xmax = -bounds + offset, bounds + offset
    ymin, ymax = -bounds, bounds
    px = (arr[:, 0] - xmin) / max(EPS, xmax - xmin) * (width - 1)
    py = (ymax - arr[:, 1]) / max(EPS, ymax - ymin) * (height - 1)
    pix = np.stack([px, py], axis=1)
    pix = np.clip(pix, -100000, 100000)
    return np.rint(pix).astype(np.int32)


def world_to_pixel(world_xy: Any, ego_state: Any, width: int, height: int, bounds: float, offset: float) -> np.ndarray:
    return local_to_pixel(world_to_local_xy(world_xy, ego_state), width, height, bounds, offset)


def geometry_xy(geometry: Any) -> np.ndarray:
    if geometry is None:
        return np.zeros((0, 2), dtype=np.float64)
    try:
        coords = np.asarray(geometry.exterior.coords, dtype=np.float64)
    except Exception:
        try:
            xy = geometry.exterior.xy
            coords = np.stack([np.asarray(xy[0], dtype=np.float64), np.asarray(xy[1], dtype=np.float64)], axis=1)
        except Exception:
            return np.zeros((0, 2), dtype=np.float64)
    if coords.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return coords.reshape(-1, coords.shape[-1])[:, :2]


def draw_polyline(img: np.ndarray, points: np.ndarray, color: tuple[int, int, int], thickness: int = 1, closed: bool = False) -> None:
    if points is None or len(points) < (3 if closed else 2):
        return
    cv2_mod = require_cv2()
    cv2_mod.polylines(img, [points.reshape(-1, 1, 2)], closed, color, thickness, cv2_mod.LINE_AA)


def draw_polygon(img: np.ndarray, points: np.ndarray, fill: tuple[int, int, int] | None, outline: tuple[int, int, int], thickness: int = 1) -> None:
    if points is None or len(points) < 3:
        return
    cv2_mod = require_cv2()
    pts = points.reshape(-1, 1, 2)
    if fill is not None:
        cv2_mod.fillPoly(img, [pts], fill, lineType=cv2_mod.LINE_AA)
    cv2_mod.polylines(img, [pts], True, outline, thickness, cv2_mod.LINE_AA)


def simulation_history_xy(log: Any) -> np.ndarray:
    points = []
    for sample in log.simulation_history.data:
        x, y, _ = state_xy_heading(sample.ego_state)
        points.append((x, y))
    if not points:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(points, dtype=np.float64)


def collect_map_features(log: Any, bounds: float, offset: float, step: int) -> list[dict[str, Any]]:
    _, _, SemanticMapLayer, Point2D = load_simulation_support()
    samples = log.simulation_history.data
    if not samples:
        return []
    map_api = log.scenario.map_api
    layers = [
        SemanticMapLayer.LANE,
        SemanticMapLayer.LANE_CONNECTOR,
        SemanticMapLayer.INTERSECTION,
        SemanticMapLayer.CARPARK_AREA,
        SemanticMapLayer.CROSSWALK,
    ]
    route_roadblock_ids = {str(x) for x in (log.scenario.get_route_roadblock_ids() or [])}
    indices = list(range(0, len(samples), max(1, int(step))))
    if indices[-1] != len(samples) - 1:
        indices.append(len(samples) - 1)

    features: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for sample_idx in indices:
        ego_state = samples[sample_idx].ego_state
        query_point = getattr(getattr(ego_state, "center", None), "point", None)
        if query_point is None:
            query_point = Point2D(float(ego_state.rear_axle.x), float(ego_state.rear_axle.y))
        try:
            objects_by_layer = map_api.get_proximal_map_objects(query_point, float(bounds) + float(offset), layers)
        except Exception:
            continue
        for layer in layers:
            layer_name = getattr(layer, "name", str(layer))
            for obj in objects_by_layer.get(layer, []):
                obj_id = str(getattr(obj, "id", id(obj)))
                key = (layer_name, obj_id)
                if key in seen:
                    continue
                polygon = geometry_xy(getattr(obj, "polygon", None))
                if len(polygon) < 3:
                    continue
                roadblock_id = ""
                try:
                    roadblock_id = str(obj.get_roadblock_id())
                except Exception:
                    pass
                baseline = np.zeros((0, 2), dtype=np.float64)
                try:
                    baseline = np.asarray([[float(p.x), float(p.y)] for p in obj.baseline_path.discrete_path], dtype=np.float64)
                except Exception:
                    pass
                features.append(
                    {
                        "layer": layer_name,
                        "id": obj_id,
                        "roadblock_id": roadblock_id,
                        "is_route": bool(roadblock_id and roadblock_id in route_roadblock_ids),
                        "polygon": polygon,
                        "baseline": baseline,
                    }
                )
                seen.add(key)
    return features


def object_type_name(track: Any) -> str:
    obj_type = getattr(track, "tracked_object_type", None)
    return getattr(obj_type, "name", str(obj_type))


def put_line(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    scale: float = 0.52,
    color: tuple[int, int, int] = (25, 25, 25),
    thickness: int = 1,
) -> None:
    cv2_mod = require_cv2()
    cv2_mod.putText(img, text, (x, y), cv2_mod.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2_mod.LINE_AA)


def draw_map_features(img: np.ndarray, features: list[dict[str, Any]], ego_state: Any, bounds: float, offset: float) -> None:
    h, w = img.shape[:2]
    for feature in features:
        pts = world_to_pixel(feature["polygon"], ego_state, w, h, bounds, offset)
        layer = feature.get("layer", "")
        is_route = bool(feature.get("is_route"))
        if layer == "CROSSWALK":
            draw_polygon(img, pts, fill=(224, 224, 224), outline=(190, 190, 190), thickness=1)
            continue
        if layer == "INTERSECTION":
            draw_polygon(img, pts, fill=(229, 232, 235), outline=(204, 209, 214), thickness=1)
            continue
        fill = (218, 236, 255) if is_route else (235, 237, 239)
        outline = (125, 169, 210) if is_route else (210, 214, 218)
        draw_polygon(img, pts, fill=fill, outline=outline, thickness=1)

    for feature in features:
        baseline = feature.get("baseline")
        if baseline is None or len(baseline) < 2:
            continue
        pts = world_to_pixel(baseline, ego_state, w, h, bounds, offset)
        color = (40, 115, 190) if feature.get("is_route") else (150, 155, 160)
        thickness = 2 if feature.get("is_route") else 1
        draw_polyline(img, pts, color=color, thickness=thickness)


def draw_agents(img: np.ndarray, sample: Any, bounds: float, offset: float) -> None:
    ego_state = sample.ego_state
    h, w = img.shape[:2]
    colors = {
        "VEHICLE": (25, 80, 210),
        "PEDESTRIAN": (150, 55, 190),
        "BICYCLE": (215, 55, 95),
    }
    tracks = getattr(getattr(sample.observation, "tracked_objects", None), "tracked_objects", [])
    for track in tracks:
        color = colors.get(object_type_name(track), (60, 60, 60))
        poly = geometry_xy(getattr(getattr(track, "box", None), "geometry", None))
        if len(poly) >= 3:
            pts = world_to_pixel(poly, ego_state, w, h, bounds, offset)
            draw_polygon(img, pts, fill=None, outline=color, thickness=2)
        try:
            center = np.asarray(track.center.array, dtype=np.float64).reshape(-1)[:2]
            heading = float(track.center.heading)
            length = float(getattr(track.box, "length", 4.5))
            front = center + np.asarray([np.cos(heading), np.sin(heading)], dtype=np.float64) * max(1.0, length * 0.55)
            line = world_to_pixel(np.stack([center, front], axis=0), ego_state, w, h, bounds, offset)
            draw_polyline(img, line, color=color, thickness=1)
        except Exception:
            pass


def draw_ego(img: np.ndarray, ego_state: Any, bounds: float, offset: float) -> None:
    h, w = img.shape[:2]
    poly = geometry_xy(getattr(getattr(ego_state, "car_footprint", None), "geometry", None))
    if len(poly) >= 3:
        pts = world_to_pixel(poly, ego_state, w, h, bounds, offset)
        draw_polygon(img, pts, fill=(255, 238, 220), outline=(225, 95, 35), thickness=3)
    arrow = local_to_pixel(np.asarray([[0.0, 0.0], [5.8, 0.0]], dtype=np.float64), w, h, bounds, offset)
    if len(arrow) == 2:
        cv2_mod = require_cv2()
        cv2_mod.arrowedLine(img, tuple(arrow[0]), tuple(arrow[1]), (225, 95, 35), 3, cv2_mod.LINE_AA, tipLength=0.25)


def draw_trajectory(img: np.ndarray, world_xy: np.ndarray, ego_state: Any, bounds: float, offset: float, color: tuple[int, int, int], thickness: int) -> None:
    if world_xy is None or len(world_xy) < 2:
        return
    h, w = img.shape[:2]
    pts = world_to_pixel(world_xy, ego_state, w, h, bounds, offset)
    draw_polyline(img, pts, color=color, thickness=thickness)


def render_fast_panel(
    log: Any,
    map_features: list[dict[str, Any]],
    history_xy: np.ndarray,
    sample_idx: int,
    panel_size: int = 720,
    bounds: float = 60.0,
    offset: float = 20.0,
) -> np.ndarray:
    samples = log.simulation_history.data
    sample_idx = min(max(0, int(sample_idx)), len(samples) - 1)
    sample = samples[sample_idx]
    ego_state = sample.ego_state
    img = np.full((int(panel_size), int(panel_size), 3), 248, dtype=np.uint8)

    draw_map_features(img, map_features, ego_state, bounds, offset)
    draw_trajectory(img, history_xy[: sample_idx + 1], ego_state, bounds, offset, color=(220, 120, 45), thickness=3)
    if getattr(sample, "trajectory", None) is not None:
        draw_trajectory(img, trajectory_to_world_xy(sample.trajectory), ego_state, bounds, offset, color=(210, 40, 40), thickness=3)
    draw_agents(img, sample, bounds, offset)
    draw_ego(img, ego_state, bounds, offset)
    sim_iter = getattr(getattr(sample, "iteration", None), "index", sample_idx)
    put_line(img, f"iter {sim_iter}", 16, int(panel_size) - 18, scale=0.45, color=(65, 65, 65))
    return img


def metric_text(metrics: dict[str, Any]) -> str:
    parts = []
    for label, key in SHORT_METRICS:
        metric_key = "progress" if key == "progress" else METRIC_COLUMNS.get(key, key)
        parts.append(f"{label}={pct_metric(metrics.get(metric_key))}")
    return " ".join(parts)


def zero_causes(metrics: dict[str, Any]) -> str:
    bad = []
    for label, key in [
        ("collision", "collision"),
        ("offroad", "offroad"),
        ("ttc", "ttc"),
        ("progress", "progress"),
        ("comfort", "comfort"),
        ("speed", "speed_limit"),
    ]:
        metric_key = "progress" if key == "progress" else METRIC_COLUMNS.get(key, key)
        val = float_or_none(metrics.get(metric_key))
        if val is not None and val < 1.0 - EPS:
            bad.append(label)
    return ",".join(bad) if bad else "none"


def compose_side_by_side(
    left_img: np.ndarray,
    right_img: np.ndarray,
    left_metrics: dict[str, Any],
    right_metrics: dict[str, Any],
    title: str,
    left_label: str,
    right_label: str,
    frame_idx: int,
    frame_count: int,
    sim_iter: int,
) -> np.ndarray:
    cv2_mod = require_cv2()
    if left_img.shape != right_img.shape:
        right_img = cv2_mod.resize(right_img, (left_img.shape[1], left_img.shape[0]), interpolation=cv2_mod.INTER_AREA)
    body = np.concatenate([left_img, right_img], axis=1)
    h, w = body.shape[:2]
    half = w // 2
    header = np.full((HEADER_H, w, 3), 248, dtype=np.uint8)
    cv2_mod.line(body, (half, 0), (half, h), (30, 30, 30), 3)
    put_line(header, f"{title} | frame {frame_idx + 1}/{frame_count} sim_iter {sim_iter}"[:230], 18, 28, scale=0.53, color=(10, 10, 10))
    delta_score = None
    if float_or_none(right_metrics.get("score")) is not None and float_or_none(left_metrics.get("score")) is not None:
        delta_score = right_metrics.get("score") - left_metrics.get("score")
    put_line(header, f"delta_score {right_label}-{left_label}={fmt_metric(delta_score)} | {left_label} causes: {zero_causes(left_metrics)}"[:210], 18, 58, scale=0.52, color=(130, 40, 20))
    put_line(header, f"{left_label}  {metric_text(left_metrics)}", 18, 94, scale=0.54, color=(185, 55, 30), thickness=1)
    put_line(header, f"{right_label}  {metric_text(right_metrics)}", half + 18, 94, scale=0.54, color=(20, 95, 165), thickness=1)
    put_line(header, f"{right_label} causes: {zero_causes(right_metrics)}"[:100], half + 18, 124, scale=0.50, color=(45, 45, 45))
    put_line(header, "progress = mean(making_progress, route_progress); values shown as percent", 18, 154, scale=0.46, color=(70, 70, 70))
    put_line(body, left_label, 24, 42, scale=1.10, color=(185, 55, 30), thickness=2)
    put_line(body, right_label, half + 24, 42, scale=1.10, color=(20, 95, 165), thickness=2)
    return np.vstack([header, body])


def render_simlog_video(
    msgpack_path: Path,
    output_path: Path,
    metrics: dict[str, Any],
    title: str,
    label: str = "FlowDrive",
    stride: int = 3,
    fps: int = 6,
    panel_size: int = 720,
    bounds: float = 60.0,
    offset: float = 20.0,
    map_cache_step: int = 20,
    quality: int = 7,
) -> dict[str, Any]:
    imageio_mod = require_imageio()
    log = load_simulation_log_cpu_safe(msgpack_path)
    map_features = collect_map_features(log, bounds, offset, map_cache_step)
    history_xy = simulation_history_xy(log)
    sample_count = len(log.simulation_history.data)
    frame_indices = list(range(0, sample_count, max(1, int(stride))))
    if frame_indices and frame_indices[-1] != sample_count - 1:
        frame_indices.append(sample_count - 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas_w = int(panel_size)
    with imageio_mod.get_writer(str(output_path), fps=int(fps), codec="libx264", quality=int(quality), macro_block_size=16) as writer:
        for frame_idx, sample_idx in enumerate(frame_indices):
            img = render_fast_panel(log, map_features, history_xy, sample_idx, panel_size, bounds, offset)
            header = np.full((HEADER_H, canvas_w, 3), 248, dtype=np.uint8)
            sim_iter = int(getattr(log.simulation_history.data[sample_idx].iteration, "index", sample_idx))
            put_line(header, f"{title} | frame {frame_idx + 1}/{len(frame_indices)} sim_iter {sim_iter}"[:120], 18, 34)
            put_line(header, f"{label}  {metric_text(metrics)}", 18, 82, scale=0.54, color=(20, 95, 165))
            put_line(header, f"causes: {zero_causes(metrics)}", 18, 122, scale=0.50, color=(45, 45, 45))
            writer.append_data(np.vstack([header, img]))
    return {
        "video": str(output_path),
        "frames": len(frame_indices),
        "source_iterations": sample_count,
        "map_features": len(map_features),
        "bytes": output_path.stat().st_size,
    }


def render_side_by_side_video(
    left_msgpack: Path,
    right_msgpack: Path,
    output_path: Path,
    left_metrics: dict[str, Any],
    right_metrics: dict[str, Any],
    title: str,
    left_label: str = "POST0",
    right_label: str = "POST1",
    stride: int = 3,
    fps: int = 6,
    panel_size: int = 720,
    bounds: float = 60.0,
    offset: float = 20.0,
    map_cache_step: int = 20,
    quality: int = 7,
) -> dict[str, Any]:
    imageio_mod = require_imageio()
    left_log = load_simulation_log_cpu_safe(left_msgpack)
    right_log = load_simulation_log_cpu_safe(right_msgpack)
    min_len = min(len(left_log.simulation_history.data), len(right_log.simulation_history.data))
    if min_len <= 0:
        raise RuntimeError("empty simulation history")
    frame_indices = list(range(0, min_len, max(1, int(stride))))
    if frame_indices[-1] != min_len - 1:
        frame_indices.append(min_len - 1)

    left_features = collect_map_features(left_log, bounds, offset, map_cache_step)
    right_features = collect_map_features(right_log, bounds, offset, map_cache_step)
    left_history_xy = simulation_history_xy(left_log)
    right_history_xy = simulation_history_xy(right_log)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio_mod.get_writer(str(output_path), fps=int(fps), codec="libx264", quality=int(quality), macro_block_size=16) as writer:
        for frame_idx, sample_idx in enumerate(frame_indices):
            left_img = render_fast_panel(left_log, left_features, left_history_xy, sample_idx, panel_size, bounds, offset)
            right_img = render_fast_panel(right_log, right_features, right_history_xy, sample_idx, panel_size, bounds, offset)
            sim_iter = int(getattr(left_log.simulation_history.data[sample_idx].iteration, "index", sample_idx))
            combined = compose_side_by_side(left_img, right_img, left_metrics, right_metrics, title, left_label, right_label, frame_idx, len(frame_indices), sim_iter)
            writer.append_data(combined)
    return {
        "video": str(output_path),
        "frames": len(frame_indices),
        "source_iterations": min_len,
        "left_map_features": len(left_features),
        "right_map_features": len(right_features),
        "bytes": output_path.stat().st_size,
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=json_default), encoding="utf-8")
