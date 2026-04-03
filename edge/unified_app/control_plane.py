#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一控制平面工具。

负责运行时配置、标识状态、模型枚举与快速标定。
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import yaml


RUNTIME_DIR_NAME = "runtime"
CONFIG_FILE_NAME = "unified_control.json"
MARKER_FILE_NAME = "unified_marker_state.json"
STATE_FILE_NAME = "state.json"
WEB_DIR_NAME = "webui_http_unified"
DEFAULT_MODEL_RELATIVE = Path("models/route_a_yolo26/yolo26n_aug_full_8419_gpu.om")
DEFAULT_DATA_YAML_RELATIVE = Path("config/yolo26_6cls.yaml")
DEFAULT_CALIBRATION_RELATIVE = Path("edge/laser_galvo/galvo_calibration.yaml")
MODEL_SEARCH_DIRS = (
    Path("models"),
    Path("d435_project/projects/yolo26_galvo/models"),
)
VALID_MODEL_SUFFIXES = {".om", ".mindir"}
VALID_MARKER_STYLES = ("rectangle", "circle")


def load_class_names(data_yaml_path: str) -> List[str]:
    path = Path(data_yaml_path)
    if not path.exists():
        return []
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return []

    raw_names = payload.get("names", [])
    try:
        nc = int(payload.get("nc", 0))
    except (TypeError, ValueError):
        nc = 0

    if isinstance(raw_names, dict):
        mapping: Dict[int, str] = {}
        for k, v in raw_names.items():
            try:
                mapping[int(k)] = str(v)
            except (TypeError, ValueError):
                continue
        total = max(nc, max(mapping.keys(), default=-1) + 1)
        return [mapping.get(i, f"Class-{i}") for i in range(total)]

    if isinstance(raw_names, list):
        names = [str(n) for n in raw_names]
        if nc > len(names):
            names.extend(f"Class-{i}" for i in range(len(names), nc))
        return names[:nc] if nc > 0 else names

    return [f"Class-{i}" for i in range(nc)] if nc > 0 else []


def _normalize_class_config(class_config: Any, *, strict: bool) -> Dict[str, dict]:
    if not isinstance(class_config, dict):
        if strict and class_config is not None:
            raise ValueError("class_config 必须是对象映射")
        return {}
    result: Dict[str, dict] = {}
    for raw_name, raw_item in class_config.items():
        name = str(raw_name).strip()
        if not name:
            if strict:
                raise ValueError("class_config 的类别名不能为空")
            continue
        if not isinstance(raw_item, dict):
            if strict:
                raise ValueError(f"class_config[{name}] 必须是对象")
            continue
        style = str(raw_item.get("style", "rectangle")).strip() or "rectangle"
        if style not in VALID_MARKER_STYLES:
            if strict:
                raise ValueError(f"不支持的类别标识样式: {style} ({name})")
            style = "rectangle"
        result[name] = {"enabled": bool(raw_item.get("enabled", True)), "style": style}
    return result


def _json_load(path: Path, default: dict) -> dict:
    if not path.exists():
        return dict(default)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return dict(default)


def _json_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def get_runtime_dir(ict_root: Path) -> Path:
    return Path(ict_root) / RUNTIME_DIR_NAME


def get_web_dir(ict_root: Path) -> Path:
    return Path(ict_root) / WEB_DIR_NAME


def get_runtime_config_path(ict_root: Path) -> Path:
    return get_runtime_dir(ict_root) / CONFIG_FILE_NAME


def get_marker_state_path(ict_root: Path) -> Path:
    return get_runtime_dir(ict_root) / MARKER_FILE_NAME


def get_live_state_path(ict_root: Path) -> Path:
    return get_web_dir(ict_root) / STATE_FILE_NAME


def _marker_rank(path: Path) -> tuple[float, str]:
    lower = path.name.lower()
    int8_bonus = -0.5 if "int8" in lower else 0.0
    if "yolo26m" in lower or "yolom" in lower:
        return (0 + int8_bonus, lower)
    if "yolo26n" in lower:
        return (1 + int8_bonus, lower)
    if "yolo26s" in lower:
        return (2 + int8_bonus, lower)
    return (3 + int8_bonus, lower)


def list_model_candidates(ict_root: Path) -> List[dict]:
    root = Path(ict_root)
    candidates: List[Path] = []
    for rel_dir in MODEL_SEARCH_DIRS:
        search_dir = root / rel_dir
        if not search_dir.exists():
            continue
        for path in search_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in VALID_MODEL_SUFFIXES:
                continue
            candidates.append(path)
    unique = sorted(set(candidates), key=_marker_rank)
    return [
        {
            "id": str(path.relative_to(root)),
            "name": path.name,
            "path": str(path),
        }
        for path in unique
    ]


def find_default_model_path(ict_root: Path) -> str:
    candidates = list_model_candidates(ict_root)
    if candidates:
        return candidates[0]["path"]
    return str(Path(ict_root) / DEFAULT_MODEL_RELATIVE)


def default_runtime_config(ict_root: Path) -> dict:
    root = Path(ict_root)
    return {
        "yolo_model": find_default_model_path(root),
        "data_yaml": str(root / DEFAULT_DATA_YAML_RELATIVE),
        "enable_hand_detection": False,
        "enable_distance_detection": False,
        "enable_video_stream": True,
        "danger_distance": 300,
        "conf_thres": 0.55,
        "enable_laser": False,
        "laser_serial": "/dev/ttyUSB0",
        "laser_baudrate": 115200,
        "laser_calibration": str(root / DEFAULT_CALIBRATION_RELATIVE),
        "camera_serial": "",
        "marker_style": "rectangle",
    }


def load_runtime_config(ict_root: Path) -> dict:
    defaults = default_runtime_config(ict_root)
    current = _json_load(get_runtime_config_path(ict_root), defaults)
    merged = dict(defaults)
    merged.update(current)
    if merged.get("marker_style") not in VALID_MARKER_STYLES:
        merged["marker_style"] = "rectangle"
    return merged


def save_runtime_config(ict_root: Path, payload: dict) -> dict:
    merged = load_runtime_config(ict_root)
    merged.update(payload)
    if merged.get("marker_style") not in VALID_MARKER_STYLES:
        raise ValueError(f"不支持的标识样式: {merged.get('marker_style')}")
    _json_save(get_runtime_config_path(ict_root), merged)
    return merged


def load_marker_state(ict_root: Path) -> dict:
    default_state = {
        "selected_track_id": None,
        "marker_style": "rectangle",
        "available_styles": list(VALID_MARKER_STYLES),
        "class_config": {},
    }
    payload = _json_load(get_marker_state_path(ict_root), default_state)
    merged = dict(default_state)
    merged.update(payload)
    if merged.get("marker_style") not in VALID_MARKER_STYLES:
        merged["marker_style"] = "rectangle"
    merged["class_config"] = _normalize_class_config(merged.get("class_config"), strict=False)
    return merged


def save_marker_state(ict_root: Path, payload: dict) -> dict:
    current = load_marker_state(ict_root)
    current.update(payload)
    if current.get("marker_style") not in VALID_MARKER_STYLES:
        raise ValueError(f"不支持的标识样式: {current.get('marker_style')}")
    current["class_config"] = _normalize_class_config(current.get("class_config"), strict=True)
    _json_save(get_marker_state_path(ict_root), current)
    return current


def load_live_state(ict_root: Path) -> dict:
    return _json_load(get_live_state_path(ict_root), {})


def list_serial_ports() -> List[str]:
    prefixes = ("/dev/ttyUSB", "/dev/ttyACM", "/dev/ttyAMA", "/dev/ttyS")
    found = []
    for prefix in prefixes:
        found.extend(str(path) for path in sorted(Path("/dev").glob(prefix.split("/")[-1] + "*")))
    return found


def list_camera_serials(ict_root: Path) -> List[str]:
    serials: List[str] = []
    state = load_live_state(ict_root)
    camera_serial = state.get("camera_serial")
    if camera_serial:
        serials.append(str(camera_serial))
    try:
        import pyrealsense2 as rs  # type: ignore

        ctx = rs.context()
        for dev in ctx.query_devices():
            serial = dev.get_info(rs.camera_info.serial_number)
            if serial and serial not in serials:
                serials.append(serial)
    except Exception:
        pass
    return serials


def _points_to_arrays(payload: dict) -> tuple[np.ndarray, np.ndarray]:
    pixel_points = np.asarray(payload.get("pixel_points", []), dtype=np.float32)
    galvo_points = np.asarray(payload.get("galvo_points", []), dtype=np.float32)
    if pixel_points.shape[0] < 4 or galvo_points.shape[0] < 4:
        raise ValueError("至少需要 4 对点才能求解 homography")
    if pixel_points.shape != galvo_points.shape:
        raise ValueError("像素点与振镜点数量不一致")
    if pixel_points.shape[1] != 2:
        raise ValueError("点格式必须是 [x, y]")
    return pixel_points, galvo_points


def solve_homography(payload: dict) -> dict:
    pixel_points, galvo_points = _points_to_arrays(payload)
    rows = []
    for (x, y), (u, v) in zip(pixel_points, galvo_points):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, x * u, y * u, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, x * v, y * v, v])
    matrix_a = np.asarray(rows, dtype=np.float64)
    _, _, vh = np.linalg.svd(matrix_a)
    matrix = vh[-1].reshape(3, 3)
    if abs(matrix[2, 2]) > 1e-9:
        matrix = matrix / matrix[2, 2]
    if not np.isfinite(matrix).all():
        raise ValueError("无法根据当前点对求解 homography")

    homogeneous = np.concatenate([pixel_points.astype(np.float64), np.ones((pixel_points.shape[0], 1))], axis=1)
    projected_h = homogeneous @ matrix.T
    projected = projected_h[:, :2] / projected_h[:, 2:3]
    errors = np.linalg.norm(projected - galvo_points, axis=1)
    return {
        "homography_matrix": matrix.tolist(),
        "mean_error": float(np.mean(errors)),
        "max_error": float(np.max(errors)),
        "point_count": int(pixel_points.shape[0]),
    }


def apply_calibration(ict_root: Path, calibration_file: str, payload: dict) -> dict:
    result = solve_homography(payload)
    target = Path(calibration_file)
    existing = {}
    if target.exists():
        existing = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    existing["homography_matrix"] = result["homography_matrix"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(existing, allow_unicode=True, sort_keys=False), encoding="utf-8")
    config = save_runtime_config(ict_root, {"laser_calibration": str(target)})
    return {
        **result,
        "calibration_file": str(target),
        "config": config,
    }


def build_config_payload(ict_root: Path) -> dict:
    config = load_runtime_config(ict_root)
    marker_state = load_marker_state(ict_root)
    return {
        "config": config,
        "marker_state": marker_state,
        "available_models": list_model_candidates(ict_root),
        "available_serial_ports": list_serial_ports(),
        "available_camera_serials": list_camera_serials(ict_root),
        "class_names": load_class_names(str(config.get("data_yaml", ""))),
        "live_state": load_live_state(ict_root),
        "available_marker_styles": list(VALID_MARKER_STYLES),
    }


def shell_env_exports(ict_root: Path) -> str:
    config = load_runtime_config(ict_root)
    exports = {
        "CONTROL_YOLO_MODEL": config.get("yolo_model", ""),
        "CONTROL_DATA_YAML": config.get("data_yaml", ""),
        "CONTROL_ENABLE_HAND_DETECTION": "1" if config.get("enable_hand_detection", False) else "0",
        "CONTROL_ENABLE_DISTANCE_DETECTION": "1" if config.get("enable_distance_detection", False) else "0",
        "CONTROL_ENABLE_VIDEO_STREAM": "1" if config.get("enable_video_stream", True) else "0",
        "CONTROL_DANGER_DISTANCE": config.get("danger_distance", ""),
        "CONTROL_CONF_THRES": config.get("conf_thres", ""),
        "CONTROL_ENABLE_LASER": "1" if config.get("enable_laser") else "0",
        "CONTROL_LASER_SERIAL": config.get("laser_serial", ""),
        "CONTROL_LASER_BAUDRATE": config.get("laser_baudrate", ""),
        "CONTROL_LASER_CALIBRATION": config.get("laser_calibration", ""),
        "CONTROL_CAMERA_SERIAL": config.get("camera_serial", ""),
    }
    lines = []
    for key, value in exports.items():
        lines.append(f"{key}={shlex.quote(str(value))}")
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Unified 控制平面工具")
    subparsers = parser.add_subparsers(dest="command", required=True)

    shell_parser = subparsers.add_parser("shell-env")
    shell_parser.add_argument("ict_root")

    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "shell-env":
        print(shell_env_exports(Path(args.ict_root)))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
