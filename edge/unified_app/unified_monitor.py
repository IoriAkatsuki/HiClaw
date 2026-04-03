#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一检测监控

能力说明：
1. YOLO 目标检测
2. MediaPipe 或 YOLO-pose 手部/腕点安全检测
3. 激光振镜打标
4. 目标 ID 保持与激光锁定跟随
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
import yaml

try:
    import mediapipe as mp
except ModuleNotFoundError:
    mp = None

try:
    import acl
except ModuleNotFoundError:
    acl = None

REPO_ROOT = Path(__file__).resolve().parents[2]
ULTRALYTICS_VENDOR_DIR = REPO_ROOT / "2026_3_12" / ".vendor_ultralytics_8419"
if ULTRALYTICS_VENDOR_DIR.exists():
    sys.path.insert(0, str(ULTRALYTICS_VENDOR_DIR))

try:
    from ultralytics import YOLO as UltralyticsYOLO
except ModuleNotFoundError:
    UltralyticsYOLO = None


# 添加 laser_galvo 路径（基于当前文件位置，避免依赖固定 HOME 路径）
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "laser_galvo"))


def _acl_status(ret) -> int:
    """兼容 ACL 接口返回 int 或 (obj, ret) 两种形式。"""
    if isinstance(ret, tuple):
        return int(ret[-1])
    if ret is None:
        return 0
    return int(ret)


def _destroy_dataset_safe(dataset) -> None:
    if dataset is None or acl is None:
        return
    destroy = getattr(acl.mdl, "destroy_dataset", None)
    if destroy is not None:
        destroy(dataset)


class AclLiteResource:
    """ACL 资源管理。"""

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.context = None
        self.stream = None

    def init(self) -> None:
        ret = acl.init()
        if ret != 0:
            raise RuntimeError(f"acl.init failed: {ret}")
        ret = acl.rt.set_device(self.device_id)
        if ret != 0:
            raise RuntimeError(f"set_device failed: {ret}")
        self.context, ret = acl.rt.create_context(self.device_id)
        if ret != 0:
            raise RuntimeError(f"create_context failed: {ret}")
        ret = acl.rt.set_context(self.context)
        if ret != 0:
            raise RuntimeError(f"set_context failed: {ret}")
        self.stream, ret = acl.rt.create_stream()
        if ret != 0:
            raise RuntimeError(f"create_stream failed: {ret}")
        print(f"✓ ACL 资源初始化成功 (Device {self.device_id})")


class AclLiteModel:
    """ACL 模型推理封装。"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model_id = None
        self.model_desc = None
        self.input_buffers: List[object] = []
        self.output_buffers: List[object] = []
        self.input_sizes: List[int] = []
        self.output_sizes: List[int] = []
        self.input_dataset = None
        self.output_dataset = None
        self.input_data_buffers: List[object] = []
        self.output_data_buffers: List[object] = []
        self.host_output_buffers: List[object] = []
        self.context = None

    def load(self) -> None:
        self.model_id, ret = acl.mdl.load_from_file(self.model_path)
        if ret != 0:
            raise RuntimeError(f"load_from_file failed: {ret}")
        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        if ret != 0:
            raise RuntimeError(f"get_desc failed: {ret}")
        self._init_io_sizes()
        self._prepare_io_buffers()
        print(f"✓ 模型加载成功: {self.model_path}")

    def _init_io_sizes(self) -> None:
        self.input_sizes = [
            acl.mdl.get_input_size_by_index(self.model_desc, i)
            for i in range(acl.mdl.get_num_inputs(self.model_desc))
        ]
        self.output_sizes = [
            acl.mdl.get_output_size_by_index(self.model_desc, i)
            for i in range(acl.mdl.get_num_outputs(self.model_desc))
        ]

    def _prepare_io_buffers(self) -> None:
        self.input_dataset = acl.mdl.create_dataset()
        self.output_dataset = acl.mdl.create_dataset()
        self.input_data_buffers = []
        self.output_data_buffers = []
        self.host_output_buffers = []

        for size in self.input_sizes:
            buf, ret = acl.rt.malloc(size, 0)
            if ret != 0:
                raise RuntimeError(f"malloc input buffer failed: {ret}")
            self.input_buffers.append(buf)
            data_buffer = acl.create_data_buffer(buf, size)
            ret = acl.mdl.add_dataset_buffer(self.input_dataset, data_buffer)
            if _acl_status(ret) != 0:
                raise RuntimeError("add input dataset buffer failed")
            self.input_data_buffers.append(data_buffer)

        for size in self.output_sizes:
            buf, ret = acl.rt.malloc(size, 0)
            if ret != 0:
                raise RuntimeError(f"malloc output buffer failed: {ret}")
            self.output_buffers.append(buf)
            data_buffer = acl.create_data_buffer(buf, size)
            ret = acl.mdl.add_dataset_buffer(self.output_dataset, data_buffer)
            if _acl_status(ret) != 0:
                raise RuntimeError("add output dataset buffer failed")
            self.output_data_buffers.append(data_buffer)

            host_buf = None
            malloc_host = getattr(acl.rt, "malloc_host", None)
            if malloc_host is not None:
                host_buf, ret = malloc_host(size)
                if ret != 0:
                    raise RuntimeError(f"malloc_host failed: {ret}")
            self.host_output_buffers.append(host_buf)

    def execute(self, image_bytes: np.ndarray) -> Optional[List[np.ndarray]]:
        if self.context is not None:
            acl.rt.set_context(self.context)

        ret = acl.rt.memcpy(
            self.input_buffers[0],
            self.input_sizes[0],
            acl.util.numpy_to_ptr(image_bytes),
            image_bytes.nbytes,
            4,
        )
        if ret != 0:
            return None

        ret = acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset)
        if ret != 0:
            return None

        outputs = []
        free_host = getattr(acl.rt, "free_host", None)
        for idx, size in enumerate(self.output_sizes):
            host_buf = self.host_output_buffers[idx]
            ephemeral = False
            if host_buf is None:
                malloc_host = getattr(acl.rt, "malloc_host", None)
                if malloc_host is None:
                    break
                host_buf, ret = malloc_host(size)
                if ret != 0:
                    break
                ephemeral = True

            ret = acl.rt.memcpy(host_buf, size, self.output_buffers[idx], size, 3)
            if ret != 0:
                if ephemeral and free_host is not None:
                    free_host(host_buf)
                break

            out_np = acl.util.ptr_to_numpy(host_buf, (size // 4,), 11)
            outputs.append(out_np.copy())

            if ephemeral and free_host is not None:
                free_host(host_buf)

        return outputs


class UltralyticsYoloModel:
    """Ultralytics 检测模型封装，支持 .pt / .onnx。"""

    def __init__(self, model_path: str, device: str = "cpu", imgsz: int = 640):
        self.model_path = model_path
        self.device = device
        self.imgsz = imgsz
        self.model = None

    def load(self) -> None:
        if UltralyticsYOLO is None:
            raise ModuleNotFoundError(
                "未找到 ultralytics 模块，请先安装 ultralytics，或使用仓库自带的 2026_3_12/.vendor_ultralytics_8419。"
            )
        self.model = UltralyticsYOLO(self.model_path)
        print(f"✓ Ultralytics 模型加载成功: {self.model_path} (device={self.device})")

    def infer(self, frame: np.ndarray, conf_thres: float) -> List[dict]:
        results = self.model.predict(
            source=frame,
            conf=conf_thres,
            device=self.device,
            imgsz=self.imgsz,
            verbose=False,
        )
        if not results:
            return []
        return convert_ultralytics_result_to_objects(results[0])


def detect_yolo_backend(model_path: str) -> str:
    suffix = Path(model_path).suffix.lower()
    if suffix == ".om":
        return "acl"
    if suffix in {".pt", ".onnx"}:
        return "ultralytics"
    raise ValueError(f"不支持的 YOLO 模型格式: {model_path}")


def _scalar_from_any(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:
            pass
    arr = np.asarray(value).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(arr[0])


def convert_ultralytics_result_to_objects(result) -> List[dict]:
    objects = []
    names = getattr(result, "names", {}) or {}
    for box in getattr(result, "boxes", []) or []:
        cls_id = int(round(_scalar_from_any(box.cls)))
        score = _scalar_from_any(box.conf)
        xyxy = np.asarray(box.xyxy).reshape(-1).tolist()
        if len(xyxy) != 4:
            continue
        if isinstance(names, dict):
            cls_name = names.get(cls_id, f"Class-{cls_id}")
        else:
            cls_name = names[cls_id] if cls_id < len(names) else f"Class-{cls_id}"
        objects.append(
            {
                "box": [float(v) for v in xyxy],
                "score": float(score),
                "cls": cls_id,
                "name": cls_name,
            }
        )
    return objects


def prepare_acl_yolo_input(frame: np.ndarray, input_format: str = "rgb_chw_float32",
                           resizer=None) -> np.ndarray:
    """为 ACL YOLO 模型准备输入张量。"""
    img = resizer.resize(frame) if resizer else cv2.resize(frame, (640, 640))
    if input_format == "bgr_hwc_uint8":
        return img.astype(np.uint8)
    if input_format == "rgb_chw_float32":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return np.ascontiguousarray(img[None, ...], dtype=np.float32)
    raise ValueError(f"不支持的 ACL YOLO 输入格式: {input_format}")


def postprocess_yolo(
    pred_flat: np.ndarray,
    ratio: float,
    dwdh: Tuple[float, float],
    nc: int = 61,
    conf_thres: float = 0.55,
    iou_thres: float = 0.45,
    names: Optional[List[str]] = None,
) -> List[dict]:
    """YOLOv8 后处理。"""
    ch = 4 + nc
    anchors = pred_flat.size // ch
    pred = pred_flat.reshape(1, ch, anchors)
    pred = np.transpose(pred, (0, 2, 1))[0]

    boxes_xywh = pred[:, :4]
    cls_scores = 1.0 / (1.0 + np.exp(-pred[:, 4:]))

    cls_ind = cls_scores.argmax(axis=1)
    cls_score = cls_scores[np.arange(len(cls_ind)), cls_ind]

    mask = cls_score >= conf_thres
    boxes_xywh = boxes_xywh[mask]
    cls_score = cls_score[mask]
    cls_ind = cls_ind[mask]

    if len(boxes_xywh) == 0:
        return []

    box_xyxy = np.zeros_like(boxes_xywh)
    box_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    box_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    box_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    box_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
    box_xyxy[:, [0, 2]] = (box_xyxy[:, [0, 2]] - dwdh[0]) / ratio
    box_xyxy[:, [1, 3]] = (box_xyxy[:, [1, 3]] - dwdh[1]) / ratio

    from collections import defaultdict

    class_boxes = defaultdict(list)
    for i in range(len(box_xyxy)):
        class_boxes[cls_ind[i]].append((box_xyxy[i], cls_score[i], i))

    keep_indices = []
    for _, boxes_list in class_boxes.items():
        boxes = np.array([item[0] for item in boxes_list])
        scores = np.array([item[1] for item in boxes_list])
        original_indices = [item[2] for item in boxes_list]

        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        order = scores.argsort()[::-1]

        while order.size > 0:
            i = order[0]
            keep_indices.append(original_indices[i])
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            union = areas[i] + areas[order[1:]] - inter + 1e-6
            iou = inter / union
            order = order[np.where(iou <= iou_thres)[0] + 1]

    dets = []
    for i in keep_indices:
        cls_id = int(cls_ind[i])
        cls_name = names[cls_id] if names and cls_id < len(names) else f"Class-{cls_id}"
        dets.append(
            {
                "box": box_xyxy[i].tolist(),
                "score": float(cls_score[i]),
                "cls": cls_id,
                "name": cls_name,
            }
        )
    return dets


def postprocess_yolov10(
    pred_flat: np.ndarray,
    img_shape: Tuple[int, int, int],
    conf_thres: float = 0.55,
    names: Optional[List[str]] = None,
) -> List[dict]:
    """YOLOv10 后处理（模型已内置 NMS，输出 [1, 300, 6]）。"""
    pred = pred_flat.reshape(-1, 6)
    scores = pred[:, 4]
    pred = pred[scores >= conf_thres]
    if len(pred) == 0:
        return []

    h_orig, w_orig = img_shape[:2]
    scale_x, scale_y = w_orig / 640.0, h_orig / 640.0

    dets = []
    for row in pred:
        x1, y1, x2, y2, score, cls_id = row
        cls_id = int(cls_id)
        cls_name = names[cls_id] if names and cls_id < len(names) else f"Class-{cls_id}"
        dets.append(
            {
                "box": [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y],
                "score": float(score),
                "cls": cls_id,
                "name": cls_name,
            }
        )
    return dets


def check_hand_near_objects(
    wrist_pos: Tuple[int, int],
    depth_mm: Optional[float],
    objects: Sequence[dict],
    danger_distance: int = 300,
) -> Tuple[bool, Optional[dict]]:
    """检查手是否接近物体。"""
    if depth_mm is None or depth_mm <= 0:
        return False, None

    wx, wy = wrist_pos
    for obj in objects:
        x1, y1, x2, y2 = obj["box"]
        if x1 - 50 <= wx <= x2 + 50 and y1 - 50 <= wy <= y2 + 50 and depth_mm < danger_distance:
            return True, obj
    return False, None


def bbox_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return inter / union


def assign_track_ids(
    objects: Sequence[dict],
    tracks: Dict[int, dict],
    next_track_id: int,
    iou_thres: float = 0.3,
    max_missing: int = 6,
) -> Tuple[List[dict], Dict[int, dict], int]:
    """用 IoU 做轻量级目标 ID 保持。"""
    updated_tracks = {track_id: dict(track) for track_id, track in tracks.items()}
    matched_track_ids: Set[int] = set()
    matched_det_indices: Set[int] = set()
    det_to_track: Dict[int, int] = {}
    candidates = []

    for det_idx, obj in enumerate(objects):
        for track_id, track in updated_tracks.items():
            if obj.get("name") != track.get("name"):
                continue
            iou = bbox_iou(obj["box"], track["box"])
            if iou >= iou_thres:
                candidates.append((iou, det_idx, track_id))

    for _, det_idx, track_id in sorted(candidates, key=lambda item: item[0], reverse=True):
        if det_idx in matched_det_indices or track_id in matched_track_ids:
            continue
        matched_det_indices.add(det_idx)
        matched_track_ids.add(track_id)
        det_to_track[det_idx] = track_id

    tracked_objects: List[dict] = []
    for det_idx, obj in enumerate(objects):
        tracked_obj = dict(obj)
        if det_idx in det_to_track:
            track_id = det_to_track[det_idx]
            tracked_obj["track_id"] = track_id
            updated_tracks[track_id] = {
                "box": list(obj["box"]),
                "name": obj.get("name"),
                "cls": obj.get("cls"),
                "score": obj.get("score", 0.0),
                "missing": 0,
            }
        else:
            tracked_obj["track_id"] = next_track_id
            updated_tracks[next_track_id] = {
                "box": list(obj["box"]),
                "name": obj.get("name"),
                "cls": obj.get("cls"),
                "score": obj.get("score", 0.0),
                "missing": 0,
            }
            next_track_id += 1
        tracked_objects.append(tracked_obj)

    active_track_ids = {obj.get("track_id") for obj in tracked_objects if obj.get("track_id") is not None}
    for track_id in list(updated_tracks):
        if track_id in active_track_ids:
            continue
        updated_tracks[track_id]["missing"] = updated_tracks[track_id].get("missing", 0) + 1
        if updated_tracks[track_id]["missing"] > max_missing:
            updated_tracks.pop(track_id, None)

    return tracked_objects, updated_tracks, next_track_id


def select_laser_targets(
    objects: Sequence[dict],
    min_score: float,
    max_targets: int,
    allowed_classes: Optional[Sequence[str]] = None,
    preferred_track_ids: Optional[Set[int]] = None,
) -> List[dict]:
    """筛选激光目标，并优先保留上一次已经锁定的 track。"""
    preferred_track_ids = preferred_track_ids or set()
    allowed = set(allowed_classes) if allowed_classes else None

    filtered = []
    for obj in objects:
        if obj.get("score", 0.0) < min_score:
            continue
        if allowed is not None and obj.get("name") not in allowed:
            continue
        filtered.append(dict(obj))

    filtered.sort(
        key=lambda obj: (
            0 if obj.get("track_id") in preferred_track_ids else 1,
            -obj.get("score", 0.0),
            obj.get("track_id", 10**9),
        )
    )
    return filtered[:max_targets]


def draw_object_annotations(
    frame: np.ndarray,
    objects: Sequence[dict],
    highlighted_track_ids: Optional[Set[int]] = None,
) -> np.ndarray:
    """在图像上绘制检测框、类别、分数与 track_id。"""
    highlighted_track_ids = highlighted_track_ids or set()
    for obj in objects:
        x1, y1, x2, y2 = [int(round(v)) for v in obj["box"]]
        track_id = obj.get("track_id", -1)
        locked = track_id in highlighted_track_ids
        color = (0, 255, 255) if locked else (0, 255, 0)
        thickness = 3 if locked else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        label = f"{obj.get('name', 'cls')}#{track_id} {obj.get('score', 0.0):.2f}"
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
        if locked:
            cv2.putText(
                frame,
                "LOCKED",
                (x1, min(frame.shape[0] - 10, y2 + 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )
    return frame


def build_debug_overlay(
    width: int,
    height: int,
    objects: Sequence[dict],
    highlighted_track_ids: Optional[Set[int]] = None,
) -> np.ndarray:
    """生成仅包含识别框和标签的透明叠加层。"""
    highlighted_track_ids = highlighted_track_ids or set()
    overlay = np.zeros((height, width, 4), dtype=np.uint8)

    for obj in objects:
        x1, y1, x2, y2 = [int(round(v)) for v in obj["box"]]
        track_id = obj.get("track_id", -1)
        locked = track_id in highlighted_track_ids
        color_bgr = (0, 255, 255) if locked else (0, 255, 0)
        color = (*color_bgr, 255)
        thickness = 3 if locked else 2
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)

        label = f"{obj.get('name', 'cls')}#{track_id} {obj.get('score', 0.0):.2f}"
        cv2.putText(
            overlay,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            lineType=cv2.LINE_AA,
        )
        if locked:
            cv2.putText(
                overlay,
                "LOCKED",
                (x1, min(height - 10, y2 + 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255, 255),
                2,
                lineType=cv2.LINE_AA,
            )
    return overlay


def build_webui_artifact_plan(write_debug_assets: bool) -> Tuple[str, ...]:
    """返回当前运行模式下需要刷新的 WebUI 产物清单。"""
    artifacts = ["frame.jpg", "state.json"]
    if write_debug_assets:
        artifacts.extend(["debug_frame.jpg", "debug_overlay.png"])
    return tuple(artifacts)


def cleanup_optional_webui_artifacts(web_dir: Path, keep_names: Sequence[str]) -> None:
    """清理当前模式不会再刷新的旧调试产物，避免前端读到陈旧文件。"""
    keep = set(keep_names)
    for name in ("debug_frame.jpg", "debug_overlay.png"):
        if name in keep:
            continue
        path = web_dir / name
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass


def expand_box(box: Sequence[float], scale: float) -> List[float]:
    """以框中心为基准放大/缩小包围框。"""
    x1, y1, x2, y2 = [float(v) for v in box]
    if scale <= 0:
        raise ValueError(f"box scale 必须大于 0，当前为 {scale}")
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    half_w = (x2 - x1) * scale / 2.0
    half_h = (y2 - y1) * scale / 2.0
    return [cx - half_w, cy - half_h, cx + half_w, cy + half_h]


def smooth_box(previous_box: Sequence[float], current_box: Sequence[float], alpha: float) -> List[float]:
    """对激光框做指数平滑，抑制微小抖动。"""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return [
        float((1.0 - alpha) * p + alpha * c)
        for p, c in zip(previous_box, current_box)
    ]


def should_refresh_laser_box(
    previous_box: Sequence[float],
    current_box: Sequence[float],
    center_threshold_px: float,
    size_threshold_ratio: float,
) -> bool:
    """判断激光框是否变化到需要重发。"""
    px1, py1, px2, py2 = [float(v) for v in previous_box]
    cx1, cy1, cx2, cy2 = [float(v) for v in current_box]

    p_cx = (px1 + px2) / 2.0
    p_cy = (py1 + py2) / 2.0
    c_cx = (cx1 + cx2) / 2.0
    c_cy = (cy1 + cy2) / 2.0
    center_shift = float(np.hypot(c_cx - p_cx, c_cy - p_cy))
    if center_shift >= center_threshold_px:
        return True

    p_w = max(1.0, px2 - px1)
    p_h = max(1.0, py2 - py1)
    c_w = max(1.0, cx2 - cx1)
    c_h = max(1.0, cy2 - cy1)
    width_ratio = abs(c_w - p_w) / p_w
    height_ratio = abs(c_h - p_h) / p_h
    return max(width_ratio, height_ratio) >= size_threshold_ratio


def resolve_camera_serial(devices, rs_module, preferred_serial: str = "") -> str:
    preferred = (preferred_serial or "").strip()
    available = [dev.get_info(rs_module.camera_info.serial_number) for dev in devices]
    if not available:
        raise RuntimeError("未找到 RealSense 设备")
    if preferred:
        if preferred not in available:
            raise RuntimeError(f"指定相机序列号不存在: {preferred}")
        return preferred
    return available[0]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo-model", required=True, help="YOLO 模型路径，支持 .om / .pt / .onnx")
    parser.add_argument("--yolo-device", default="cpu", help="Ultralytics 推理设备，如 cpu / 0")
    parser.add_argument(
        "--yolo-acl-input-format",
        default="rgb_chw_float32",
        choices=["rgb_chw_float32", "bgr_hwc_uint8"],
        help="ACL YOLO 模型输入格式，最新 yolo26 OM 默认使用 rgb_chw_float32。",
    )
    parser.add_argument("--disable-hand", action="store_true", help="关闭手部检测（MediaPipe）")
    parser.add_argument(
        "--disable-distance",
        "--disable-distance-detection",
        dest="disable_distance",
        action="store_true",
        help="关闭距离危险检测（仍可保留手部检测）",
    )
    parser.add_argument("--data-yaml", required=True, help="数据配置 YAML")
    parser.add_argument("--danger-distance", type=int, default=300, help="危险距离 (mm)")
    parser.add_argument("--conf-thres", type=float, default=0.55, help="YOLO 置信度阈值")
    parser.add_argument("--track-iou-thres", type=float, default=0.3, help="对象跟踪 IoU 阈值")
    parser.add_argument("--track-max-missing", type=int, default=6, help="对象丢失后保留帧数")
    parser.add_argument("--enable-laser", action="store_true", help="启用激光振镜标记")
    parser.add_argument("--laser-serial", default="/dev/ttyUSB0", help="激光振镜串口设备")
    parser.add_argument("--laser-baudrate", type=int, default=115200, help="激光振镜串口波特率")
    parser.add_argument("--laser-calibration", help="激光校准文件（可选）")
    parser.add_argument("--laser-target-classes", nargs="+", help="仅激光标记这些类别（可选）")
    parser.add_argument("--laser-min-score", type=float, default=0.7, help="激光标记最小置信度")
    parser.add_argument("--laser-box-scale", type=float, default=1.2, help="激光绘制框相对检测框的放大倍率")
    parser.add_argument("--laser-smoothing-alpha", type=float, default=0.35, help="激光框平滑系数，越大越跟手，越小越稳")
    parser.add_argument("--laser-center-threshold-px", type=float, default=8.0, help="中心位移超过该像素阈值时才重发激光框")
    parser.add_argument("--laser-size-threshold-ratio", type=float, default=0.12, help="框尺寸变化超过该比例时才重发激光框")
    parser.add_argument("--laser-force-refresh-interval", type=float, default=0.35, help="即使变化不大也强制重发的最大间隔（秒）")
    parser.add_argument("--laser-update-interval", type=float, default=0.0, help="激光更新最小间隔（秒）")
    parser.add_argument("--max-laser-targets", type=int, default=10, help="激光同时标记的最大目标数")
    parser.add_argument("--write-debug-assets", action="store_true", help="额外写出调试帧和透明叠加层")
    parser.add_argument("--camera-serial", default="", help="指定 RealSense 序列号；为空时自动选择第一个设备")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    yolo_backend = detect_yolo_backend(args.yolo_model)
    needs_acl = yolo_backend == "acl"

    if needs_acl and acl is None:
        raise ModuleNotFoundError("未找到 acl 模块，请先安装 Ascend ACL Python 运行时。")
    if not args.disable_hand and mp is None:
        raise ModuleNotFoundError("未找到 mediapipe 模块，请先安装 mediapipe。")

    print("=" * 60)
    print("统一检测监控 - 物体 + 手部安全")
    print("=" * 60)

    with open(args.data_yaml, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    names = data_cfg.get("names", [])
    print(f"✓ 加载 {len(names)} 个类别")

    res = None
    if needs_acl:
        res = AclLiteResource()
        res.init()

    if yolo_backend == "acl":
        yolo_model = AclLiteModel(args.yolo_model)
        yolo_model.load()
        yolo_model.context = res.context
    else:
        yolo_model = UltralyticsYoloModel(args.yolo_model, device=args.yolo_device)
        yolo_model.load()

    hands = None
    if args.disable_hand:
        print("\n手部检测: 已关闭")
    else:
        print("\n初始化 MediaPipe Hands...")
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        print("✓ MediaPipe Hands 已初始化 (轻量级追踪模式)")

    print("\n初始化 RealSense D435...")
    import pyrealsense2 as rs

    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        raise RuntimeError("未找到 RealSense 设备")
    serial = resolve_camera_serial(devices, rs, args.camera_serial)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    color_sensor = profile.get_device().first_color_sensor()
    color_sensor.set_option(rs.option.enable_auto_exposure, 1)
    exposure_range = color_sensor.get_option_range(rs.option.exposure)
    target_exposure = max(exposure_range.min, min(exposure_range.max, 33333))
    color_sensor.set_option(rs.option.enable_auto_exposure, 0)
    color_sensor.set_option(rs.option.exposure, target_exposure)
    time.sleep(0.1)
    color_sensor.set_option(rs.option.enable_auto_exposure, 1)

    print(f"✓ RealSense D435 已连接 ({serial})")
    print(f"  曝光范围: {exposure_range.min:.0f}-{exposure_range.max:.0f}μs")
    print(f"  初始曝光: {target_exposure:.0f}μs (~1/{1000000/target_exposure:.0f}s)")
    print("  自动曝光: 已启用")

    galvo = None
    if args.enable_laser:
        try:
            from galvo_controller import LaserGalvoController

            galvo = LaserGalvoController(
                serial_port=args.laser_serial,
                baudrate=args.laser_baudrate,
                calibration_file=args.laser_calibration,
            )
            if galvo.connect():
                print(f"✓ 激光振镜已连接: {args.laser_serial}")
            else:
                print("✗ 激光振镜连接失败，继续运行（无激光标记）")
                galvo = None
        except Exception as exc:
            print(f"✗ 激光初始化失败: {exc}")
            galvo = None

    web_dir = Path.home() / "ICT" / "webui_http_unified"
    web_dir.mkdir(parents=True, exist_ok=True)
    webui_artifacts = build_webui_artifact_plan(args.write_debug_assets)
    cleanup_optional_webui_artifacts(web_dir, webui_artifacts)

    # ── DVPP hardware accelerators (offload resize + JPEG from CPU) ──
    dvpp_resizer = None
    dvpp_jpege = None
    try:
        from dvpp_stream import DvppVpcResizer, DvppJpegEncoder
        dvpp_resizer = DvppVpcResizer(640, 480, 640, 640)
        print("✓ DVPP VPC resizer ready (640x480 → 640x640)")
    except Exception as e:
        print(f"  DVPP resizer unavailable, using cv2: {e}")
    try:
        from dvpp_stream import DvppJpegEncoder
        dvpp_jpege = DvppJpegEncoder(640, 480, quality=75)
        print("✓ DVPP JPEGE encoder ready")
    except Exception as e:
        print(f"  DVPP JPEGE unavailable, using cv2: {e}")

    print(f"\n✓ WebUI 输出: {web_dir}")
    print(f"  产物: {', '.join(webui_artifacts)}")
    print("=" * 60)
    print("\n开始监控...\n")

    last_ts = None
    frame_count = 0
    last_hand_results = None
    distance_detection_enabled = not args.disable_distance
    last_laser_time = 0.0
    last_force_refresh_time = 0.0
    track_state: Dict[int, dict] = {}
    next_track_id = 1
    locked_track_ids: Set[int] = set()
    laser_active = False
    last_laser_box: Optional[List[float]] = None
    last_laser_track_id: Optional[int] = None

    executor = ThreadPoolExecutor(max_workers=2 if not args.disable_hand else 1)

    def run_yolo_inference(frame: np.ndarray, h_orig: int, w_orig: int) -> Tuple[List[dict], float]:
        t0 = time.time()
        objects: List[dict] = []
        if yolo_backend == "acl":
            img_yolo = prepare_acl_yolo_input(frame, args.yolo_acl_input_format, dvpp_resizer)
            yolo_outputs = yolo_model.execute(img_yolo)
            if yolo_outputs:
                total = yolo_outputs[0].size
                if total == 300 * 6 or (total % 300 == 0 and total // 300 <= 7):
                    objects = postprocess_yolov10(
                        yolo_outputs[0], frame.shape, conf_thres=args.conf_thres, names=names
                    )
                else:
                    ratio = min(640 / w_orig, 640 / h_orig)
                    dw = (640 - int(w_orig * ratio)) / 2
                    dh = (640 - int(h_orig * ratio)) / 2
                    objects = postprocess_yolo(
                        yolo_outputs[0],
                        ratio,
                        (dw, dh),
                        nc=len(names),
                        conf_thres=args.conf_thres,
                        names=names,
                    )
        else:
            objects = yolo_model.infer(frame, args.conf_thres)
        return objects, (time.time() - t0) * 1000.0

    def run_mediapipe_detection(frame: np.ndarray, should_detect: bool):
        t0 = time.time()
        if should_detect:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame_rgb)
        else:
            result = None
        return result, (time.time() - t0) * 1000.0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            h_orig, w_orig = frame.shape[:2]
            debug_frame = frame.copy() if args.write_debug_assets else None

            objects, yolo_ms = run_yolo_inference(frame, h_orig, w_orig)
            objects, track_state, next_track_id = assign_track_ids(
                objects,
                track_state,
                next_track_id,
                iou_thres=args.track_iou_thres,
                max_missing=args.track_max_missing,
            )

            if args.disable_hand:
                hand_results = None
                hand_ms = 0.0
            else:
                should_detect_hand = frame_count % 3 == 0
                hand_future = executor.submit(run_mediapipe_detection, frame, should_detect_hand)
                hand_results_new, hand_ms = hand_future.result()
                if hand_results_new is not None:
                    last_hand_results = hand_results_new
                hand_results = last_hand_results

            is_danger = False
            min_depth_mm = None
            hand_count = 0
            danger_obj = None

            if hand_results is not None and hasattr(hand_results, "multi_hand_landmarks") and hand_results.multi_hand_landmarks:
                hand_count = len(hand_results.multi_hand_landmarks)
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    key_points = [0, 2, 4, 5, 8, 9, 12, 13, 16, 17, 20]
                    for idx in key_points:
                        landmark = hand_landmarks.landmark[idx]
                        px, py = int(landmark.x * w_orig), int(landmark.y * h_orig)
                        color = (0, 0, 255) if idx == 0 else (0, 255, 0)
                        cv2.circle(frame, (px, py), 5, color, -1)

                    wrist = hand_landmarks.landmark[0]
                    wx, wy = int(wrist.x * w_orig), int(wrist.y * h_orig)
                    wx = max(0, min(w_orig - 1, wx))
                    wy = max(0, min(h_orig - 1, wy))

                    try:
                        depth_m = depth_frame.get_distance(wx, wy)
                        depth_mm = depth_m * 1000.0
                        if distance_detection_enabled and depth_mm > 0:
                            if min_depth_mm is None or depth_mm < min_depth_mm:
                                min_depth_mm = depth_mm
                            if depth_mm < args.danger_distance:
                                is_danger = True
                                cv2.circle(frame, (wx, wy), 15, (0, 0, 255), -1)
                                near_obj, obj = check_hand_near_objects(
                                    (wx, wy), depth_mm, objects, args.danger_distance
                                )
                                if near_obj:
                                    danger_obj = obj
                                    cv2.putText(
                                        frame,
                                        f"DANGER! Hand near {obj['name']}: {depth_mm:.0f}mm",
                                        (30, 40),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.9,
                                        (0, 0, 255),
                                        3,
                                    )
                                else:
                                    cv2.putText(
                                        frame,
                                        f"DANGER! Hand too close: {depth_mm:.0f}mm",
                                        (30, 40),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.9,
                                        (0, 0, 255),
                                        3,
                                    )
                    except Exception:
                        pass

            current_candidates = select_laser_targets(
                objects,
                min_score=args.laser_min_score,
                max_targets=args.max_laser_targets,
                allowed_classes=args.laser_target_classes,
                preferred_track_ids=locked_track_ids,
            )
            current_candidate_ids = {obj.get("track_id") for obj in current_candidates}
            highlighted_ids = locked_track_ids or current_candidate_ids
            debug_overlay = None
            if args.write_debug_assets:
                debug_overlay = build_debug_overlay(w_orig, h_orig, objects, highlighted_track_ids=highlighted_ids)
            draw_object_annotations(frame, objects, highlighted_track_ids=highlighted_ids)

            laser_marked = False
            if galvo is not None:
                now = time.time()
                if now - last_laser_time >= args.laser_update_interval:
                    if current_candidates and not is_danger:
                        obj = current_candidates[0]
                        target_track_id = obj["track_id"]
                        target_box = expand_box(obj["box"], args.laser_box_scale)
                        if last_laser_box is not None and last_laser_track_id == target_track_id:
                            target_box = smooth_box(last_laser_box, target_box, args.laser_smoothing_alpha)

                        needs_refresh = (
                            not laser_active
                            or last_laser_box is None
                            or last_laser_track_id != target_track_id
                            or should_refresh_laser_box(
                                last_laser_box,
                                target_box,
                                center_threshold_px=args.laser_center_threshold_px,
                                size_threshold_ratio=args.laser_size_threshold_ratio,
                            )
                            or (now - last_force_refresh_time) >= args.laser_force_refresh_interval
                        )

                        if needs_refresh:
                            galvo.task_index = 0
                            try:
                                if hasattr(galvo, "draw_box_sweep"):
                                    galvo.draw_box_sweep(
                                        target_box,
                                        pixel_coords=True,
                                        image_width=w_orig,
                                        image_height=h_orig,
                                        max_tasks=8,
                                    )
                                else:
                                    galvo.draw_box(
                                        target_box,
                                        pixel_coords=True,
                                        image_width=w_orig,
                                        image_height=h_orig,
                                        steps_per_edge=15,
                                    )
                                galvo.update_tasks()
                                locked_track_ids = {target_track_id}
                                laser_marked = True
                                laser_active = True
                                last_laser_box = list(target_box)
                                last_laser_track_id = target_track_id
                                last_laser_time = now
                                last_force_refresh_time = now
                            except Exception as exc:
                                print(
                                    "[LASER_DRAW_ERROR] "
                                    f"class={obj.get('name', 'unknown')} "
                                    f"score={obj.get('score', 0.0):.3f} "
                                    f"box={obj.get('box')} error={exc}"
                                )
                        else:
                            locked_track_ids = {target_track_id}
                            laser_active = True
                    elif laser_active:
                        try:
                            # 参考 Detect_choose_serial.py：当没有安全目标时立即清空旧任务。
                            galvo.update_tasks()
                        except Exception as exc:
                            print(f"[LASER_CLEAR_ERROR] {exc}")
                        locked_track_ids.clear()
                        laser_active = False
                        last_laser_box = None
                        last_laser_track_id = None
                        last_laser_time = now

            now = time.time()
            fps = 1.0 / (now - last_ts) if last_ts else 0.0
            last_ts = now

            info = (
                f"FPS: {fps:.1f} | YOLO: {yolo_ms:.1f}ms | Hand: {hand_ms:.1f}ms "
                f"| Obj: {len(objects)} | Locked: {sorted(locked_track_ids)}"
            )
            cv2.putText(frame, info, (10, h_orig - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            frame_count += 1
            if frame_count % 30 == 0:
                status = "⚠ DANGER" if is_danger else "✓ SAFE"
                print(
                    f"{status} | FPS: {fps:.1f} | YOLO: {yolo_ms:.1f}ms | Hand: {hand_ms:.1f}ms | "
                    f"物体: {len(objects)} | 手: {hand_count} | 锁定: {sorted(locked_track_ids)} "
                    f"| 激光: {'ON' if laser_active else 'OFF'}"
                )

            if dvpp_jpege:
                try:
                    jpeg_bytes = dvpp_jpege.encode(frame)
                    (web_dir / "frame.jpg").write_bytes(jpeg_bytes)
                except Exception:
                    cv2.imwrite(str(web_dir / "frame.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            else:
                cv2.imwrite(str(web_dir / "frame.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if args.write_debug_assets and debug_frame is not None and debug_overlay is not None:
                cv2.imwrite(str(web_dir / "debug_frame.jpg"), debug_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                cv2.imwrite(str(web_dir / "debug_overlay.png"), debug_overlay, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            state = {
                "fps": fps,
                "yolo_ms": yolo_ms,
                "hand_ms": hand_ms,
                "frame_width": w_orig,
                "frame_height": h_orig,
                "objects": len(objects),
                "object_details": [
                    {
                        "track_id": obj.get("track_id"),
                        "name": obj.get("name"),
                        "score": round(float(obj.get("score", 0.0)), 4),
                        "box": [int(round(v)) for v in obj.get("box", [])],
                        "locked": obj.get("track_id") in locked_track_ids,
                    }
                    for obj in objects[:10]
                ],
                "hands": hand_count,
                "is_danger": is_danger,
                "laser_enabled": galvo is not None,
                "laser_marked": laser_marked,
                "laser_active": laser_active,
                "locked_track_ids": sorted(locked_track_ids),
                "min_depth_mm": min_depth_mm,
                "danger_object": danger_obj["name"] if danger_obj else None,
                "write_debug_assets": args.write_debug_assets,
                "ts": now,
            }
            with open(web_dir / "state.json", "w", encoding="utf-8") as f:
                json.dump(state, f)

    except KeyboardInterrupt:
        print("\n\n监控已停止")
    finally:
        executor.shutdown(wait=True)
        if hands is not None:
            hands.close()
        pipeline.stop()
        if galvo:
            try:
                galvo.disconnect()
            except Exception as exc:
                print(f"[GALVO_DISCONNECT_ERROR] {exc}")
        print("✓ 资源已释放")


if __name__ == "__main__":
    main()
