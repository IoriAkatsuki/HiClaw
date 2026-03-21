#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一检测监控（多进程版）

目标：
1. 一个进程独占 D435 采集
2. 一个进程做 YOLO/手部推理与跟踪
3. 一个进程负责激光控制
4. 一个进程负责 WebUI 文件输出
"""

from __future__ import annotations

import argparse
import ctypes
import json
import multiprocessing as mp
import os
import queue
import sys
import time
from multiprocessing import shared_memory
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import unified_monitor as um  # noqa: E402
import control_plane as cp  # noqa: E402


FRAME_WIDTH = 640
FRAME_HEIGHT = 480
COLOR_SHAPE = (FRAME_HEIGHT, FRAME_WIDTH, 3)
DEPTH_SHAPE = (FRAME_HEIGHT, FRAME_WIDTH)
OVERLAY_SHAPE = (FRAME_HEIGHT, FRAME_WIDTH, 4)
TEXT_SLOT_SIZE = 4096


def build_webui_artifact_plan(write_debug_assets: bool) -> Tuple[str, ...]:
    return um.build_webui_artifact_plan(write_debug_assets)


def emit_marker_command(galvo, style: str, box: Sequence[float], *, frame_width: int, frame_height: int) -> bool:
    if style == "circle":
        x1, y1, x2, y2 = [float(v) for v in box]
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        radius_px = max(1.0, max(abs(x2 - x1), abs(y2 - y1)) / 2.0)
        gx, gy = galvo.pixel_to_galvo(center_x, center_y, frame_width, frame_height)
        radius = int(max(1.0, radius_px * 102.4))
        return galvo.draw_circle(gx, gy, radius, task_index=0)
    return galvo.draw_box(
        box,
        pixel_coords=True,
        task_index=0,
        image_width=frame_width,
        image_height=frame_height,
        steps_per_edge=15,
    )


def _resolve_marker_style(marker_state: dict, obj: dict) -> Tuple[bool, str]:
    global_style = marker_state.get("marker_style", "rectangle")
    if global_style not in cp.VALID_MARKER_STYLES:
        global_style = "rectangle"
    class_config = marker_state.get("class_config")
    if not isinstance(class_config, dict):
        return True, global_style
    entry = class_config.get(str(obj.get("name") or ""))
    if not isinstance(entry, dict):
        return True, global_style
    if not bool(entry.get("enabled", True)):
        return False, global_style
    style = entry.get("style", global_style)
    if style not in cp.VALID_MARKER_STYLES:
        style = global_style
    return True, style


def create_shared_buffer(shape: Tuple[int, ...], dtype: np.dtype):
    size = int(np.prod(shape) * np.dtype(dtype).itemsize)
    shm = shared_memory.SharedMemory(create=True, size=size)
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    arr.fill(0)
    return shm


def attach_shared_buffer(name: str, shape: Tuple[int, ...], dtype: np.dtype):
    shm = shared_memory.SharedMemory(name=name)
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    return shm, arr


class RuntimeSharedState:
    def __init__(
        self,
        *,
        camera_fps,
        depth_scale,
        laser_enabled,
        laser_marked,
        laser_active,
        camera_serial,
        hand_backend,
        locked_track_ids,
        fatal_error,
        laser_error,
        homography_matrix,
        worker_cpu_map,
        worker_pids,
    ):
        self.camera_fps = camera_fps
        self.depth_scale = depth_scale
        self.laser_enabled = laser_enabled
        self.laser_marked = laser_marked
        self.laser_active = laser_active
        self.camera_serial = camera_serial
        self.hand_backend = hand_backend
        self.locked_track_ids = locked_track_ids
        self.fatal_error = fatal_error
        self.laser_error = laser_error
        self.homography_matrix = homography_matrix
        self.worker_cpu_map = worker_cpu_map
        self.worker_pids = worker_pids


def create_text_slot(ctx, size: int = TEXT_SLOT_SIZE):
    return ctx.Array(ctypes.c_char, size, lock=True)


def _slot_buffer(slot):
    return slot.get_obj() if hasattr(slot, "get_obj") else slot


def write_text_slot(slot, text: Optional[str]) -> None:
    raw = _slot_buffer(slot)
    encoded = (text or "").encode("utf-8")
    max_len = max(0, len(raw) - 1)
    if len(encoded) > max_len:
        encoded = encoded[:max_len]
    with slot.get_lock():
        raw[: len(raw)] = b"\0" * len(raw)
        if encoded:
            raw[: len(encoded)] = encoded


def read_text_slot(slot) -> str:
    with slot.get_lock():
        raw = bytes(_slot_buffer(slot)[:])
    return raw.split(b"\0", 1)[0].decode("utf-8", errors="ignore")


def write_json_slot(slot, value) -> None:
    write_text_slot(slot, json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def read_json_slot(slot, default):
    text = read_text_slot(slot)
    if not text:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default


def set_shared_value(shared_value, value) -> None:
    with shared_value.get_lock():
        shared_value.value = value


def get_shared_value(shared_value):
    return shared_value.value


def set_shared_bool(shared_value, value: bool) -> None:
    set_shared_value(shared_value, 1 if value else 0)


def get_shared_bool(shared_value) -> bool:
    return bool(get_shared_value(shared_value))


def create_runtime_shared_state(ctx, *, laser_enabled: bool) -> RuntimeSharedState:
    runtime = RuntimeSharedState(
        camera_fps=ctx.Value("d", 0.0),
        depth_scale=ctx.Value("d", 0.001),
        laser_enabled=ctx.Value("b", 1 if laser_enabled else 0),
        laser_marked=ctx.Value("b", 0),
        laser_active=ctx.Value("b", 0),
        camera_serial=create_text_slot(ctx, 128),
        hand_backend=create_text_slot(ctx, 64),
        locked_track_ids=create_text_slot(ctx, 256),
        fatal_error=create_text_slot(ctx, 1024),
        laser_error=create_text_slot(ctx, 1024),
        homography_matrix=create_text_slot(ctx, 2048),
        worker_cpu_map=create_text_slot(ctx, 256),
        worker_pids=create_text_slot(ctx, 256),
    )
    write_json_slot(runtime.locked_track_ids, [])
    write_json_slot(runtime.worker_cpu_map, {})
    write_json_slot(runtime.worker_pids, {})
    write_json_slot(runtime.homography_matrix, None)
    return runtime


def list_available_cpus() -> List[int]:
    if hasattr(os, "sched_getaffinity"):
        return sorted(int(cpu) for cpu in os.sched_getaffinity(0))
    count = os.cpu_count() or 1
    return list(range(count))


def plan_worker_cpus(
    worker_names: Sequence[str],
    available_cpus: Optional[Sequence[int]] = None,
) -> Dict[str, Optional[int]]:
    cpus = list(available_cpus) if available_cpus is not None else list_available_cpus()
    if not cpus:
        return {name: None for name in worker_names}
    return {name: cpus[index % len(cpus)] for index, name in enumerate(worker_names)}


def apply_worker_cpu_affinity(worker_name: str, cpu_id: Optional[int]) -> None:
    if cpu_id is None or not hasattr(os, "sched_setaffinity"):
        return
    try:
        os.sched_setaffinity(0, {int(cpu_id)})
        print(f"[MP] {worker_name} pinned to cpu{cpu_id}", flush=True)
    except Exception as exc:
        print(f"[MP] {worker_name} pin cpu{cpu_id} failed: {exc!r}", flush=True)


def put_latest_message(target_queue, payload) -> bool:
    while True:
        try:
            target_queue.put_nowait(payload)
            return True
        except queue.Full:
            try:
                target_queue.get_nowait()
            except queue.Empty:
                return False


def drain_latest_message(source_queue, latest_payload=None):
    updated = False
    while True:
        try:
            latest_payload = source_queue.get_nowait()
            updated = True
        except queue.Empty:
            return latest_payload, updated


def build_default_detection_snapshot() -> dict:
    return {
        "fps": 0.0,
        "yolo_ms": 0.0,
        "hand_ms": 0.0,
        "objects": 0,
        "object_details": [],
        "laser_objects": [],
        "hands": 0,
        "is_danger": False,
        "min_depth_mm": None,
        "danger_object": None,
        "ts": time.time(),
    }


def serialize_objects_for_state(objects: Sequence[dict], locked_track_ids: Iterable[int]):
    locked = set(locked_track_ids)
    object_details = []
    laser_objects = []
    for obj in objects:
        serializable = {
            "track_id": obj.get("track_id"),
            "name": obj.get("name"),
            "score": float(obj.get("score", 0.0)),
            "cls": int(obj.get("cls", -1)) if obj.get("cls") is not None else None,
            "box": [float(v) for v in obj.get("box", [])],
        }
        laser_objects.append(serializable)
    for obj in laser_objects[:10]:
        object_details.append(
            {
                "track_id": obj.get("track_id"),
                "name": obj.get("name"),
                "score": round(float(obj.get("score", 0.0)), 4),
                "box": [int(round(v)) for v in obj.get("box", [])],
                "locked": obj.get("track_id") in locked,
            }
        )
    return object_details, laser_objects


class DepthFrameProxy:
    def __init__(self, depth_map: np.ndarray, depth_scale: float):
        self.depth_map = depth_map
        self.depth_scale = depth_scale

    def get_distance(self, x: int, y: int) -> float:
        if x < 0 or y < 0 or y >= self.depth_map.shape[0] or x >= self.depth_map.shape[1]:
            return 0.0
        return float(self.depth_map[y, x]) * self.depth_scale


def capture_worker(color_name: str, depth_name: str, frame_seq, runtime, frame_ready_event, stop_event, worker_cpu, preferred_camera_serial: str):
    import pyrealsense2 as rs

    apply_worker_cpu_affinity("camera_capture", worker_cpu)
    color_shm, color_buf = attach_shared_buffer(color_name, COLOR_SHAPE, np.uint8)
    depth_shm, depth_buf = attach_shared_buffer(depth_name, DEPTH_SHAPE, np.uint16)

    pipeline = rs.pipeline()
    config = rs.config()
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        write_text_slot(runtime.fatal_error, "未找到 RealSense 设备")
        stop_event.set()
        return
    try:
        serial = um.resolve_camera_serial(devices, rs, preferred_camera_serial)
    except Exception as exc:
        write_text_slot(runtime.fatal_error, str(exc))
        stop_event.set()
        return
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, FRAME_WIDTH, FRAME_HEIGHT, rs.format.z16, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())

    color_sensor = profile.get_device().first_color_sensor()
    color_sensor.set_option(rs.option.enable_auto_exposure, 1)
    exposure_range = color_sensor.get_option_range(rs.option.exposure)
    target_exposure = max(exposure_range.min, min(exposure_range.max, 33333))
    color_sensor.set_option(rs.option.enable_auto_exposure, 0)
    color_sensor.set_option(rs.option.exposure, target_exposure)
    time.sleep(0.1)
    color_sensor.set_option(rs.option.enable_auto_exposure, 1)

    write_text_slot(runtime.camera_serial, serial)
    set_shared_value(runtime.depth_scale, depth_scale)

    frame_counter = 0
    last_ts = time.time()
    try:
        while not stop_event.is_set():
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_np = np.asanyarray(color_frame.get_data())
            depth_np = np.asanyarray(depth_frame.get_data())
            np.copyto(color_buf, color_np)
            np.copyto(depth_buf, depth_np)
            with frame_seq.get_lock():
                frame_seq.value += 1
            frame_ready_event.set()

            frame_counter += 1
            now = time.time()
            dt = now - last_ts
            if dt >= 1.0:
                set_shared_value(runtime.camera_fps, frame_counter / dt)
                frame_counter = 0
                last_ts = now
    finally:
        pipeline.stop()
        color_shm.close()
        depth_shm.close()


def detection_worker(
    args,
    names,
    color_name: str,
    depth_name: str,
    annotated_name: str,
    overlay_name: str,
    frame_seq,
    detect_seq,
    runtime,
    writer_queue,
    laser_queue,
    frame_ready_event,
    stop_event,
    detect_ready_event,
    worker_cpu,
):
    apply_worker_cpu_affinity("detector", worker_cpu)
    color_shm, color_buf = attach_shared_buffer(color_name, COLOR_SHAPE, np.uint8)
    depth_shm, depth_buf = attach_shared_buffer(depth_name, DEPTH_SHAPE, np.uint16)
    annotated_shm, annotated_buf = attach_shared_buffer(annotated_name, COLOR_SHAPE, np.uint8)
    overlay_shm, overlay_buf = attach_shared_buffer(overlay_name, OVERLAY_SHAPE, np.uint8)

    yolo_backend = um.detect_yolo_backend(args.yolo_model)
    needs_acl = yolo_backend == "acl" or args.pose_model is not None

    res = None
    if needs_acl:
        res = um.AclLiteResource()
        res.init()

    if yolo_backend == "acl":
        yolo_model = um.AclLiteModel(args.yolo_model)
        yolo_model.load()
        yolo_model.context = res.context
    else:
        yolo_model = um.UltralyticsYoloModel(args.yolo_model, device=args.yolo_device)
        yolo_model.load()

    pose_model = None
    hands = None
    if args.disable_hand:
        write_text_slot(runtime.hand_backend, "disabled")
    elif args.pose_model:
        pose_model = um.AclLiteModel(args.pose_model)
        pose_model.load()
        pose_model.context = res.context
        write_text_slot(runtime.hand_backend, "pose")
    else:
        if um.mp is None:
            write_text_slot(runtime.fatal_error, "未找到 mediapipe 且未提供 pose-model")
            stop_event.set()
            return
        mp_hands = um.mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        write_text_slot(runtime.hand_backend, "mediapipe")

    last_seq = -1
    last_hand_results = None
    track_state: Dict[int, dict] = {}
    next_track_id = 1

    try:
        while not stop_event.is_set():
            if not frame_ready_event.wait(timeout=0.1):
                continue
            seq = int(frame_seq.value)
            if seq <= last_seq:
                if int(frame_seq.value) <= last_seq:
                    frame_ready_event.clear()
                continue

            frame = color_buf.copy()
            depth_map = depth_buf.copy()
            seq_check = int(frame_seq.value)
            if seq != seq_check:
                continue
            last_seq = seq
            if int(frame_seq.value) <= last_seq:
                frame_ready_event.clear()

            h_orig, w_orig = frame.shape[:2]
            seq_captured = seq
            if yolo_backend == "acl":
                img_yolo = um.prepare_acl_yolo_input(frame, args.yolo_acl_input_format)
                yolo_t0 = time.time()
                yolo_outputs = yolo_model.execute(img_yolo)
                yolo_ms = (time.time() - yolo_t0) * 1000.0
                objects: List[dict] = []
                if yolo_outputs:
                    total = yolo_outputs[0].size
                    if total == 300 * 6 or (total % 300 == 0 and total // 300 <= 7):
                        objects = um.postprocess_yolov10(
                            yolo_outputs[0], frame.shape, conf_thres=args.conf_thres, names=names
                        )
                    else:
                        ratio = min(640 / w_orig, 640 / h_orig)
                        dw = (640 - int(w_orig * ratio)) / 2
                        dh = (640 - int(h_orig * ratio)) / 2
                        objects = um.postprocess_yolo(
                            yolo_outputs[0],
                            ratio,
                            (dw, dh),
                            nc=len(names),
                            conf_thres=args.conf_thres,
                            names=names,
                        )
            else:
                yolo_t0 = time.time()
                objects = yolo_model.infer(frame, args.conf_thres)
                yolo_ms = (time.time() - yolo_t0) * 1000.0

            # 跳帧：若推理期间相机已更新 ≥2 帧，丢弃过时结果并立即处理最新帧
            if int(frame_seq.value) >= seq_captured + 2:
                continue

            objects, track_state, next_track_id = um.assign_track_ids(
                objects, track_state, next_track_id, iou_thres=args.track_iou_thres, max_missing=args.track_max_missing
            )

            is_danger = False
            min_depth_mm = None
            hand_count = 0
            danger_obj = None
            depth_frame = DepthFrameProxy(depth_map, float(get_shared_value(runtime.depth_scale)))

            if args.disable_hand:
                hand_ms = 0.0
            elif pose_model is not None:
                should_detect_pose = seq % (args.pose_infer_skip + 1) == 0
                pose_t0 = time.time()
                if should_detect_pose:
                    img_pose = cv2.resize(frame, (640, 640)).astype(np.uint8)
                    outputs = pose_model.execute(img_pose)
                    pose_results_new = um.postprocess_pose(outputs or [], frame.shape, conf_thres=args.pose_conf_thres)
                else:
                    pose_results_new = None
                hand_ms = (time.time() - pose_t0) * 1000.0
                if pose_results_new is not None:
                    last_hand_results = pose_results_new
                hand_results = last_hand_results if last_hand_results is not None else []
                hand_count = len(hand_results)
                for person in hand_results:
                    wrists = um.extract_pose_wrists(person, depth_frame, conf_thres=args.pose_conf_thres)
                    for wrist in wrists:
                        wx, wy = wrist["pixel"]
                        depth_mm = wrist["depth_mm"]
                        if depth_mm > 0 and (min_depth_mm is None or depth_mm < min_depth_mm):
                            min_depth_mm = depth_mm
                        if depth_mm > 0 and depth_mm < args.danger_distance:
                            is_danger = True
                            near_obj, obj = um.check_hand_near_objects((wx, wy), depth_mm, objects, args.danger_distance)
                            if near_obj:
                                danger_obj = obj
            else:
                should_detect_hand = seq % 3 == 0
                hand_t0 = time.time()
                if should_detect_hand:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    hand_results_new = hands.process(frame_rgb)
                else:
                    hand_results_new = None
                hand_ms = (time.time() - hand_t0) * 1000.0
                if hand_results_new is not None:
                    last_hand_results = hand_results_new
                hand_results = last_hand_results
                if hand_results is not None and hand_results.multi_hand_landmarks:
                    hand_count = len(hand_results.multi_hand_landmarks)
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        wrist = hand_landmarks.landmark[0]
                        wx, wy = int(wrist.x * w_orig), int(wrist.y * h_orig)
                        wx = max(0, min(w_orig - 1, wx))
                        wy = max(0, min(h_orig - 1, wy))
                        depth_mm = depth_frame.get_distance(wx, wy) * 1000.0
                        if depth_mm > 0 and (min_depth_mm is None or depth_mm < min_depth_mm):
                            min_depth_mm = depth_mm
                        if depth_mm > 0 and depth_mm < args.danger_distance:
                            is_danger = True
                            near_obj, obj = um.check_hand_near_objects((wx, wy), depth_mm, objects, args.danger_distance)
                            if near_obj:
                                danger_obj = obj

            locked_track_ids = set(read_json_slot(runtime.locked_track_ids, []))
            current_candidates = um.select_laser_targets(
                objects,
                min_score=args.laser_min_score,
                max_targets=args.max_laser_targets,
                allowed_classes=args.laser_target_classes,
                preferred_track_ids=locked_track_ids,
            )
            current_candidate_ids = {obj.get("track_id") for obj in current_candidates}
            highlighted_ids = locked_track_ids or current_candidate_ids

            annotated = frame.copy()
            if is_danger:
                text = f"DANGER {danger_obj['name']}" if danger_obj else "DANGER"
                cv2.putText(annotated, text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
            um.draw_object_annotations(annotated, objects, highlighted_track_ids=highlighted_ids)
            overlay = None
            if args.write_debug_assets:
                overlay = um.build_debug_overlay(w_orig, h_orig, objects, highlighted_track_ids=highlighted_ids)

            np.copyto(annotated_buf, annotated)
            if overlay is not None:
                np.copyto(overlay_buf, overlay)

            object_details, laser_objects = serialize_objects_for_state(objects, locked_track_ids)
            writer_snapshot = {
                "fps": float(get_shared_value(runtime.camera_fps)),
                "yolo_ms": float(yolo_ms),
                "hand_ms": float(hand_ms),
                "objects": len(objects),
                "object_details": object_details,
                "laser_objects": laser_objects,
                "hands": int(hand_count),
                "is_danger": bool(is_danger),
                "min_depth_mm": None if min_depth_mm is None else float(min_depth_mm),
                "danger_object": danger_obj["name"] if danger_obj else None,
                "ts": time.time(),
            }
            put_latest_message(writer_queue, writer_snapshot)
            put_latest_message(
                laser_queue,
                {
                    "laser_objects": laser_objects,
                    "is_danger": bool(is_danger),
                    "detect_ts": time.time(),
                },
            )
            with detect_seq.get_lock():
                detect_seq.value += 1
            detect_ready_event.set()
    finally:
        if hands is not None:
            hands.close()
        color_shm.close()
        depth_shm.close()
        annotated_shm.close()
        overlay_shm.close()


def laser_worker(args, runtime, laser_queue, stop_event, worker_cpu):
    apply_worker_cpu_affinity("laser_worker", worker_cpu)
    set_shared_bool(runtime.laser_enabled, False)
    set_shared_bool(runtime.laser_marked, False)
    set_shared_bool(runtime.laser_active, False)
    write_json_slot(runtime.locked_track_ids, [])
    write_text_slot(runtime.laser_error, "")
    if not args.enable_laser:
        while not stop_event.is_set():
            time.sleep(0.1)
        return

    from edge.laser_galvo.galvo_controller import LaserGalvoController

    galvo = LaserGalvoController(
        serial_port=args.laser_serial,
        baudrate=args.laser_baudrate,
        calibration_file=args.laser_calibration,
    )
    if not galvo.connect():
        write_text_slot(runtime.laser_error, f"激光振镜连接失败: {args.laser_serial}")
        return

    set_shared_bool(runtime.laser_enabled, True)
    last_laser_time = 0.0
    last_force_refresh_time = 0.0
    last_laser_box: Optional[List[float]] = None
    last_laser_track_id: Optional[int] = None
    laser_active = False
    locked_track_ids: Set[int] = set()
    latest_payload = None
    ict_root = Path.home() / "ICT"
    # 坐标外推状态：track_id → (center_x, center_y, timestamp)
    _extrap_state: Dict[int, Tuple[float, float, float]] = {}
    # marker_state TTL 缓存，避免热循环每帧读文件
    _marker_state_cache: dict = {}
    _marker_state_ts: float = 0.0

    try:
        while not stop_event.is_set():
            now = time.time()
            timeout = 0.05
            if laser_active:
                remaining = args.laser_force_refresh_interval - (now - last_force_refresh_time)
                timeout = max(0.01, min(0.05, remaining))
            try:
                latest_payload = laser_queue.get(timeout=timeout)
                latest_payload, _ = drain_latest_message(laser_queue, latest_payload)
                has_new_detect = True
            except queue.Empty:
                has_new_detect = False

            now = time.time()
            if latest_payload is None:
                continue
            if not has_new_detect and (not laser_active or (now - last_force_refresh_time) < args.laser_force_refresh_interval):
                continue

            objects = list(latest_payload.get("laser_objects", []))
            is_danger = bool(latest_payload.get("is_danger", False))
            if now - _marker_state_ts > 0.1:
                _marker_state_cache = cp.load_marker_state(ict_root)
                _marker_state_ts = now
            marker_state = _marker_state_cache
            # 外推状态清理：无论有无候选目标都执行，避免无候选帧时长期滞留
            stale = [k for k, (_, _, ts) in _extrap_state.items() if now - ts > 5.0]
            for k in stale:
                del _extrap_state[k]
            selected_track_id = marker_state.get("selected_track_id")
            preferred_track_ids = locked_track_ids
            if selected_track_id is not None:
                preferred_track_ids = {int(selected_track_id)}
            eligible_objects = [o for o in objects if _resolve_marker_style(marker_state, o)[0]]
            current_candidates = um.select_laser_targets(
                eligible_objects,
                min_score=args.laser_min_score,
                max_targets=args.max_laser_targets,
                allowed_classes=args.laser_target_classes,
                preferred_track_ids=preferred_track_ids,
            )

            laser_marked = False
            if now - last_laser_time >= args.laser_update_interval:
                if current_candidates and not is_danger:
                    obj = current_candidates[0]
                    target_track_id = obj["track_id"]
                    target_box = um.expand_box(obj["box"], args.laser_box_scale)

                    # 坐标外推：补偿推理延迟导致的目标位置滞后
                    detect_ts = latest_payload.get("detect_ts", now)
                    cx = (target_box[0] + target_box[2]) / 2.0
                    cy = (target_box[1] + target_box[3]) / 2.0
                    if target_track_id in _extrap_state:
                        prev_cx, prev_cy, prev_ts = _extrap_state[target_track_id]
                        dt_detect = detect_ts - prev_ts
                        if 0 < dt_detect < 0.5:
                            vx = (cx - prev_cx) / dt_detect
                            vy = (cy - prev_cy) / dt_detect
                            if abs(vx) <= 600 and abs(vy) <= 600:
                                dt_lag = max(0.0, now - detect_ts)
                                sx, sy = vx * dt_lag, vy * dt_lag
                                target_box = [
                                    target_box[0] + sx, target_box[1] + sy,
                                    target_box[2] + sx, target_box[3] + sy,
                                ]
                    _extrap_state[target_track_id] = (cx, cy, detect_ts)

                    if last_laser_box is not None and last_laser_track_id == target_track_id:
                        target_box = um.smooth_box(last_laser_box, target_box, args.laser_smoothing_alpha)

                    needs_refresh = (
                        not laser_active
                        or last_laser_box is None
                        or last_laser_track_id != target_track_id
                        or um.should_refresh_laser_box(
                            last_laser_box,
                            target_box,
                            center_threshold_px=args.laser_center_threshold_px,
                            size_threshold_ratio=args.laser_size_threshold_ratio,
                        )
                        or (now - last_force_refresh_time) >= args.laser_force_refresh_interval
                    )
                    if needs_refresh:
                        try:
                            galvo.task_index = 0
                            _, effective_marker_style = _resolve_marker_style(marker_state, obj)
                            galvo.begin_batch()
                            emit_marker_command(
                                galvo,
                                effective_marker_style,
                                target_box,
                                frame_width=FRAME_WIDTH,
                                frame_height=FRAME_HEIGHT,
                            )
                            galvo.update_tasks()  # 批量刷新：cmd + U; 合并单次写入
                            laser_marked = True
                            laser_active = True
                            locked_track_ids = {target_track_id}
                            last_laser_box = list(target_box)
                            last_laser_track_id = target_track_id
                            last_laser_time = now
                            last_force_refresh_time = now
                        except Exception as exc:
                            write_text_slot(runtime.laser_error, repr(exc))
                    else:
                        laser_active = True
                        locked_track_ids = {target_track_id}
                elif laser_active:
                    try:
                        galvo.update_tasks()
                    except Exception as exc:
                        write_text_slot(runtime.laser_error, repr(exc))
                    laser_active = False
                    laser_marked = False
                    locked_track_ids = set()
                    last_laser_box = None
                    last_laser_track_id = None
                    last_laser_time = now

            set_shared_bool(runtime.laser_marked, laser_marked)
            set_shared_bool(runtime.laser_active, laser_active)
            write_json_slot(runtime.locked_track_ids, sorted(locked_track_ids))
    finally:
        try:
            galvo.disconnect()
        except Exception:
            pass


def _safe_write_bytes(path: Path, data: bytes):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def writer_worker(
    color_name: str,
    annotated_name: str,
    overlay_name: str,
    frame_seq,
    detect_seq,
    detect_ready_event,
    runtime,
    names,
    writer_queue,
    stop_event,
    writer_fps: float,
    worker_cpu,
    write_debug_assets: bool,
    config_snapshot: dict,
    ws_port: int = 8003,
):
    apply_worker_cpu_affinity("writer", worker_cpu)
    color_shm, color_buf = attach_shared_buffer(color_name, COLOR_SHAPE, np.uint8)
    annotated_shm, annotated_buf = attach_shared_buffer(annotated_name, COLOR_SHAPE, np.uint8)
    overlay_shm, overlay_buf = attach_shared_buffer(overlay_name, OVERLAY_SHAPE, np.uint8)

    web_dir = Path.home() / "ICT" / "webui_http_unified"
    web_dir.mkdir(parents=True, exist_ok=True)
    webui_artifacts = build_webui_artifact_plan(write_debug_assets)
    um.cleanup_optional_webui_artifacts(web_dir, webui_artifacts)
    ict_root = Path.home() / "ICT"

    # --- DVPP + WebSocket encoder init (three-tier graceful degradation) ---
    jpege = venc = broadcaster = None
    try:
        import acl as _acl
        _acl.init()
        _acl.rt.set_device(0)
        _acl.rt.create_context(0)
    except Exception:
        pass
    try:
        from dvpp_stream import DvppJpegEncoder
        jpege = DvppJpegEncoder(FRAME_WIDTH, FRAME_HEIGHT)
        print("[writer] DVPP JPEGE encoder ready", flush=True)
    except Exception as e:
        print(f"[writer] DVPP JPEGE unavailable, fallback cv2: {e}", flush=True)
    try:
        from dvpp_stream import VencStreamEncoder
        venc = VencStreamEncoder(FRAME_WIDTH, FRAME_HEIGHT)
        print("[writer] VENC H.264 encoder ready", flush=True)
    except Exception as e:
        print(f"[writer] VENC H.264 unavailable: {e}", flush=True)
    try:
        from dvpp_stream import WebSocketBroadcaster
        broadcaster = WebSocketBroadcaster(
            width=FRAME_WIDTH, height=FRAME_HEIGHT,
            fps=int(max(writer_fps, 1)), port=ws_port,
        )
        broadcaster.set_codec("h264" if venc else "mjpeg")
        broadcaster.start()
        print(f"[writer] WebSocket broadcaster on port {ws_port} (codec={'h264' if venc else 'mjpeg'})", flush=True)
    except Exception as e:
        print(f"[writer] WebSocket broadcaster unavailable: {e}", flush=True)

    min_interval = 1.0 / max(writer_fps, 1.0)
    next_write_time = 0.0
    latest_snapshot = build_default_detection_snapshot()
    last_written_detect_seq = -1

    try:
        while not stop_event.is_set():
            remaining = max(0.001, next_write_time - time.time()) if next_write_time else 0.1
            woke = detect_ready_event.wait(timeout=remaining)
            if woke:
                detect_ready_event.clear()
            now = time.time()
            if next_write_time and now < next_write_time:
                continue

            current_detect_seq = detect_seq.value
            if current_detect_seq == last_written_detect_seq:
                continue

            latest_snapshot, _ = drain_latest_message(writer_queue, latest_snapshot)

            raw_frame = color_buf.copy() if write_debug_assets else None
            annotated = annotated_buf.copy()
            jpeg_bytes = None
            if jpege:
                try:
                    jpeg_bytes = jpege.encode(annotated)
                    _safe_write_bytes(web_dir / "frame.jpg", jpeg_bytes)
                except Exception:
                    jpeg_bytes = None
            if jpeg_bytes is None:
                cv2.imwrite(str(web_dir / "frame.jpg"), annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if write_debug_assets and raw_frame is not None:
                overlay = overlay_buf.copy()
                cv2.imwrite(str(web_dir / "debug_frame.jpg"), raw_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                cv2.imwrite(str(web_dir / "debug_overlay.png"), overlay, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            state_payload = {
                "names": list(names),
                **latest_snapshot,
                "laser_enabled": get_shared_bool(runtime.laser_enabled),
                "laser_marked": get_shared_bool(runtime.laser_marked),
                "laser_active": get_shared_bool(runtime.laser_active),
                "locked_track_ids": read_json_slot(runtime.locked_track_ids, []),
                "frame_width": FRAME_WIDTH,
                "frame_height": FRAME_HEIGHT,
                "camera_serial": read_text_slot(runtime.camera_serial),
                "depth_scale": float(get_shared_value(runtime.depth_scale)),
                "capture_seq": int(frame_seq.value),
                "detect_seq": current_detect_seq,
                "hand_backend": read_text_slot(runtime.hand_backend),
                "camera_fps": float(get_shared_value(runtime.camera_fps)),
                "worker_cpu_map": read_json_slot(runtime.worker_cpu_map, {}),
                "worker_pids": read_json_slot(runtime.worker_pids, {}),
                "write_debug_assets": bool(write_debug_assets),
                "homography_matrix": read_json_slot(runtime.homography_matrix, None),
                "config": dict(config_snapshot),
                "marker_state": cp.load_marker_state(ict_root),
            }
            fatal_error = read_text_slot(runtime.fatal_error)
            if fatal_error:
                state_payload["fatal_error"] = fatal_error
            laser_error = read_text_slot(runtime.laser_error)
            if laser_error:
                state_payload["laser_error"] = laser_error
            with open(web_dir / "state.json", "w", encoding="utf-8") as handle:
                json.dump(state_payload, handle)
            state_json_str = json.dumps(state_payload) if broadcaster else None

            # --- WebSocket broadcast ---
            if broadcaster:
                if venc:
                    try:
                        h264 = venc.encode(annotated)
                        if h264:
                            broadcaster.broadcast_binary(h264)
                    except Exception:
                        pass
                elif jpeg_bytes is not None:
                    broadcaster.broadcast_binary(jpeg_bytes)
                broadcaster.broadcast_text(state_json_str)

            last_written_detect_seq = current_detect_seq
            next_write_time = now + min_interval
    finally:
        if broadcaster:
            broadcaster.stop()
        if venc:
            venc.close()
        if jpege:
            jpege.close()
        color_shm.close()
        annotated_shm.close()
        overlay_shm.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = um.build_arg_parser()
    parser.add_argument("--writer-fps", type=float, default=30.0, help="WebUI 写图与状态输出频率")
    parser.add_argument("--ws-port", type=int, default=8003, help="WebSocket 推流端口")
    parser.add_argument("--cpu-affinity", choices=("auto", "off"), default="auto", help="Linux 下自动为采集/检测/写盘/激光分配 CPU 核")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.enable_laser and not Path(args.laser_serial).exists():
        print(f"警告: 串口设备不存在，激光链路将不可用: {args.laser_serial}", flush=True)

    ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else mp.get_start_method())
    stop_event = ctx.Event()
    frame_ready_event = ctx.Event()
    frame_seq = ctx.Value("i", 0)
    detect_seq = ctx.Value("i", 0)
    detect_ready_event = ctx.Event()
    writer_queue = ctx.Queue(maxsize=1)
    laser_queue = ctx.Queue(maxsize=1)

    with open(args.data_yaml, "r", encoding="utf-8") as handle:
        data_cfg = json.loads(json.dumps(__import__("yaml").safe_load(handle)))
    names = list(data_cfg.get("names", []))
    runtime = create_runtime_shared_state(ctx, laser_enabled=bool(args.enable_laser))
    write_text_slot(runtime.hand_backend, "pending")
    config_snapshot = {
        "danger_distance": args.danger_distance,
        "conf_thres": args.conf_thres,
        "yolo_model": str(Path(args.yolo_model).name),
        "pose_model": str(args.pose_model or ""),
        "data_yaml": str(args.data_yaml),
        "camera_serial": str(args.camera_serial or ""),
        "enable_laser": bool(args.enable_laser),
        "laser_serial": str(args.laser_serial),
        "laser_baudrate": int(args.laser_baudrate),
        "laser_calibration": str(args.laser_calibration or ""),
    }

    # Load calibration matrix for WebUI
    if args.laser_calibration and Path(args.laser_calibration).exists():
        import yaml
        try:
            with open(args.laser_calibration, "r") as f:
                calib_data = yaml.safe_load(f)
                if "homography_matrix" in calib_data:
                    write_json_slot(runtime.homography_matrix, calib_data["homography_matrix"])
        except Exception as e:
            print(f"警告: 加载 WebUI 标定矩阵失败: {e}", flush=True)

    color_shm = create_shared_buffer(COLOR_SHAPE, np.uint8)
    depth_shm = create_shared_buffer(DEPTH_SHAPE, np.uint16)
    annotated_shm = create_shared_buffer(COLOR_SHAPE, np.uint8)
    overlay_shm = create_shared_buffer(OVERLAY_SHAPE, np.uint8)

    worker_names = ["camera_capture", "detector", "writer", "laser_worker"]
    cpu_plan = plan_worker_cpus(worker_names) if args.cpu_affinity == "auto" else {name: None for name in worker_names}
    write_json_slot(runtime.worker_cpu_map, cpu_plan)
    print(f"[MP] cpu plan: {cpu_plan}", flush=True)

    processes = [
        ctx.Process(
            target=capture_worker,
            name="camera_capture",
            args=(color_shm.name, depth_shm.name, frame_seq, runtime, frame_ready_event, stop_event, cpu_plan["camera_capture"], args.camera_serial),
        ),
        ctx.Process(
            target=detection_worker,
            name="detector",
            args=(
                args,
                names,
                color_shm.name,
                depth_shm.name,
                annotated_shm.name,
                overlay_shm.name,
                frame_seq,
                detect_seq,
                runtime,
                writer_queue,
                laser_queue,
                frame_ready_event,
                stop_event,
                detect_ready_event,
                cpu_plan["detector"],
            ),
        ),
        ctx.Process(
            target=writer_worker,
            name="writer",
            args=(
                color_shm.name,
                annotated_shm.name,
                overlay_shm.name,
                frame_seq,
                detect_seq,
                detect_ready_event,
                runtime,
                names,
                writer_queue,
                stop_event,
                args.writer_fps,
                cpu_plan["writer"],
                args.write_debug_assets,
                config_snapshot,
                args.ws_port,
            ),
        ),
        ctx.Process(
            target=laser_worker,
            name="laser_worker",
            args=(args, runtime, laser_queue, stop_event, cpu_plan["laser_worker"]),
        ),
    ]

    pid_map = {}
    for process in processes:
        process.start()
        pid_map[process.name] = process.pid
        print(f"[MP] started {process.name} pid={process.pid}", flush=True)
    write_json_slot(runtime.worker_pids, pid_map)

    try:
        while not stop_event.is_set():
            dead = [p for p in processes if not p.is_alive()]
            if dead:
                write_text_slot(runtime.fatal_error, f"worker_exit: {[p.name for p in dead]}")
                break
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n停止多进程统一监控...", flush=True)
    finally:
        stop_event.set()
        frame_ready_event.set()
        detect_ready_event.set()
        for process in processes:
            process.join(timeout=5.0)
        for process in processes:
            if process.is_alive():
                process.terminate()
        for work_queue in (writer_queue, laser_queue):
            try:
                work_queue.close()
            except Exception:
                pass
        for shm in (color_shm, depth_shm, annotated_shm, overlay_shm):
            try:
                shm.close()
            except Exception:
                pass
            try:
                shm.unlink()
            except Exception:
                pass


if __name__ == "__main__":
    main()
