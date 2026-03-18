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


FRAME_WIDTH = 640
FRAME_HEIGHT = 480
COLOR_SHAPE = (FRAME_HEIGHT, FRAME_WIDTH, 3)
DEPTH_SHAPE = (FRAME_HEIGHT, FRAME_WIDTH)
OVERLAY_SHAPE = (FRAME_HEIGHT, FRAME_WIDTH, 4)
TEXT_SLOT_SIZE = 4096


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
        worker_cpu_map=create_text_slot(ctx, 256),
        worker_pids=create_text_slot(ctx, 256),
    )
    write_json_slot(runtime.locked_track_ids, [])
    write_json_slot(runtime.worker_cpu_map, {})
    write_json_slot(runtime.worker_pids, {})
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


def capture_worker(color_name: str, depth_name: str, frame_seq, runtime, frame_ready_event, stop_event, worker_cpu):
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
    serial = devices[0].get_info(rs.camera_info.serial_number)
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
            overlay = um.build_debug_overlay(w_orig, h_orig, objects, highlighted_track_ids=highlighted_ids)

            np.copyto(annotated_buf, annotated)
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
                },
            )
            with detect_seq.get_lock():
                detect_seq.value += 1
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
            current_candidates = um.select_laser_targets(
                objects,
                min_score=args.laser_min_score,
                max_targets=args.max_laser_targets,
                allowed_classes=args.laser_target_classes,
                preferred_track_ids=locked_track_ids,
            )

            laser_marked = False
            if now - last_laser_time >= args.laser_update_interval:
                if current_candidates and not is_danger:
                    obj = current_candidates[0]
                    target_track_id = obj["track_id"]
                    target_box = um.expand_box(obj["box"], args.laser_box_scale)
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
                            galvo.draw_box(
                                target_box,
                                pixel_coords=True,
                                task_index=0,
                                image_width=FRAME_WIDTH,
                                image_height=FRAME_HEIGHT,
                                steps_per_edge=15,
                            )
                            galvo.update_tasks()
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


def writer_worker(
    color_name: str,
    annotated_name: str,
    overlay_name: str,
    frame_seq,
    runtime,
    names,
    writer_queue,
    stop_event,
    writer_fps: float,
    worker_cpu,
):
    apply_worker_cpu_affinity("writer", worker_cpu)
    color_shm, color_buf = attach_shared_buffer(color_name, COLOR_SHAPE, np.uint8)
    annotated_shm, annotated_buf = attach_shared_buffer(annotated_name, COLOR_SHAPE, np.uint8)
    overlay_shm, overlay_buf = attach_shared_buffer(overlay_name, OVERLAY_SHAPE, np.uint8)

    web_dir = Path.home() / "ICT" / "webui_http_unified"
    web_dir.mkdir(parents=True, exist_ok=True)

    min_interval = 1.0 / max(writer_fps, 1.0)
    next_write_time = 0.0
    latest_snapshot = build_default_detection_snapshot()

    try:
        while not stop_event.is_set():
            timeout = 0.1 if next_write_time == 0.0 else max(0.01, next_write_time - time.time())
            try:
                latest_snapshot = writer_queue.get(timeout=timeout)
                latest_snapshot, _ = drain_latest_message(writer_queue, latest_snapshot)
            except queue.Empty:
                pass

            if next_write_time and time.time() < next_write_time:
                continue

            raw_frame = color_buf.copy()
            annotated = annotated_buf.copy()
            overlay = overlay_buf.copy()
            cv2.imwrite(str(web_dir / "frame.jpg"), annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
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
                "hand_backend": read_text_slot(runtime.hand_backend),
                "camera_fps": float(get_shared_value(runtime.camera_fps)),
                "worker_cpu_map": read_json_slot(runtime.worker_cpu_map, {}),
                "worker_pids": read_json_slot(runtime.worker_pids, {}),
            }
            fatal_error = read_text_slot(runtime.fatal_error)
            if fatal_error:
                state_payload["fatal_error"] = fatal_error
            laser_error = read_text_slot(runtime.laser_error)
            if laser_error:
                state_payload["laser_error"] = laser_error
            with open(web_dir / "state.json", "w", encoding="utf-8") as handle:
                json.dump(state_payload, handle)
            next_write_time = time.time() + min_interval
    finally:
        color_shm.close()
        annotated_shm.close()
        overlay_shm.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = um.build_arg_parser()
    parser.add_argument("--writer-fps", type=float, default=10.0, help="WebUI 写图与状态输出频率")
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
    writer_queue = ctx.Queue(maxsize=1)
    laser_queue = ctx.Queue(maxsize=1)

    with open(args.data_yaml, "r", encoding="utf-8") as handle:
        data_cfg = json.loads(json.dumps(__import__("yaml").safe_load(handle)))
    names = list(data_cfg.get("names", []))
    runtime = create_runtime_shared_state(ctx, laser_enabled=bool(args.enable_laser))
    write_text_slot(runtime.hand_backend, "pending")

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
            args=(color_shm.name, depth_shm.name, frame_seq, runtime, frame_ready_event, stop_event, cpu_plan["camera_capture"]),
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
                runtime,
                names,
                writer_queue,
                stop_event,
                args.writer_fps,
                cpu_plan["writer"],
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
