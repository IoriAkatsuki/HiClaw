#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
激光振镜自动标定程序
建立摄像头像素坐标与振镜坐标之间的映射关系。
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import serial
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from edge.common.calibration.camera_galvo_mapping import CameraIntrinsics, pixel_depth_to_cam


class GalvoCalibrator:
    """振镜标定器。"""

    INT16_MIN = -32768
    INT16_MAX = 32767

    def __init__(
        self,
        serial_port="/dev/ttyUSB0",
        baudrate=115200,
        laser_color="red",
        capture_attempts_per_point=10,
        settle_ms=500,
        min_valid_points=4,
        mean_error_thres=500.0,
        max_error_thres=1000.0,
        grid_size=3,
        x_range=15000,
        y_range=15000,
        grid_margin=5000,
        marker_radius=250,
        edge_margin_px=40,
        range_step=3000,
        range_refine_step=500,
        safe_limit=27767,
        diagnostic_file=None,
        detector_profile=None,
    ):
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.laser_color = laser_color
        self.capture_attempts_per_point = int(capture_attempts_per_point)
        self.settle_ms = int(settle_ms)
        self.min_valid_points = int(min_valid_points)
        self.mean_error_thres = float(mean_error_thres)
        self.max_error_thres = float(max_error_thres)
        self.grid_size = int(grid_size)
        self.x_range = int(x_range)
        self.y_range = int(y_range)
        self.grid_margin = int(grid_margin)
        self.marker_radius = int(marker_radius)
        self.edge_margin_px = int(edge_margin_px)
        self.range_step = int(range_step)
        self.range_refine_step = int(range_refine_step)
        self.safe_limit = int(safe_limit)
        self.diagnostic_file = diagnostic_file

        self.ser = None
        self.homography_matrix = None
        self.quality_metrics = {}
        self.last_failure_reason = ""
        self.image_resolution = (640, 480)
        self.last_marker_pose = None

        self.all_grid_points = []
        self.galvo_points = []
        self.valid_galvo_points = []
        self.pixel_points = []
        self.cam_points = []
        self.depth_points = []
        self.detection_history = []

        self.detector_profile = self._default_detector_profile()
        if detector_profile:
            self.detector_profile.update(detector_profile)

        self._init_galvo_grid()

    def _default_detector_profile(self):
        return {
            "hsv_red_low_1": [0, 65, 70],
            "hsv_red_high_1": [10, 255, 255],
            "hsv_red_low_2": [160, 65, 70],
            "hsv_red_high_2": [180, 255, 255],
            "diff_thresh": 24,
            "open_kernel": 3,
            "close_kernel": 5,
            "area_min": 6,
            "area_max": 1500,
            "min_circularity": 0.10,
            "min_peak_v": 150,
            "min_inlier_radius_px": 2.5,
            "ransac_reproj_threshold": 5.0,
        }

    def _init_galvo_grid(self):
        """初始化振镜标定网格。"""
        xs = np.linspace(-self.x_range, self.x_range, self.grid_size)
        ys = np.linspace(-self.y_range, self.y_range, self.grid_size)
        self.galvo_points = [
            (int(round(x)), int(round(y)))
            for y in ys
            for x in xs
        ]
        self.all_grid_points = list(self.galvo_points)
        print(f"✓ 初始化 {len(self.galvo_points)} 个振镜标定点")

    def connect_serial(self):
        """连接串口。"""
        try:
            self.ser = serial.Serial(
                port=self.serial_port,
                baudrate=self.baudrate,
                timeout=1.0,
                write_timeout=1.0,
            )
            time.sleep(2)
            print(f"✓ 串口已连接: {self.serial_port} @ {self.baudrate}")
            return True
        except Exception as exc:
            self.last_failure_reason = f"串口连接失败: {exc}"
            print(f"✗ {self.last_failure_reason}")
            return False

    def disconnect_serial(self):
        if self.ser is not None and self.ser.is_open:
            try:
                self.clear_tasks()
            except Exception:
                pass
            self.ser.close()
            self.ser = None

    def _write_command(self, command):
        if self.ser is None or not self.ser.is_open:
            self.last_failure_reason = "串口未连接"
            return False
        try:
            self.ser.write(command.encode("ascii"))
            self.ser.flush()
            return True
        except Exception as exc:
            self.last_failure_reason = f"命令发送失败: {exc}"
            return False

    def _send_packet(self, tokens):
        payload = ";".join(token.rstrip(";") for token in tokens if token) + ";"
        return self._write_command(payload)

    def clear_tasks(self):
        self.last_marker_pose = None
        return self._send_packet(["U"])

    def show_marker(self, x, y, radius=None, slot=0):
        """用一个小圆形任务在当前点位持续发光，供摄像头检测。"""
        x = int(np.clip(x, self.INT16_MIN, self.INT16_MAX))
        y = int(np.clip(y, self.INT16_MIN, self.INT16_MAX))
        radius = int(max(1, radius if radius is not None else self.marker_radius))
        self.last_marker_pose = (x, y, radius, int(slot))
        return self._send_packet([f"{slot}C,{x},{y},{radius}", "U"])

    def send_galvo_position(self, x, y):
        """兼容旧接口：新固件使用小圆标记点代替即时 G 命令。"""
        return self.show_marker(x, y, radius=self.marker_radius, slot=0)

    def enable_laser(self, enable=True):
        """兼容旧接口：新固件无法独立开关激光，只能通过任务显示/清空控制。"""
        if not enable:
            return self.clear_tasks()
        if self.last_marker_pose is None:
            self.last_failure_reason = "当前固件不支持独立开光，请先设置 marker pose"
            return False
        x, y, radius, slot = self.last_marker_pose
        return self.show_marker(x, y, radius=radius, slot=slot)

    def detect_laser_spot(self, frame=None, frame_on=None, frame_off=None, debug=False):
        """
        检测激光光斑位置。

        返回:
            {
                "pos": (x, y),
                "score": float,
                "area": float,
                "circularity": float,
            }
        """
        if frame_on is None:
            frame_on = frame
        if frame_on is None:
            raise ValueError("需要提供 frame 或 frame_on")

        hsv = cv2.cvtColor(frame_on, cv2.COLOR_BGR2HSV)
        prof = self.detector_profile

        lower_red1 = np.array(prof["hsv_red_low_1"], dtype=np.uint8)
        upper_red1 = np.array(prof["hsv_red_high_1"], dtype=np.uint8)
        lower_red2 = np.array(prof["hsv_red_low_2"], dtype=np.uint8)
        upper_red2 = np.array(prof["hsv_red_high_2"], dtype=np.uint8)

        mask_red = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red |= cv2.inRange(hsv, lower_red2, upper_red2)

        if frame_off is not None:
            diff = cv2.absdiff(frame_on, frame_off)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, diff_mask = cv2.threshold(
                diff_gray,
                int(prof["diff_thresh"]),
                255,
                cv2.THRESH_BINARY,
            )

            v_channel = hsv[:, :, 2]
            _, bright_mask = cv2.threshold(
                v_channel,
                int(prof["min_peak_v"]),
                255,
                cv2.THRESH_BINARY,
            )
            mask = (mask_red & diff_mask) | (bright_mask & diff_mask)
        else:
            mask = mask_red

        open_kernel = max(1, int(prof["open_kernel"]))
        close_kernel = max(1, int(prof["close_kernel"]))
        kernel_open = np.ones((open_kernel, open_kernel), np.uint8)
        kernel_close = np.ones((close_kernel, close_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best = None
        best_score = -1.0

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < prof["area_min"] or area > prof["area_max"]:
                continue

            perimeter = cv2.arcLength(contour, True)
            circularity = 0.0 if perimeter <= 0 else (4.0 * np.pi * area) / (perimeter * perimeter)
            if circularity < prof["min_circularity"]:
                continue

            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue

            cx = float(moments["m10"] / moments["m00"])
            cy = float(moments["m01"] / moments["m00"])
            score = area * max(circularity, 1e-6)

            if score > best_score:
                best_score = score
                best = {
                    "pos": (cx, cy),
                    "score": score,
                    "area": area,
                    "circularity": circularity,
                }

        if debug and best is not None:
            preview = frame_on.copy()
            cv2.circle(preview, (int(best["pos"][0]), int(best["pos"][1])), 8, (0, 255, 0), 2)
            cv2.imshow("Laser Detection", preview)
            cv2.imshow("Laser Mask", mask)

        return best

    def _robust_aggregate(self, detections, min_inliers=4):
        """对多帧检测结果做稳健聚合，剔除离群点。"""
        if not detections:
            return None, {"inlier_count": 0, "total_count": 0}

        pts = np.array([det["pos"] for det in detections], dtype=np.float32)
        scores = np.array([det.get("score", 1.0) for det in detections], dtype=np.float32)

        median = np.median(pts, axis=0)
        dists = np.linalg.norm(pts - median, axis=1)
        median_dist = float(np.median(dists))
        inlier_radius = max(
            float(self.detector_profile["min_inlier_radius_px"]),
            median_dist * 2.5 if median_dist > 0 else float(self.detector_profile["min_inlier_radius_px"]),
        )

        inlier_mask = dists <= inlier_radius
        if int(inlier_mask.sum()) < min_inliers:
            nearest = np.argsort(dists)[:min(min_inliers, len(detections))]
            inlier_mask = np.zeros(len(detections), dtype=bool)
            inlier_mask[nearest] = True

        inliers = pts[inlier_mask]
        inlier_scores = scores[inlier_mask]
        if len(inliers) == 0:
            return None, {
                "inlier_count": 0,
                "total_count": len(detections),
                "radius_px": inlier_radius,
            }

        weights = np.clip(inlier_scores, 1e-6, None)
        center = np.average(inliers, axis=0, weights=weights)

        stats = {
            "inlier_count": int(len(inliers)),
            "total_count": int(len(detections)),
            "radius_px": float(inlier_radius),
            "std_x": float(np.std(inliers[:, 0])),
            "std_y": float(np.std(inliers[:, 1])),
        }
        return (float(center[0]), float(center[1])), stats

    def passes_quality_gate(self, valid_points, mean_error, max_error):
        """质量门限判断。"""
        if valid_points < self.min_valid_points:
            return False, "valid_points_below_threshold"
        if mean_error > self.mean_error_thres:
            return False, "mean_error_above_threshold"
        if max_error > self.max_error_thres:
            return False, "max_error_above_threshold"
        return True, "pass"

    def calculate_homography(self):
        """计算单应性矩阵并输出质量指标。"""
        galvo_points = self.valid_galvo_points or self.galvo_points[: len(self.pixel_points)]
        source_points = self.pixel_points
        mapping_source = "pixel"
        if len(galvo_points) != len(source_points):
            n_points = min(len(galvo_points), len(source_points))
            galvo_points = galvo_points[:n_points]
            source_points = source_points[:n_points]
        else:
            source_points = source_points

        if len(galvo_points) < 4:
            self.last_failure_reason = f"有效点不足: {len(galvo_points)}"
            return False

        galvo_pts = np.array(galvo_points, dtype=np.float32)
        pixel_pts = np.array(source_points, dtype=np.float32)

        # 当前正式标定目标是 pixel -> galvo 的 2D 视角映射关系。
        # 每个点在进入这里之前已经经过多帧稳健聚合，因此优先使用全部稳定点做拟合，
        # 避免目的坐标单位过大时，RANSAC 阈值过小把好点误判为外点。
        homography, mask = cv2.findHomography(pixel_pts, galvo_pts, 0)
        if homography is None:
            self.last_failure_reason = "单应性矩阵计算失败"
            return False

        pred = cv2.perspectiveTransform(pixel_pts.reshape(-1, 1, 2), homography).reshape(-1, 2)
        errors = np.linalg.norm(pred - galvo_pts, axis=1)

        valid_points = len(galvo_pts)
        inlier_points = valid_points
        mean_error = float(np.mean(errors))
        max_error = float(np.max(errors))
        error_p95 = float(np.percentile(errors, 95))
        passed, gate_reason = self.passes_quality_gate(valid_points, mean_error, max_error)

        self.homography_matrix = homography
        self.quality_metrics = {
            "valid_points": int(valid_points),
            "inlier_points": int(inlier_points),
            "inlier_ratio": float(inlier_points / valid_points if valid_points else 0.0),
            "mean_error": mean_error,
            "max_error": max_error,
            "error_p95": error_p95,
            "pass": bool(passed),
            "gate_reason": gate_reason,
            "mapping_source": mapping_source,
        }
        self.last_failure_reason = "" if passed else gate_reason

        print("✓ 单应性矩阵计算成功")
        print(f"  平均重投影误差: {mean_error:.2f} 振镜单位")
        print(f"  最大重投影误差: {max_error:.2f} 振镜单位")
        return passed

    def _open_camera(self, camera_id=0):
        """优先使用 RealSense，失败后回退到普通 USB 摄像头。"""
        try:
            import pyrealsense2 as rs

            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            profile = pipeline.start(config)
            color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
            intr = color_profile.get_intrinsics()
            print("✓ 使用 RealSense D435")
            return {
                "type": "realsense",
                "pipeline": pipeline,
                "align": rs.align(rs.stream.color),
                "intrinsics": CameraIntrinsics(
                    fx=float(intr.fx),
                    fy=float(intr.fy),
                    cx=float(intr.ppx),
                    cy=float(intr.ppy),
                ),
            }
        except Exception:
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                raise RuntimeError("无法打开摄像头")
            print("✓ 使用 USB 摄像头")
            return {
                "type": "usb",
                "cap": cap,
                "intrinsics": None,
            }

    def _capture_frame(self, camera_ctx):
        if camera_ctx["type"] == "realsense":
            frames = camera_ctx["pipeline"].wait_for_frames()
            aligned = camera_ctx["align"].process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame:
                return None, None
            return np.asanyarray(color_frame.get_data()), depth_frame

        ret, frame = camera_ctx["cap"].read()
        return (frame, None) if ret else (None, None)

    def _read_camera_frame(self, camera_ctx):
        frame, _ = self._capture_frame(camera_ctx)
        return frame

    def _close_camera(self, camera_ctx):
        if camera_ctx["type"] == "realsense":
            camera_ctx["pipeline"].stop()
        else:
            camera_ctx["cap"].release()

    def sample_depth(self, depth_frame, x, y, radius=2):
        """在 ROI 内取深度中值，忽略 0 值。"""
        if depth_frame is None:
            return None
        x = int(round(x))
        y = int(round(y))
        values = []
        for py in range(y - radius, y + radius + 1):
            for px in range(x - radius, x + radius + 1):
                try:
                    depth = float(depth_frame.get_distance(px, py))
                except Exception:
                    depth = 0.0
                if depth > 0:
                    values.append(depth)
        if not values:
            return None
        return float(np.median(np.asarray(values, dtype=np.float32)))

    def _is_detection_inside_margin(self, detection, margin_px=None):
        if detection is None:
            return False
        margin_px = self.edge_margin_px if margin_px is None else int(margin_px)
        x, y = detection["pos"]
        width, height = self.image_resolution
        return margin_px <= x <= (width - margin_px) and margin_px <= y <= (height - margin_px)

    def _probe_marker_detection(self, camera_ctx, gx, gy, debug=False):
        """发送一个小圆 marker，并返回该点在相机中的检测结果。"""
        self.clear_tasks()
        time.sleep(0.05)
        frame_off, depth_off = self._capture_frame(camera_ctx)
        if frame_off is None:
            return None

        self.image_resolution = (frame_off.shape[1], frame_off.shape[0])
        if not self.show_marker(gx, gy, radius=self.marker_radius, slot=0):
            return None
        time.sleep(self.settle_ms / 1000.0)

        detections = []
        latest_depth_frame = depth_off
        for _ in range(self.capture_attempts_per_point):
            frame_on, depth_on = self._capture_frame(camera_ctx)
            if frame_on is None:
                continue
            if depth_on is not None:
                latest_depth_frame = depth_on
            det = self.detect_laser_spot(frame_on=frame_on, frame_off=frame_off, debug=debug)
            if det is not None:
                detections.append(det)

        self.clear_tasks()
        center, stats = self._robust_aggregate(
            detections,
            min_inliers=min(4, max(1, len(detections))),
        )
        if center is None:
            return None
        depth = self.sample_depth(latest_depth_frame, center[0], center[1]) if latest_depth_frame is not None else None
        cam_xy = None
        intrinsics = camera_ctx.get("intrinsics")
        if depth is not None and intrinsics is not None:
            cam_pt = pixel_depth_to_cam(center[0], center[1], depth, intrinsics)
            cam_xy = (float(cam_pt[0]), float(cam_pt[1]))
        return {
            "pos": center,
            "stats": stats,
            "depth": depth,
            "cam_xy": cam_xy,
        }

    def _discover_axis_limit(self, camera_ctx, axis="x", sign=1, debug=False):
        last_good = 0
        failed_at = self.safe_limit
        candidate = self.range_step

        while candidate <= self.safe_limit:
            gx = sign * candidate if axis == "x" else 0
            gy = sign * candidate if axis == "y" else 0
            det = self._probe_marker_detection(camera_ctx, gx, gy, debug=debug)
            if det is not None and self._is_detection_inside_margin(det):
                last_good = candidate
                candidate += self.range_step
            else:
                failed_at = candidate
                break

        probe = last_good + self.range_refine_step
        while probe < failed_at:
            gx = sign * probe if axis == "x" else 0
            gy = sign * probe if axis == "y" else 0
            det = self._probe_marker_detection(camera_ctx, gx, gy, debug=False)
            if det is not None and self._is_detection_inside_margin(det):
                last_good = probe
                probe += self.range_refine_step
            else:
                break
        return last_good

    def discover_safe_ranges(self, camera_id=0, debug=True, safety_factor=0.85):
        """先沿四个方向探测相机可见边界，再返回对称安全范围。"""
        print("\n" + "=" * 60)
        print("开始探测安全扫描范围")
        print("=" * 60)

        camera_ctx = self._open_camera(camera_id)
        try:
            frame = self._read_camera_frame(camera_ctx)
            if frame is None:
                raise RuntimeError("未读取到初始相机帧")
            self.image_resolution = (frame.shape[1], frame.shape[0])

            pos_x = self._discover_axis_limit(camera_ctx, axis="x", sign=1, debug=debug)
            neg_x = self._discover_axis_limit(camera_ctx, axis="x", sign=-1, debug=debug)
            pos_y = self._discover_axis_limit(camera_ctx, axis="y", sign=1, debug=debug)
            neg_y = self._discover_axis_limit(camera_ctx, axis="y", sign=-1, debug=debug)
        finally:
            self.clear_tasks()
            self._close_camera(camera_ctx)
            if debug:
                cv2.destroyAllWindows()

        x_range = int(max(1000, min(pos_x, neg_x) * safety_factor))
        y_range = int(max(1000, min(pos_y, neg_y) * safety_factor))
        result = {
            "x_range": x_range,
            "y_range": y_range,
            "limits": {
                "+x": pos_x,
                "-x": neg_x,
                "+y": pos_y,
                "-y": neg_y,
            },
            "image_resolution": list(self.image_resolution),
        }
        print(f"✓ 建议安全范围: x=±{x_range}, y=±{y_range}")
        print(f"  原始边界: {result['limits']}")
        return result

    def calibrate_with_camera(self, camera_id=0, debug=True):
        """执行自动标定。"""
        print("\n" + "=" * 60)
        print("开始自动标定")
        print("=" * 60)

        try:
            camera_ctx = self._open_camera(camera_id)
        except Exception as exc:
            self.last_failure_reason = f"无法打开摄像头: {exc}"
            print(f"✗ {self.last_failure_reason}")
            return False

        self.valid_galvo_points = []
        self.pixel_points = []
        self.cam_points = []
        self.depth_points = []
        self.detection_history = []

        try:
            for index, (gx, gy) in enumerate(self.galvo_points, start=1):
                print(f"\n点 {index}/{len(self.galvo_points)}: 振镜位置 ({gx}, {gy})")
                det = self._probe_marker_detection(camera_ctx, gx, gy, debug=debug)
                if debug:
                    key = cv2.waitKey(20)
                    if key == ord("q"):
                        self.last_failure_reason = "用户中止标定"
                        return False

                self.detection_history.append(
                    {
                        "galvo_point": [gx, gy],
                        "aggregate": det,
                    }
                )

                if det is None:
                    print("  ✗ 未得到稳定光斑位置")
                    continue

                center = det["pos"]
                stats = det["stats"]
                depth = det.get("depth")
                cam_xy = det.get("cam_xy")
                self.valid_galvo_points.append((gx, gy))
                self.pixel_points.append((center[0], center[1]))
                if cam_xy is not None:
                    self.cam_points.append((cam_xy[0], cam_xy[1]))
                self.depth_points.append(depth)
                print(
                    "  ✓ 像素位置: "
                    f"({center[0]:.1f}, {center[1]:.1f}) "
                    f"± ({stats['std_x']:.1f}, {stats['std_y']:.1f})"
                )
                if depth is not None:
                    print(f"    深度: {depth * 1000.0:.1f} mm")

            if debug:
                cv2.destroyAllWindows()
        finally:
            self.clear_tasks()
            self._close_camera(camera_ctx)

        if len(self.valid_galvo_points) < self.min_valid_points:
            self.last_failure_reason = (
                f"有效点不足: {len(self.valid_galvo_points)} < {self.min_valid_points}"
            )
            print(f"\n✗ {self.last_failure_reason}")
            return False

        print(f"\n✓ 成功检测 {len(self.valid_galvo_points)} 个标定点")
        return self.calculate_homography()

    def save_calibration(self, output_path="galvo_calibration.yaml"):
        """保存标定结果。"""
        if self.homography_matrix is None:
            self.last_failure_reason = "没有可保存的标定数据"
            print(f"✗ {self.last_failure_reason}")
            return False

        x_values = [point[0] for point in self.all_grid_points]
        y_values = [point[1] for point in self.all_grid_points]

        data = {
            "homography_matrix": self.homography_matrix.tolist(),
            "galvo_points": [list(point) for point in self.valid_galvo_points],
            "pixel_points": [list(point) for point in self.pixel_points],
            "cam_points": [list(point) for point in self.cam_points],
            "depth_points_m": [None if depth is None else float(depth) for depth in self.depth_points],
            "all_grid_points": [list(point) for point in self.all_grid_points],
            "quality": self.quality_metrics,
            "mapping_note": "正式标定默认使用 pixel_points -> galvo_points；cam_points/depth_points 仅作诊断记录。",
            "image_resolution": list(self.image_resolution),
            "coordinate_convention": {
                "position": "int16 [-32768, 32767]",
                "dimension": "uint16 [0, 65535]",
            },
            "galvo_range": {
                "x_min": min(x_values),
                "x_max": max(x_values),
                "y_min": min(y_values),
                "y_max": max(y_values),
            },
            "meta": {
                "laser_color": self.laser_color,
                "grid_size": self.grid_size,
                "capture_attempts_per_point": self.capture_attempts_per_point,
                "settle_ms": self.settle_ms,
                "mean_error_thres": self.mean_error_thres,
                "max_error_thres": self.max_error_thres,
                "min_valid_points": self.min_valid_points,
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(output_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)

        print(f"✓ 标定数据已保存: {output_path}")
        return True

    def load_calibration(self, input_path="galvo_calibration.yaml"):
        """加载标定结果。"""
        try:
            with open(input_path, "r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle)
        except Exception as exc:
            self.last_failure_reason = f"加载标定数据失败: {exc}"
            print(f"✗ {self.last_failure_reason}")
            return False

        self.homography_matrix = np.array(data["homography_matrix"], dtype=np.float32)
        self.valid_galvo_points = [tuple(point) for point in data.get("galvo_points", [])]
        self.pixel_points = [tuple(point) for point in data.get("pixel_points", [])]
        self.cam_points = [tuple(point) for point in data.get("cam_points", [])]
        self.depth_points = list(data.get("depth_points_m", []))
        self.all_grid_points = [tuple(point) for point in data.get("all_grid_points", self.valid_galvo_points)]
        self.quality_metrics = data.get("quality", {})
        if "image_resolution" in data:
            width, height = data["image_resolution"]
            self.image_resolution = (int(width), int(height))
        print(f"✓ 标定数据已加载: {input_path}")
        return True

    def save_diagnostic(self, output_path=None, extra_context=None):
        """保存诊断信息。"""
        path = output_path or self.diagnostic_file
        if not path:
            return False

        payload = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "last_failure_reason": self.last_failure_reason,
            "quality_metrics": self.quality_metrics,
            "valid_galvo_points": [list(point) for point in self.valid_galvo_points],
            "pixel_points": [list(point) for point in self.pixel_points],
            "cam_points": [list(point) for point in self.cam_points],
            "depth_points_m": [None if depth is None else float(depth) for depth in self.depth_points],
            "all_grid_points": [list(point) for point in self.all_grid_points],
            "detection_history": self.detection_history,
        }
        if extra_context:
            payload["extra_context"] = extra_context

        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        return True

    def pixel_to_galvo(self, x_pixel, y_pixel):
        """将像素坐标转换为振镜坐标。"""
        if self.homography_matrix is None:
            raise RuntimeError("未进行标定，请先加载或生成标定结果")

        pixel_pt = np.array([[[x_pixel, y_pixel]]], dtype=np.float32)
        galvo_pt = cv2.perspectiveTransform(pixel_pt, self.homography_matrix)
        x_galvo = int(np.clip(galvo_pt[0][0][0], self.INT16_MIN, self.INT16_MAX))
        y_galvo = int(np.clip(galvo_pt[0][0][1], self.INT16_MIN, self.INT16_MAX))
        return x_galvo, y_galvo

    def test_calibration(self, camera_id=0):
        """交互式测试标定效果。"""
        print("\n" + "=" * 60)
        print("测试标定精度")
        print("=" * 60)
        print("点击图像中任意位置，激光将移动到该位置；按 q 退出")

        camera_ctx = self._open_camera(camera_id)

        def mouse_callback(event, x, y, _flags, _param):
            if event == cv2.EVENT_LBUTTONDOWN:
                gx, gy = self.pixel_to_galvo(x, y)
                print(f"像素({x}, {y}) -> 振镜({gx}, {gy})")
                self.show_marker(gx, gy, radius=self.marker_radius, slot=0)

        cv2.namedWindow("Calibration Test")
        cv2.setMouseCallback("Calibration Test", mouse_callback)

        try:
            while True:
                frame = self._read_camera_frame(camera_ctx)
                if frame is None:
                    break
                cv2.imshow("Calibration Test", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            self.clear_tasks()
            self._close_camera(camera_ctx)
            cv2.destroyAllWindows()

    @staticmethod
    def suggest_scan_ranges(
        galvo_points,
        pixel_points,
        image_width,
        image_height,
        safety_factor=0.9,
        grid_margin=5000,
        safe_limit=27767,
    ):
        """根据已有点对建议新的扫描范围。"""
        galvo_pts = np.array(galvo_points, dtype=np.float32)
        pixel_pts = np.array(pixel_points, dtype=np.float32)

        galvo_span_x = float(galvo_pts[:, 0].max() - galvo_pts[:, 0].min())
        galvo_span_y = float(galvo_pts[:, 1].max() - galvo_pts[:, 1].min())
        pixel_span_x = max(float(pixel_pts[:, 0].max() - pixel_pts[:, 0].min()), 1.0)
        pixel_span_y = max(float(pixel_pts[:, 1].max() - pixel_pts[:, 1].min()), 1.0)

        scale_x = galvo_span_x / pixel_span_x
        scale_y = galvo_span_y / pixel_span_y

        pixel_center = pixel_pts.mean(axis=0)
        max_offset_x = max(pixel_center[0], image_width - pixel_center[0])
        max_offset_y = max(pixel_center[1], image_height - pixel_center[1])

        suggested_x = int(min(safe_limit, max_offset_x * scale_x * safety_factor + grid_margin))
        suggested_y = int(min(safe_limit, max_offset_y * scale_y * safety_factor + grid_margin))

        return {
            "x_range": max(1000, suggested_x),
            "y_range": max(1000, suggested_y),
            "pixel_center": [float(pixel_center[0]), float(pixel_center[1])],
            "scale_x": float(scale_x),
            "scale_y": float(scale_y),
        }


def main():
    parser = argparse.ArgumentParser(description="激光振镜自动标定")
    parser.add_argument("--serial-port", default="/dev/ttyUSB0", help="串口设备")
    parser.add_argument("--baudrate", type=int, default=115200, help="波特率")
    parser.add_argument("--output", default="galvo_calibration.yaml", help="输出文件")
    parser.add_argument("--test", action="store_true", help="测试已有标定")
    parser.add_argument("--load", type=str, help="加载标定文件进行测试")
    parser.add_argument("--grid-size", type=int, default=3, help="标定网格维度")
    parser.add_argument("--x-range", type=int, default=15000, help="X 轴扫描半幅")
    parser.add_argument("--y-range", type=int, default=15000, help="Y 轴扫描半幅")
    parser.add_argument("--auto-range", action="store_true", help="先自动探测安全扫描范围，再执行网格标定")
    parser.add_argument("--range-step", type=int, default=3000, help="探边界时的粗扫步长")
    parser.add_argument("--range-refine-step", type=int, default=500, help="探边界时的细扫步长")
    parser.add_argument("--edge-margin-px", type=int, default=40, help="红点距离画面边缘的安全像素边距")
    parser.add_argument("--marker-radius", type=int, default=250, help="标定时绘制的小圆半径")
    parser.add_argument("--safe-limit", type=int, default=27767, help="探边界时允许的最大坐标绝对值")
    parser.add_argument("--min-valid-points", type=int, default=4, help="最小有效点数")
    parser.add_argument("--mean-error-thres", type=float, default=500.0, help="平均误差阈值")
    parser.add_argument("--max-error-thres", type=float, default=1000.0, help="最大误差阈值")
    args = parser.parse_args()

    calibrator = GalvoCalibrator(
        serial_port=args.serial_port,
        baudrate=args.baudrate,
        grid_size=args.grid_size,
        x_range=args.x_range,
        y_range=args.y_range,
        marker_radius=args.marker_radius,
        edge_margin_px=args.edge_margin_px,
        range_step=args.range_step,
        range_refine_step=args.range_refine_step,
        safe_limit=args.safe_limit,
        min_valid_points=args.min_valid_points,
        mean_error_thres=args.mean_error_thres,
        max_error_thres=args.max_error_thres,
    )

    if not calibrator.connect_serial():
        return 1

    try:
        if args.test or args.load:
            calibration_path = args.load or args.output
            if not calibrator.load_calibration(calibration_path):
                return 1
            calibrator.test_calibration()
            return 0

        if args.auto_range:
            suggestion = calibrator.discover_safe_ranges(camera_id=0, debug=True)
            calibrator.x_range = suggestion["x_range"]
            calibrator.y_range = suggestion["y_range"]
            calibrator._init_galvo_grid()

        success = calibrator.calibrate_with_camera(camera_id=0, debug=True)
        if not success:
            print("\n✗ 标定失败")
            return 1

        calibrator.save_calibration(args.output)
        print("\n" + "=" * 60)
        print("标定完成！")
        print(f"标定文件: {args.output}")
        print("\n运行测试:")
        print(f"  python3 {Path(__file__).name} --test --load {args.output}")
        print("=" * 60)
        return 0
    finally:
        calibrator.disconnect_serial()


if __name__ == "__main__":
    raise SystemExit(main())
