#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相机像素 + 深度 → 振镜坐标 映射封装

设计目标：
1. 提供统一的标定与推理阶段接口，便于在 PC 与 板卡上复用。
2. 假设工作平面近似为一个平面（例如传送带/工作台），通过 2D 单应性完成
   「相机平面坐标 → 振镜平面坐标」的转换。

依赖：
- numpy
- opencv-python (用于单应性求解)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import cv2


@dataclass
class CameraIntrinsics:
    """相机内参（针孔模型）。

    fx, fy: 焦距（像素）
    cx, cy: 主点坐标（像素）
    """

    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class HomographyCalib:
    """相机平面坐标 → 振镜平面坐标 的单应性标定结果。"""

    H: np.ndarray  # 3x3 齐次变换矩阵

    def save(self, path: str) -> None:
        """保存为 npy 文件。"""
        np.save(path, self.H)

    @staticmethod
    def load(path: str) -> "HomographyCalib":
        """从 npy 文件加载。"""
        H = np.load(path)
        if H.shape != (3, 3):
            raise ValueError(f"无效的 H 形状: {H.shape}, 期望 (3,3)")
        return HomographyCalib(H=H)


def pixel_depth_to_cam(
    u: float,
    v: float,
    depth: float,
    intr: CameraIntrinsics,
) -> np.ndarray:
    """像素 + 深度 → 相机坐标系三维点。

    参数：
        u, v   : 像素坐标（与深度对齐后的彩色图）
        depth  : 深度值，单位可自定（米/毫米），只要后续保持一致;
        intr   : 相机内参

    返回：
        shape=(3,) 的 numpy 数组 [X, Y, Z]，单位与 depth 一致。
    """
    if depth <= 0:
        raise ValueError(f"非法深度值: {depth}")

    X = (u - intr.cx) / intr.fx * depth
    Y = (v - intr.cy) / intr.fy * depth
    Z = depth
    return np.array([X, Y, Z], dtype=np.float32)


def calibrate_camera_to_galvo_homography(
    cam_points: Iterable[Tuple[float, float]],
    galvo_points: Iterable[Tuple[float, float]],
) -> HomographyCalib:
    """利用一组对应点标定相机平面坐标到振镜坐标的单应性矩阵 H。

    典型流程：
        1. 振镜打 N 个点，记录每个点的控制指令 (X_galvo, Y_galvo)；
        2. 相机拍图 + 深度，算出每个点在相机平面上的坐标 (x_cam, y_cam)，
           例如直接用相机坐标系的 X,Y，或者用像素坐标 (u,v) 也可以；
        3. 把 (x_cam, y_cam) 与 (X_galvo, Y_galvo) 送入本函数，即可得到 H。

    参数：
        cam_points   : 相机侧平面坐标 list，如 [(x1, y1), (x2, y2), ...]
        galvo_points : 振镜控制坐标 list，对应 [(Xg1, Yg1), ...]

    返回：
        HomographyCalib(H)，满足  [Xg, Yg, 1]^T ∝ H · [x_cam, y_cam, 1]^T
    """
    cam_pts = np.asarray(list(cam_points), dtype=np.float32)
    galvo_pts = np.asarray(list(galvo_points), dtype=np.float32)

    if cam_pts.shape != galvo_pts.shape:
        raise ValueError(f"相机点与振镜点数量不一致: {cam_pts.shape} vs {galvo_pts.shape}")
    if cam_pts.shape[0] < 4:
        raise ValueError("单应性估计至少需要 4 个非共线点")

    H, mask = cv2.findHomography(cam_pts, galvo_pts, method=cv2.RANSAC)
    if H is None:
        raise RuntimeError("cv2.findHomography 失败，检查点分布是否退化")
    return HomographyCalib(H=H.astype(np.float32))


def cam_to_galvo(
    x_cam: float,
    y_cam: float,
    calib: HomographyCalib,
) -> Tuple[float, float]:
    """相机平面坐标 → 振镜平面坐标。

    参数：
        x_cam, y_cam : 相机平面坐标（可用相机坐标系 X,Y 或像素 u,v）
        calib        : 标定结果（单应性）

    返回：
        (X_galvo, Y_galvo) 振镜控制坐标，单位与标定数据保持一致
    """
    vec = np.array([x_cam, y_cam, 1.0], dtype=np.float32)
    out = calib.H @ vec
    if out[2] == 0:
        raise ZeroDivisionError("单应性变换结果 w=0，检查 H 是否合理")
    Xg = float(out[0] / out[2])
    Yg = float(out[1] / out[2])
    return Xg, Yg


def pixel_depth_to_galvo(
    u: float,
    v: float,
    depth: float,
    intr: CameraIntrinsics,
    calib: HomographyCalib,
) -> Tuple[float, float]:
    """封装：像素 + 深度 → 振镜坐标。

    典型用法（在线推理阶段）：
        1. YOLO 输出框中心 (u, v)；
        2. 在对齐后的深度图上取该点/邻域的 depth；
        3. 调用本函数得到 (X_galvo, Y_galvo)；
        4. 将振镜移动到该位置执行打标/喷射/分拣。
    """
    P_cam = pixel_depth_to_cam(u, v, depth, intr)
    # 这里只用 X,Y 作为平面坐标，Z 视为对平面坐标的微小影响，可根据项目需要再扩展
    Xcam, Ycam = float(P_cam[0]), float(P_cam[1])
    return cam_to_galvo(Xcam, Ycam, calib)


def collect_calib_points_example() -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """示例函数：返回虚构的标定点对，方便测试与演示。

    实际工程中，你需要用真实的相机/振镜采集数据，再替换此处。
    """
    # 假设四个角点：相机平面坐标
    cam_pts = [
        (0.0, 0.0),
        (100.0, 0.0),
        (100.0, 50.0),
        (0.0, 50.0),
    ]
    # 振镜坐标（单位/量纲由实际控制接口决定）
    galvo_pts = [
        (-1.0, -1.0),
        (1.0, -1.0),
        (1.0, 1.0),
        (-1.0, 1.0),
    ]
    return cam_pts, galvo_pts


if __name__ == "__main__":
    # 简单自检：用虚构数据估计 H 再做一次映射
    cam_pts, galvo_pts = collect_calib_points_example()
    calib = calibrate_camera_to_galvo_homography(cam_pts, galvo_pts)
    x_test, y_test = 50.0, 25.0  # 理论上是中心点，对应 (0, 0)
    Xg, Yg = cam_to_galvo(x_test, y_test, calib)
    print("Test cam_to_galvo:", Xg, Yg)

