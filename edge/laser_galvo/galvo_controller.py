#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
激光振镜控制器
用于物体检测后用激光绘制边界框
"""
import serial
import time
import numpy as np
import cv2
from pathlib import Path


class LaserGalvoController:
    """激光振镜控制器 - 适配STM32文本协议"""

    # 无标定时的像素→振镜线性系数（像素中心对应振镜原点，坐标系反转）
    _PX_TO_GALVO_X = 102.4  # 振镜单位/像素，X 方向
    _PX_TO_GALVO_Y = 136.5  # 振镜单位/像素，Y 方向

    def __init__(self, serial_port='/dev/ttyUSB0', baudrate=115200, calibration_file=None):
        """
        初始化控制器

        Args:
            serial_port: 串口设备路径
            baudrate: 波特率
            calibration_file: 标定文件路径（可选）
        """
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.ser = None
        self.homography_matrix = None
        self.task_index = 0
        self._batching = False
        self._pending_cmds: list = []

        if calibration_file:
            self.load_calibration(calibration_file)

    def connect(self):
        """连接串口"""
        try:
            self.ser = serial.Serial(
                port=self.serial_port,
                baudrate=self.baudrate,
                timeout=0.1,
                write_timeout=0.1
            )
            time.sleep(2)  # 等待连接稳定
            print(f"✓ 振镜串口已连接: {self.serial_port}")
            return True
        except Exception as e:
            print(f"✗ 振镜串口连接失败: {e}")
            return False

    def disconnect(self):
        """断开串口"""
        if self.ser and self.ser.is_open:
            self.update_tasks()
            self.ser.close()
            print("✓ 振镜串口已断开")

    def load_calibration(self, calibration_file):
        """加载标定文件"""
        import yaml
        try:
            with open(calibration_file, 'r') as f:
                data = yaml.safe_load(f)
            self.homography_matrix = np.array(data['homography_matrix'], dtype=np.float32)
            print(f"✓ 加载标定文件: {calibration_file}")
            return True
        except Exception as e:
            print(f"✗ 加载标定文件失败: {e}")
            return False

    def pixel_to_galvo(self, x_pixel, y_pixel, image_width=640, image_height=480):
        """
        像素坐标转振镜坐标

        Args:
            x_pixel: 像素X坐标
            y_pixel: 像素Y坐标
            image_width: 图像宽度
            image_height: 图像高度

        Returns:
            (x_galvo, y_galvo): 振镜坐标值 (-32768到32767，STM32 int16_t)
        """
        if self.homography_matrix is not None:
            # 使用标定矩阵转换
            pixel_pt = np.array([[[x_pixel, y_pixel]]], dtype=np.float32)
            galvo_pt = cv2.perspectiveTransform(pixel_pt, self.homography_matrix)
            x_galvo = int(np.clip(galvo_pt[0][0][0], -32768, 32767))
            y_galvo = int(np.clip(galvo_pt[0][0][1], -32768, 32767))
        else:
            # 使用用户提供的映射系数
            # 像素中心对应振镜(0, 0)
            # X方向：1像素 = 102.4 单位
            # Y方向：1像素 = 136.5 单位
            # 注意：左上角为正正，右下角为负负（坐标系反转）
            center_x = image_width / 2.0  # 320
            center_y = image_height / 2.0  # 240

            x_galvo = int((center_x - x_pixel) * self._PX_TO_GALVO_X)  # 反转X
            y_galvo = int((center_y - y_pixel) * self._PX_TO_GALVO_Y)  # 反转Y

            # 限制在int16_t范围内
            x_galvo = int(np.clip(x_galvo, -32768, 32767))
            y_galvo = int(np.clip(y_galvo, -32768, 32767))

        return (x_galvo, y_galvo)

    def begin_batch(self) -> None:
        """开始批量模式：后续命令缓存，不立即写串口。"""
        self._batching = True
        self._pending_cmds = []

    def end_batch(self) -> bool:
        """结束批量模式：将所有缓存命令合并为单次 serial.write() 写入。"""
        self._batching = False
        if not self._pending_cmds:
            return True
        payload = "".join(self._pending_cmds)
        self._pending_cmds = []
        if self.ser is None or not self.ser.is_open:
            return False
        try:
            self.ser.write(payload.encode("utf-8"))
            return True
        except Exception as e:
            print(f"✗ 批量发送失败: {e}")
            return False

    def _send_text_command(self, command_str):
        if self.ser is None or not self.ser.is_open:
            return False
        if not command_str.endswith(";"):
            command_str = command_str + ";"
        if self._batching:
            self._pending_cmds.append(command_str)
            return True
        try:
            self.ser.write(command_str.encode("utf-8"))
            return True
        except Exception as e:
            print(f"✗ 发送命令失败: {e}")
            return False

    def update_tasks(self):
        """发送更新标志，让STM32切换到新的任务缓冲区"""
        if self._batching:
            self._pending_cmds.append("U;")
            return self.end_batch()
        return self._send_text_command("U;")

    def laser_on(self):
        """新固件不再支持独立 L1 指令，保留兼容接口。"""
        print("[GALVO] 当前固件不支持独立 laser_on 指令")
        return False

    def laser_off(self):
        """清空任务，停止当前绘制。"""
        return self.update_tasks()

    def draw_circle(self, x_galvo, y_galvo, radius, task_index=None):
        """
        发送绘制圆形命令

        Args:
            x_galvo: 中心X坐标 (int16)
            y_galvo: 中心Y坐标 (int16)
            radius: 半径
            task_index: 任务索引（0-9），默认自动递增
        """
        if task_index is None:
            task_index = self.task_index
            self.task_index = (self.task_index + 1) % 10

        cmd = f"{task_index}C,{x_galvo},{y_galvo},{radius}"
        return self._send_text_command(cmd)

    def _project_box_corners(
        self,
        box,
        *,
        pixel_coords: bool,
        image_width: int = 640,
        image_height: int = 480,
    ):
        """将框的四个角点转换到振镜坐标，顺序为左上、右上、右下、左下。"""
        x1, y1, x2, y2 = [float(v) for v in box]
        if not pixel_coords:
            return [
                (int(round(x1)), int(round(y1))),
                (int(round(x2)), int(round(y1))),
                (int(round(x2)), int(round(y2))),
                (int(round(x1)), int(round(y2))),
            ]

        if self.homography_matrix is not None:
            pixel_corners = np.array(
                [[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]],
                dtype=np.float32,
            )
            galvo_corners = cv2.perspectiveTransform(pixel_corners, self.homography_matrix).reshape(-1, 2)
            return [
                (
                    int(np.clip(round(float(pt[0])), -32768, 32767)),
                    int(np.clip(round(float(pt[1])), -32768, 32767)),
                )
                for pt in galvo_corners
            ]

        return [
            self.pixel_to_galvo(x1, y1, image_width, image_height),
            self.pixel_to_galvo(x2, y1, image_width, image_height),
            self.pixel_to_galvo(x2, y2, image_width, image_height),
            self.pixel_to_galvo(x1, y2, image_width, image_height),
        ]

    def draw_box_sweep(
        self,
        box,
        *,
        pixel_coords: bool = True,
        task_index=None,
        image_width: int = 640,
        image_height: int = 480,
        max_tasks: int = 8,
        min_radius: int = 450,
        max_radius: int = 1400,
    ):
        """
        用多个圆形任务沿矩形四边采样，近似旧版逐边扫框效果。

        这样不依赖 STM32 的 `R` 实现质量，适合作为单目标演示模式。
        """
        start_index = self.task_index if task_index is None else int(task_index)
        max_slots = max(1, 10 - start_index)
        sample_budget = max(4, min(int(max_tasks), max_slots))
        corners = self._project_box_corners(
            box,
            pixel_coords=pixel_coords,
            image_width=image_width,
            image_height=image_height,
        )
        closed = corners + [corners[0]]
        edge_lengths = [
            float(np.hypot(closed[i + 1][0] - closed[i][0], closed[i + 1][1] - closed[i][1]))
            for i in range(4)
        ]
        total_length = sum(edge_lengths)
        if total_length <= 0:
            cx = sum(pt[0] for pt in corners) / 4.0
            cy = sum(pt[1] for pt in corners) / 4.0
            return self.draw_circle(int(round(cx)), int(round(cy)), min_radius, task_index=start_index)

        allocations = [1, 1, 1, 1]
        remaining = sample_budget - 4
        while remaining > 0:
            edge_idx = int(np.argmax(edge_lengths))
            allocations[edge_idx] += 1
            # 轻微衰减，避免额外点全部堆到最长边。
            edge_lengths[edge_idx] *= 0.82
            remaining -= 1

        sampled_points = []
        for edge_idx, samples in enumerate(allocations):
            start = np.array(closed[edge_idx], dtype=np.float32)
            end = np.array(closed[edge_idx + 1], dtype=np.float32)
            for step in range(samples):
                t = step / float(samples)
                point = start * (1.0 - t) + end * t
                rounded = (int(round(float(point[0]))), int(round(float(point[1]))))
                if not sampled_points or sampled_points[-1] != rounded:
                    sampled_points.append(rounded)

        if len(sampled_points) > sample_budget:
            sampled_points = sampled_points[:sample_budget]

        if len(sampled_points) >= 2:
            gap = min(
                float(np.hypot(b[0] - a[0], b[1] - a[1]))
                for a, b in zip(sampled_points, sampled_points[1:] + sampled_points[:1])
                if a != b
            )
        else:
            gap = float(min_radius * 2)
        radius = int(np.clip(gap * 0.45, min_radius, max_radius))

        ok = True
        for offset, (gx, gy) in enumerate(sampled_points):
            ok = self.draw_circle(gx, gy, radius, task_index=start_index + offset) and ok
        return ok

    def draw_box(
        self,
        box,
        pixel_coords=True,
        task_index=None,
        image_width=640,
        image_height=480,
        steps_per_edge=None
    ):
        """
        绘制矩形边界框（发送R命令到STM32）

        Args:
            box: 边界框，格式为 [x1, y1, x2, y2]
            pixel_coords: box是否为像素坐标（True）还是振镜坐标（False）
            task_index: 任务索引（0-9），默认自动递增
            image_width: 图像宽度（用于像素转换）
            image_height: 图像高度（用于像素转换）
            steps_per_edge: 兼容参数，当前STM32文本协议不使用该参数
        """
        # 向后兼容旧调用方：当前协议仅支持矩形参数，不支持插值步数控制。
        _ = steps_per_edge
        x1, y1, x2, y2 = box

        if pixel_coords:
            # 计算中心点和尺寸
            center_x_pixel = (x1 + x2) / 2
            center_y_pixel = (y1 + y2) / 2
            width_pixel = abs(x2 - x1)
            height_pixel = abs(y2 - y1)

            if self.homography_matrix is not None:
                # 中心：直接投影像素中心，避免 AABB 中心在透视变换下偏离真实投影位置
                center_x_galvo, center_y_galvo = self.pixel_to_galvo(
                    center_x_pixel, center_y_pixel, image_width, image_height
                )
                # 宽高：仍用四角投影后的 AABB 包围盒（STM32 R 命令只支持轴对齐矩形）
                galvo_corners = np.array(
                    self._project_box_corners(
                        box,
                        pixel_coords=True,
                        image_width=image_width,
                        image_height=image_height,
                    ),
                    dtype=np.float32,
                )
                min_x, min_y = galvo_corners.min(axis=0)
                max_x, max_y = galvo_corners.max(axis=0)
                width_galvo = int(round(max(1.0, max_x - min_x)))
                height_galvo = int(round(max(1.0, max_y - min_y)))
            else:
                # 转换中心点到振镜坐标
                center_x_galvo, center_y_galvo = self.pixel_to_galvo(
                    center_x_pixel, center_y_pixel, image_width, image_height
                )

                # 无标定时退回线性比例近似。
                width_galvo = int(width_pixel * 102.4)
                height_galvo = int(height_pixel * 136.5)
        else:
            # 已经是振镜坐标
            center_x_galvo = int((x1 + x2) / 2)
            center_y_galvo = int((y1 + y2) / 2)
            width_galvo = abs(x2 - x1)
            height_galvo = abs(y2 - y1)

        # 发送矩形绘制命令
        if task_index is None:
            task_index = self.task_index
            self.task_index = (self.task_index + 1) % 10

        cmd = f"{task_index}R,{center_x_galvo},{center_y_galvo},{width_galvo},{height_galvo}"
        return self._send_text_command(cmd)

    def draw_boxes(self, boxes, image_width=640, image_height=480, delay_between_boxes=0.05):
        """
        绘制多个边界框（批量发送后统一更新）

        Args:
            boxes: 边界框列表，每个元素为 [x1, y1, x2, y2]
            image_width: 图像宽度
            image_height: 图像高度
            delay_between_boxes: 每个命令之间的延迟（秒）
        """
        # 重置任务索引
        self.task_index = 0

        # 批量发送所有box命令
        for i, box in enumerate(boxes[:10]):  # STM32最多支持10个任务
            self.draw_box(box, pixel_coords=True, task_index=i,
                         image_width=image_width, image_height=image_height)
            time.sleep(delay_between_boxes)

        # 发送更新命令，让STM32开始绘制
        self.update_tasks()
        return True

    def test_pattern(self):
        """测试图案：绘制一个中心矩形"""
        print("绘制测试图案...")

        # 中心矩形（振镜坐标）
        box = [0, 0, 2000, 2000]  # 中心点(1000, 1000), 宽度2000, 高度2000

        self.draw_box(box, pixel_coords=False, task_index=0)
        self.update_tasks()
        print("✓ 测试图案命令已发送")


def main():
    """测试程序"""
    import argparse

    parser = argparse.ArgumentParser(description='激光振镜控制器测试（STM32文本协议）')
    parser.add_argument('--serial-port', default='/dev/ttyUSB0', help='串口设备')
    parser.add_argument('--baudrate', type=int, default=115200, help='波特率')
    parser.add_argument('--calibration', help='标定文件（可选）')
    parser.add_argument('--test', action='store_true', help='运行测试图案')
    args = parser.parse_args()

    # 创建控制器
    controller = LaserGalvoController(
        serial_port=args.serial_port,
        baudrate=args.baudrate,
        calibration_file=args.calibration
    )

    # 连接
    if not controller.connect():
        return

    try:
        if args.test:
            # 测试模式
            controller.test_pattern()
            time.sleep(2)
        else:
            # 交互模式
            print("\n命令 (STM32文本协议):")
            print("  box x1,y1,x2,y2  - 绘制边界框（像素坐标）")
            print("  circle x,y,r     - 绘制圆形")
            print("  test             - 测试图案")
            print("  update           - 发送更新标志")
            print("  quit             - 退出")

            while True:
                cmd = input("\n> ").strip().lower()

                if cmd == 'quit':
                    break
                elif cmd == 'test':
                    controller.test_pattern()
                elif cmd == 'update':
                    controller.update_tasks()
                    print("✓ 更新命令已发送")
                elif cmd.startswith('box'):
                    parts = cmd.split()[1].split(',')
                    if len(parts) == 4:
                        box = [int(p) for p in parts]
                        controller.draw_box(box, pixel_coords=True)
                        controller.update_tasks()
                        print(f"✓ 发送矩形命令: {box}")
                elif cmd.startswith('circle'):
                    parts = cmd.split()[1].split(',')
                    if len(parts) == 3:
                        x, y, r = int(parts[0]), int(parts[1]), int(parts[2])
                        controller.draw_circle(x, y, r)
                        controller.update_tasks()
                        print(f"✓ 发送圆形命令: ({x}, {y}), r={r}")
                else:
                    print("未知命令")

    finally:
        controller.disconnect()


if __name__ == '__main__':
    main()
