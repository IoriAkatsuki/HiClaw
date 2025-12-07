#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
激光振镜控制器
用于物体检测后用激光绘制边界框
"""
import serial
import struct
import time
import numpy as np
import cv2
from pathlib import Path


class LaserGalvoController:
    """激光振镜控制器 - 适配STM32文本协议"""

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
        self.task_index = 0  # 任务索引，用于STM32协议

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
            self.laser_off()
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
            # 简单线性映射：像素坐标 -> 振镜坐标
            # 中心为(0,0)，范围-32768到32767
            x_galvo = int((x_pixel - image_width/2) * (65535 / image_width))
            y_galvo = int((y_pixel - image_height/2) * (65535 / image_height))
            x_galvo = np.clip(x_galvo, -32768, 32767)
            y_galvo = np.clip(y_galvo, -32768, 32767)

        return (x_galvo, y_galvo)

    def _send_text_command(self, command_str):
        """
        发送文本命令到STM32

        Args:
            command_str: 命令字符串（如 "R0 1000,2000,3000,4000"）
        """
        if self.ser is None or not self.ser.is_open:
            return False

        try:
            # 发送命令（添加换行符）
            cmd_bytes = (command_str + '\n').encode('utf-8')
            self.ser.write(cmd_bytes)
            time.sleep(0.005)  # 短暂延迟确保STM32处理
            return True
        except Exception as e:
            print(f"✗ 发送命令失败: {e}")
            return False

    def update_tasks(self):
        """发送更新标志，让STM32切换到新的任务缓冲区"""
        return self._send_text_command('U')

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

        cmd = f"C{task_index} {x_galvo},{y_galvo},{radius}"
        return self._send_text_command(cmd)

    def draw_box(self, box, pixel_coords=True, task_index=None, image_width=640, image_height=480):
        """
        绘制矩形边界框（发送R命令到STM32）

        Args:
            box: 边界框，格式为 [x1, y1, x2, y2]
            pixel_coords: box是否为像素坐标（True）还是振镜坐标（False）
            task_index: 任务索引（0-9），默认自动递增
            image_width: 图像宽度（用于像素转换）
            image_height: 图像高度（用于像素转换）
        """
        x1, y1, x2, y2 = box

        if pixel_coords:
            # 计算中心点和尺寸
            center_x_pixel = (x1 + x2) / 2
            center_y_pixel = (y1 + y2) / 2
            width_pixel = abs(x2 - x1)
            height_pixel = abs(y2 - y1)

            # 转换中心点到振镜坐标
            center_x_galvo, center_y_galvo = self.pixel_to_galvo(
                center_x_pixel, center_y_pixel, image_width, image_height
            )

            # 计算振镜坐标系中的宽度和高度（简单比例缩放）
            width_galvo = int(width_pixel * (65535 / image_width))
            height_galvo = int(height_pixel * (65535 / image_height))
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

        cmd = f"R{task_index} {center_x_galvo},{center_y_galvo},{width_galvo},{height_galvo}"
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
