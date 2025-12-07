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
    """激光振镜控制器"""

    # 串口命令
    CMD_MOVE = 0x01      # 移动到指定位置
    CMD_LASER_ON = 0x02  # 打开激光
    CMD_LASER_OFF = 0x03 # 关闭激光
    CMD_DRAW_LINE = 0x04 # 绘制直线
    CMD_DRAW_BOX = 0x05  # 绘制矩形

    def __init__(self, serial_port='/dev/ttyUSB0', baudrate=115200, calibration_file=None):
        """
        初始化控制器

        Args:
            serial_port: 串口设备路径
            baudrate: 波特率
            calibration_file: 标定文件路径
        """
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.ser = None
        self.homography_matrix = None

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

    def pixel_to_galvo(self, x_pixel, y_pixel):
        """
        像素坐标转振镜坐标

        Args:
            x_pixel: 像素X坐标
            y_pixel: 像素Y坐标

        Returns:
            (x_galvo, y_galvo): 振镜DAC值 (0-65535)
        """
        if self.homography_matrix is None:
            raise RuntimeError("未加载标定数据")

        pixel_pt = np.array([[[x_pixel, y_pixel]]], dtype=np.float32)
        galvo_pt = cv2.perspectiveTransform(pixel_pt, self.homography_matrix)

        x_galvo = int(np.clip(galvo_pt[0][0][0], 0, 65535))
        y_galvo = int(np.clip(galvo_pt[0][0][1], 0, 65535))

        return (x_galvo, y_galvo)

    def _calculate_crc16(self, data):
        """计算CRC16校验和"""
        crc = 0xFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x0001:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc >>= 1
        return crc

    def _send_command(self, cmd, x=0, y=0, param=0):
        """
        发送命令到STM32

        协议格式 (10字节):
        [0xAA] [0x55] [CMD] [X_H] [X_L] [Y_H] [Y_L] [PARAM] [CRC_H] [CRC_L]

        Args:
            cmd: 命令字节
            x: X坐标 (0-65535)
            y: Y坐标 (0-65535)
            param: 附加参数
        """
        if self.ser is None or not self.ser.is_open:
            return False

        # 构建数据包
        packet = bytearray([
            0xAA, 0x55,  # 帧头
            cmd,         # 命令
            (x >> 8) & 0xFF,  # X高字节
            x & 0xFF,         # X低字节
            (y >> 8) & 0xFF,  # Y高字节
            y & 0xFF,         # Y低字节
            param             # 参数
        ])

        # 计算CRC
        crc = self._calculate_crc16(packet)
        packet.append((crc >> 8) & 0xFF)  # CRC高字节
        packet.append(crc & 0xFF)         # CRC低字节

        try:
            self.ser.write(packet)
            return True
        except Exception as e:
            print(f"✗ 发送命令失败: {e}")
            return False

    def move_to(self, x_galvo, y_galvo):
        """
        移动到指定振镜坐标

        Args:
            x_galvo: X轴DAC值 (0-65535)
            y_galvo: Y轴DAC值 (0-65535)
        """
        return self._send_command(self.CMD_MOVE, x_galvo, y_galvo)

    def move_to_pixel(self, x_pixel, y_pixel):
        """
        移动到指定像素坐标

        Args:
            x_pixel: 像素X坐标
            y_pixel: 像素Y坐标
        """
        x_galvo, y_galvo = self.pixel_to_galvo(x_pixel, y_pixel)
        return self.move_to(x_galvo, y_galvo)

    def laser_on(self):
        """打开激光"""
        return self._send_command(self.CMD_LASER_ON)

    def laser_off(self):
        """关闭激光"""
        return self._send_command(self.CMD_LASER_OFF)

    def draw_line(self, x1_galvo, y1_galvo, x2_galvo, y2_galvo, steps=20):
        """
        绘制直线

        Args:
            x1_galvo, y1_galvo: 起点振镜坐标
            x2_galvo, y2_galvo: 终点振镜坐标
            steps: 插值步数
        """
        # 移动到起点
        self.move_to(x1_galvo, y1_galvo)
        time.sleep(0.01)

        # 打开激光
        self.laser_on()
        time.sleep(0.01)

        # 插值移动
        for i in range(steps + 1):
            t = i / steps
            x = int(x1_galvo + t * (x2_galvo - x1_galvo))
            y = int(y1_galvo + t * (y2_galvo - y1_galvo))
            self.move_to(x, y)
            time.sleep(0.005)  # 控制绘制速度

        # 关闭激光
        self.laser_off()

    def draw_box(self, box, steps_per_edge=20, pixel_coords=True):
        """
        绘制矩形边界框

        Args:
            box: 边界框，格式为 [x1, y1, x2, y2]
            steps_per_edge: 每条边的插值步数
            pixel_coords: box是否为像素坐标（True）还是振镜坐标（False）
        """
        x1, y1, x2, y2 = box

        if pixel_coords:
            # 转换四个角点
            tl_galvo = self.pixel_to_galvo(x1, y1)  # 左上
            tr_galvo = self.pixel_to_galvo(x2, y1)  # 右上
            br_galvo = self.pixel_to_galvo(x2, y2)  # 右下
            bl_galvo = self.pixel_to_galvo(x1, y2)  # 左下
        else:
            tl_galvo = (x1, y1)
            tr_galvo = (x2, y1)
            br_galvo = (x2, y2)
            bl_galvo = (x1, y2)

        corners = [tl_galvo, tr_galvo, br_galvo, bl_galvo]

        # 移动到第一个角点
        self.move_to(*corners[0])
        time.sleep(0.02)

        # 打开激光
        self.laser_on()
        time.sleep(0.01)

        # 绘制四条边
        for i in range(4):
            start = corners[i]
            end = corners[(i + 1) % 4]

            # 插值移动
            for step in range(steps_per_edge + 1):
                t = step / steps_per_edge
                x = int(start[0] + t * (end[0] - start[0]))
                y = int(start[1] + t * (end[1] - start[1]))
                self.move_to(x, y)
                time.sleep(0.003)  # 控制速度

        # 关闭激光
        self.laser_off()
        time.sleep(0.01)

    def draw_boxes(self, boxes, delay_between_boxes=0.1):
        """
        绘制多个边界框

        Args:
            boxes: 边界框列表，每个元素为 [x1, y1, x2, y2]
            delay_between_boxes: 每个框之间的延迟（秒）
        """
        for box in boxes:
            self.draw_box(box)
            time.sleep(delay_between_boxes)

    def test_pattern(self):
        """测试图案：绘制一个中心方框"""
        print("绘制测试图案...")

        # 中心点 (假设摄像头分辨率640x480)
        center_x, center_y = 320, 240
        size = 100

        box = [
            center_x - size,
            center_y - size,
            center_x + size,
            center_y + size
        ]

        self.draw_box(box, pixel_coords=True)
        print("✓ 测试图案绘制完成")


def main():
    """测试程序"""
    import argparse

    parser = argparse.ArgumentParser(description='激光振镜控制器测试')
    parser.add_argument('--serial-port', default='/dev/ttyUSB0', help='串口设备')
    parser.add_argument('--baudrate', type=int, default=115200, help='波特率')
    parser.add_argument('--calibration', required=True, help='标定文件')
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
        else:
            # 交互模式
            print("\n命令:")
            print("  box x1,y1,x2,y2  - 绘制边界框")
            print("  move x,y         - 移动到位置")
            print("  laser on/off     - 激光开关")
            print("  test             - 测试图案")
            print("  quit             - 退出")

            while True:
                cmd = input("\n> ").strip().lower()

                if cmd == 'quit':
                    break
                elif cmd == 'test':
                    controller.test_pattern()
                elif cmd.startswith('box'):
                    parts = cmd.split()[1].split(',')
                    if len(parts) == 4:
                        box = [int(p) for p in parts]
                        controller.draw_box(box)
                elif cmd.startswith('move'):
                    parts = cmd.split()[1].split(',')
                    if len(parts) == 2:
                        x, y = int(parts[0]), int(parts[1])
                        controller.move_to_pixel(x, y)
                elif cmd == 'laser on':
                    controller.laser_on()
                elif cmd == 'laser off':
                    controller.laser_off()
                else:
                    print("未知命令")

    finally:
        controller.disconnect()


if __name__ == '__main__':
    main()
