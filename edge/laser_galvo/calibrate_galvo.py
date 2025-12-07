#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
激光振镜自动标定程序
建立摄像头像素坐标与振镜DAC值之间的映射关系
"""
import cv2
import numpy as np
import serial
import time
import yaml
from pathlib import Path


class GalvoCalibrator:
    """振镜标定器"""

    def __init__(self, serial_port='/dev/ttyUSB0', baudrate=115200):
        """
        初始化标定器

        Args:
            serial_port: 串口设备路径
            baudrate: 波特率
        """
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.ser = None

        # 标定点坐标 (振镜DAC值，0-65535)
        # 3x3网格，从左上到右下
        self.galvo_points = []
        self._init_galvo_grid()

        # 检测到的像素坐标
        self.pixel_points = []

        # 单应性矩阵
        self.homography_matrix = None

    def _init_galvo_grid(self):
        """初始化振镜标定网格 (3x3)"""
        # 定义安全工作区域 (留出边界)
        margin = 8000
        x_min, x_max = margin, 65535 - margin
        y_min, y_max = margin, 65535 - margin

        # 3x3网格
        for row in range(3):
            for col in range(3):
                x = int(x_min + col * (x_max - x_min) / 2)
                y = int(y_min + row * (y_max - y_min) / 2)
                self.galvo_points.append((x, y))

        print(f"✓ 初始化 {len(self.galvo_points)} 个振镜标定点")

    def connect_serial(self):
        """连接串口"""
        try:
            self.ser = serial.Serial(
                port=self.serial_port,
                baudrate=self.baudrate,
                timeout=1.0
            )
            time.sleep(2)  # 等待连接稳定
            print(f"✓ 串口已连接: {self.serial_port} @ {self.baudrate}")
            return True
        except Exception as e:
            print(f"✗ 串口连接失败: {e}")
            return False

    def send_galvo_position(self, x, y):
        """
        发送振镜位置命令

        Args:
            x: X轴DAC值 (0-65535)
            y: Y轴DAC值 (0-65535)
        """
        if self.ser is None:
            return False

        # 简化版串口协议: "Gx,y\n"
        # 实际应用中应使用带CRC的二进制协议
        cmd = f"G{x},{y}\n"
        try:
            self.ser.write(cmd.encode('ascii'))
            return True
        except Exception as e:
            print(f"✗ 发送命令失败: {e}")
            return False

    def enable_laser(self, enable=True):
        """控制激光开关"""
        if self.ser is None:
            return False
        cmd = "L1\n" if enable else "L0\n"
        try:
            self.ser.write(cmd.encode('ascii'))
            return True
        except Exception as e:
            print(f"✗ 激光控制失败: {e}")
            return False

    def detect_laser_spot(self, frame, debug=False):
        """
        检测激光光斑位置

        Args:
            frame: BGR图像
            debug: 是否显示调试信息

        Returns:
            (x, y): 光斑中心像素坐标，失败返回None
        """
        # 转换到HSV空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 红色激光阈值 (可根据实际情况调整)
        # 红色在HSV中有两个范围
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # 形态学处理
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None

        # 找到最大轮廓（假设是激光光斑）
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        # 面积过滤（光斑应该是小而明亮的）
        if area < 10 or area > 1000:
            return None

        # 计算质心
        M = cv2.moments(largest_contour)
        if M['m00'] == 0:
            return None

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        if debug:
            debug_frame = frame.copy()
            cv2.drawContours(debug_frame, [largest_contour], -1, (0, 255, 0), 2)
            cv2.circle(debug_frame, (cx, cy), 10, (0, 0, 255), -1)
            cv2.putText(debug_frame, f"({cx}, {cy})", (cx + 15, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow('Laser Detection', debug_frame)
            cv2.imshow('Mask', mask)

        return (cx, cy)

    def calibrate_with_camera(self, camera_id=0, debug=True):
        """
        使用摄像头执行自动标定

        Args:
            camera_id: 摄像头ID或RealSense配置
            debug: 是否显示调试窗口

        Returns:
            bool: 标定是否成功
        """
        print("\n" + "=" * 60)
        print("开始自动标定")
        print("=" * 60)

        # 初始化摄像头 (支持USB摄像头或RealSense)
        try:
            import pyrealsense2 as rs
            use_realsense = True
            print("✓ 使用 RealSense D435")

            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            pipeline.start(config)

        except ImportError:
            use_realsense = False
            print("✓ 使用 USB 摄像头")
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                print("✗ 无法打开摄像头")
                return False

        # 打开激光
        self.enable_laser(True)
        time.sleep(0.5)

        print(f"\n开始标定 {len(self.galvo_points)} 个点...")

        self.pixel_points = []

        for i, (gx, gy) in enumerate(self.galvo_points):
            print(f"\n点 {i+1}/{len(self.galvo_points)}: 振镜位置 ({gx}, {gy})")

            # 移动振镜
            self.send_galvo_position(gx, gy)
            time.sleep(0.5)  # 等待振镜稳定

            # 捕获多帧取平均（提高精度）
            detected_positions = []

            for attempt in range(10):
                # 读取图像
                if use_realsense:
                    frames = pipeline.wait_for_frames()
                    color_frame = frames.get_color_frame()
                    if not color_frame:
                        continue
                    frame = np.asanyarray(color_frame.get_data())
                else:
                    ret, frame = cap.read()
                    if not ret:
                        continue

                # 检测激光光斑
                pos = self.detect_laser_spot(frame, debug=debug and attempt == 0)

                if pos is not None:
                    detected_positions.append(pos)

                if debug:
                    key = cv2.waitKey(50)
                    if key == ord('q'):
                        print("用户中止标定")
                        if use_realsense:
                            pipeline.stop()
                        else:
                            cap.release()
                        cv2.destroyAllWindows()
                        return False

            # 计算平均位置
            if len(detected_positions) == 0:
                print(f"  ✗ 未检测到激光光斑，跳过此点")
                continue

            avg_x = np.mean([p[0] for p in detected_positions])
            avg_y = np.mean([p[1] for p in detected_positions])
            std_x = np.std([p[0] for p in detected_positions])
            std_y = np.std([p[1] for p in detected_positions])

            print(f"  ✓ 像素位置: ({avg_x:.1f}, {avg_y:.1f}) ± ({std_x:.1f}, {std_y:.1f})")

            self.pixel_points.append((avg_x, avg_y))

        # 关闭激光
        self.enable_laser(False)

        # 释放资源
        if use_realsense:
            pipeline.stop()
        else:
            cap.release()

        if debug:
            cv2.destroyAllWindows()

        # 检查标定点数量
        if len(self.pixel_points) < 4:
            print(f"\n✗ 检测到的点太少 ({len(self.pixel_points)})，至少需要4个点")
            return False

        print(f"\n✓ 成功检测 {len(self.pixel_points)} 个标定点")

        # 计算单应性矩阵
        return self.calculate_homography()

    def calculate_homography(self):
        """计算单应性矩阵"""
        print("\n计算单应性矩阵...")

        # 确保点对数量一致
        n_points = min(len(self.galvo_points), len(self.pixel_points))
        galvo_pts = np.array(self.galvo_points[:n_points], dtype=np.float32)
        pixel_pts = np.array(self.pixel_points[:n_points], dtype=np.float32)

        # 计算从像素到振镜的单应性矩阵
        self.homography_matrix, mask = cv2.findHomography(
            pixel_pts, galvo_pts, cv2.RANSAC, 5.0
        )

        if self.homography_matrix is None:
            print("✗ 单应性矩阵计算失败")
            return False

        # 计算重投影误差
        galvo_pred = cv2.perspectiveTransform(
            pixel_pts.reshape(-1, 1, 2), self.homography_matrix
        ).reshape(-1, 2)

        errors = np.linalg.norm(galvo_pred - galvo_pts, axis=1)
        mean_error = np.mean(errors)
        max_error = np.max(errors)

        print(f"✓ 单应性矩阵计算成功")
        print(f"  平均重投影误差: {mean_error:.1f} DAC单位")
        print(f"  最大重投影误差: {max_error:.1f} DAC单位")

        # 计算对应的毫米误差（假设工作区域100mm对应50000 DAC）
        scale = 100.0 / 50000.0  # mm per DAC unit
        print(f"  平均误差: {mean_error * scale:.2f} mm")
        print(f"  最大误差: {max_error * scale:.2f} mm")

        return True

    def save_calibration(self, output_path='galvo_calibration.yaml'):
        """保存标定结果"""
        if self.homography_matrix is None:
            print("✗ 没有可保存的标定数据")
            return False

        data = {
            'homography_matrix': self.homography_matrix.tolist(),
            'galvo_points': self.galvo_points,
            'pixel_points': self.pixel_points,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(output_path, 'w') as f:
            yaml.dump(data, f)

        print(f"✓ 标定数据已保存: {output_path}")
        return True

    def load_calibration(self, input_path='galvo_calibration.yaml'):
        """加载标定结果"""
        try:
            with open(input_path, 'r') as f:
                data = yaml.safe_load(f)

            self.homography_matrix = np.array(data['homography_matrix'], dtype=np.float32)
            self.galvo_points = data['galvo_points']
            self.pixel_points = data['pixel_points']

            print(f"✓ 标定数据已加载: {input_path}")
            return True
        except Exception as e:
            print(f"✗ 加载标定数据失败: {e}")
            return False

    def pixel_to_galvo(self, x_pixel, y_pixel):
        """
        将像素坐标转换为振镜坐标

        Args:
            x_pixel: 像素X坐标
            y_pixel: 像素Y坐标

        Returns:
            (x_galvo, y_galvo): 振镜DAC值
        """
        if self.homography_matrix is None:
            raise RuntimeError("未进行标定，请先调用 calibrate_with_camera()")

        pixel_pt = np.array([[[x_pixel, y_pixel]]], dtype=np.float32)
        galvo_pt = cv2.perspectiveTransform(pixel_pt, self.homography_matrix)

        x_galvo = int(np.clip(galvo_pt[0][0][0], 0, 65535))
        y_galvo = int(np.clip(galvo_pt[0][0][1], 0, 65535))

        return (x_galvo, y_galvo)

    def test_calibration(self, camera_id=0):
        """测试标定精度"""
        print("\n" + "=" * 60)
        print("测试标定精度")
        print("=" * 60)
        print("点击图像中任意位置，激光将移动到该位置")
        print("按 'q' 退出测试")

        # 初始化摄像头
        try:
            import pyrealsense2 as rs
            use_realsense = True

            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            pipeline.start(config)

        except ImportError:
            use_realsense = False
            cap = cv2.VideoCapture(camera_id)

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # 转换坐标
                gx, gy = self.pixel_to_galvo(x, y)
                print(f"像素({x}, {y}) → 振镜({gx}, {gy})")

                # 移动激光
                self.enable_laser(True)
                self.send_galvo_position(gx, gy)

        cv2.namedWindow('Calibration Test')
        cv2.setMouseCallback('Calibration Test', mouse_callback)

        while True:
            if use_realsense:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                frame = np.asanyarray(color_frame.get_data())
            else:
                ret, frame = cap.read()
                if not ret:
                    break

            cv2.imshow('Calibration Test', frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        self.enable_laser(False)

        if use_realsense:
            pipeline.stop()
        else:
            cap.release()

        cv2.destroyAllWindows()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='激光振镜自动标定')
    parser.add_argument('--serial-port', default='/dev/ttyUSB0', help='串口设备')
    parser.add_argument('--baudrate', type=int, default=115200, help='波特率')
    parser.add_argument('--output', default='galvo_calibration.yaml', help='输出文件')
    parser.add_argument('--test', action='store_true', help='测试已有标定')
    parser.add_argument('--load', type=str, help='加载标定文件进行测试')
    args = parser.parse_args()

    # 创建标定器
    calibrator = GalvoCalibrator(args.serial_port, args.baudrate)

    # 连接串口
    if not calibrator.connect_serial():
        print("提示: 如果没有实际硬件，可以使用模拟模式进行测试")
        return

    if args.test or args.load:
        # 测试模式
        if args.load:
            calibrator.load_calibration(args.load)
        else:
            calibrator.load_calibration(args.output)

        calibrator.test_calibration()
    else:
        # 标定模式
        success = calibrator.calibrate_with_camera(camera_id=0, debug=True)

        if success:
            calibrator.save_calibration(args.output)
            print("\n" + "=" * 60)
            print("标定完成！")
            print(f"标定文件: {args.output}")
            print("\n运行测试:")
            print(f"  python3 {Path(__file__).name} --test --load {args.output}")
            print("=" * 60)
        else:
            print("\n✗ 标定失败")


if __name__ == '__main__':
    main()
