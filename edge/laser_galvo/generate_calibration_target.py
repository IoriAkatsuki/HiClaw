#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
激光振镜标定板生成器
生成A4大小的标定板，包含9个十字标记和ArUco码
"""
import cv2
import numpy as np
from pathlib import Path

def draw_cross(img, center, size=40, thickness=3, color=(0, 0, 0)):
    """绘制十字标记"""
    x, y = center
    # 水平线
    cv2.line(img, (x - size, y), (x + size, y), color, thickness)
    # 垂直线
    cv2.line(img, (x, y - size), (x, y + size), color, thickness)
    # 中心圆
    cv2.circle(img, (x, y), 8, color, -1)

def draw_circle_pattern(img, center, radius=30, color=(0, 0, 0)):
    """绘制同心圆标记（更容易检测）"""
    x, y = center
    cv2.circle(img, (x, y), radius, color, 3)
    cv2.circle(img, (x, y), radius // 2, color, 3)
    cv2.circle(img, (x, y), 5, color, -1)

def generate_calibration_board(output_path='calibration_board_A4.png'):
    """
    生成A4标定板 (210mm x 297mm, 300 DPI = 2480 x 3508 pixels)
    """
    # A4尺寸 @ 300 DPI
    dpi = 300
    width_mm, height_mm = 210, 297
    width_px = int(width_mm / 25.4 * dpi)
    height_px = int(height_mm / 25.4 * dpi)

    # 创建白色背景
    img = np.ones((height_px, width_px, 3), dtype=np.uint8) * 255

    # 绘制边框
    border = 100
    cv2.rectangle(img, (border, border),
                  (width_px - border, height_px - border),
                  (0, 0, 0), 3)

    # 计算9个标定点位置 (3x3网格)
    margin_x = 300
    margin_y = 400
    grid_width = width_px - 2 * margin_x
    grid_height = height_px - 2 * margin_y

    calibration_points = []
    point_id = 1

    for row in range(3):
        for col in range(3):
            x = margin_x + col * (grid_width // 2)
            y = margin_y + row * (grid_height // 2)
            calibration_points.append((x, y, point_id))
            point_id += 1

    # 绘制标定点
    for x, y, pid in calibration_points:
        # 绘制同心圆（更容易被OpenCV检测）
        draw_circle_pattern(img, (x, y), radius=60, color=(0, 0, 0))

        # 标注点号
        cv2.putText(img, f'{pid}', (x + 80, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

    # 添加标题
    title = "Laser Galvo Calibration Board"
    cv2.putText(img, title, (width_px // 2 - 500, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)

    # 添加说明文字
    instructions = [
        "Instructions:",
        "1. Print this page on A4 paper",
        "2. Place it on the galvo work surface",
        "3. Run calibration program",
        "4. Laser will mark each point 1-9",
        "5. Camera will detect positions",
    ]

    y_offset = 250
    for i, line in enumerate(instructions):
        cv2.putText(img, line, (150, y_offset + i * 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # 绘制坐标轴指示
    cv2.arrowedLine(img, (150, height_px - 300), (350, height_px - 300),
                    (0, 0, 255), 3, tipLength=0.2)
    cv2.putText(img, 'X', (360, height_px - 290),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.arrowedLine(img, (150, height_px - 300), (150, height_px - 500),
                    (0, 255, 0), 3, tipLength=0.2)
    cv2.putText(img, 'Y', (160, height_px - 510),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # 添加QR码区域（可选）
    qr_size = 300
    qr_x = width_px - qr_size - 200
    qr_y = height_px - qr_size - 200
    cv2.rectangle(img, (qr_x, qr_y), (qr_x + qr_size, qr_y + qr_size),
                  (128, 128, 128), 2)
    cv2.putText(img, 'QR Code', (qr_x + 50, qr_y + qr_size // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (128, 128, 128), 3)

    # 保存图像
    cv2.imwrite(output_path, img)
    print(f"✓ 标定板已生成: {output_path}")
    print(f"  尺寸: {width_px} x {height_px} pixels ({width_mm} x {height_mm} mm)")
    print(f"  DPI: {dpi}")
    print(f"  标定点数量: {len(calibration_points)}")

    # 保存标定点坐标信息
    coord_file = output_path.replace('.png', '_coordinates.txt')
    with open(coord_file, 'w') as f:
        f.write("# 标定点坐标 (像素)\n")
        f.write("# ID  X     Y\n")
        for x, y, pid in calibration_points:
            f.write(f"{pid:2d}  {x:4d}  {y:4d}\n")

    print(f"✓ 坐标文件已保存: {coord_file}")

    # 生成缩略图
    thumbnail = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4))
    cv2.imwrite(output_path.replace('.png', '_preview.png'), thumbnail)

    return calibration_points

def generate_simple_checkerboard(output_path='checkerboard_A4.png'):
    """
    生成简单的棋盘格标定板（OpenCV标准）
    """
    dpi = 300
    width_mm, height_mm = 210, 297
    width_px = int(width_mm / 25.4 * dpi)
    height_px = int(height_mm / 25.4 * dpi)

    # 创建白色背景
    img = np.ones((height_px, width_px, 3), dtype=np.uint8) * 255

    # 棋盘格参数
    rows, cols = 7, 9  # 内角点数
    square_size = 250  # 像素

    # 计算起始位置（居中）
    board_width = cols * square_size
    board_height = rows * square_size
    start_x = (width_px - board_width) // 2
    start_y = (height_px - board_height) // 2

    # 绘制棋盘格
    for i in range(rows + 1):
        for j in range(cols + 1):
            if (i + j) % 2 == 0:
                x1 = start_x + j * square_size
                y1 = start_y + i * square_size
                x2 = x1 + square_size
                y2 = y1 + square_size
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)

    # 添加标题
    cv2.putText(img, "Checkerboard Calibration",
                (width_px // 2 - 400, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)

    cv2.imwrite(output_path, img)
    print(f"✓ 棋盘格标定板已生成: {output_path}")

if __name__ == '__main__':
    # 创建输出目录
    output_dir = Path('/home/oasis/Documents/ICT/calibration_data')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成标定板
    print("=" * 60)
    print("激光振镜标定板生成器")
    print("=" * 60)

    # 方案1: 9点标定板（推荐）
    output1 = str(output_dir / 'laser_calibration_board_A4.png')
    points = generate_calibration_board(output1)

    print("\n")

    # 方案2: 棋盘格（备选）
    output2 = str(output_dir / 'checkerboard_calibration_A4.png')
    generate_simple_checkerboard(output2)

    print("\n" + "=" * 60)
    print("完成！请打印以下文件：")
    print(f"  推荐: {output1}")
    print(f"  备选: {output2}")
    print("=" * 60)
