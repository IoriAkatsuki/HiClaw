#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8-pose 模型下载和测试脚本
用于验证姿态估计模型功能，准备转换为 Ascend OM 格式
"""
import cv2
import numpy as np
from ultralytics import YOLO
import time

def test_yolov8_pose():
    """测试 YOLOv8-pose 模型"""
    print("=" * 60)
    print("YOLOv8-pose 模型测试")
    print("=" * 60)

    # 1. 加载模型（首次运行会自动下载）
    print("\n[1/5] 加载 YOLOv8n-pose 模型...")
    model = YOLO('yolov8n-pose.pt')
    print(f"✓ 模型类型: {model.task}")
    print(f"✓ 关键点数量: {model.model.model[-1].kpt_shape}")

    # 2. 创建测试图像（黑色背景 + 文字）
    print("\n[2/5] 创建测试图像...")
    test_img = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2.putText(test_img, "Place person in camera view", (50, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # 3. 运行推理测试
    print("\n[3/5] 运行推理测试...")
    start = time.time()
    results = model(test_img, verbose=False)
    infer_time = (time.time() - start) * 1000
    print(f"✓ 推理时间: {infer_time:.2f} ms")

    # 4. 解析输出格式
    print("\n[4/5] 分析输出格式...")
    result = results[0]
    if result.keypoints is not None:
        kpts = result.keypoints.data  # shape: [num_persons, 17, 3] (x, y, conf)
        print(f"✓ 输出形状: {kpts.shape}")
        print(f"✓ 关键点信息: 17个关键点 (COCO格式)")
        print(f"   - 手腕: 索引 9 (左), 10 (右)")
        print(f"   - 肘部: 索引 7 (左), 8 (右)")
        print(f"   - 肩部: 索引 5 (左), 6 (右)")
    else:
        print("⚠ 测试图像中未检测到人体（正常，因为是空白图像）")

    # 5. 导出 ONNX（准备转换为 OM）
    print("\n[5/5] 导出 ONNX 模型...")
    try:
        onnx_path = model.export(format='onnx', imgsz=640)
        print(f"✓ ONNX 模型已导出: {onnx_path}")

        # 检查 ONNX 文件
        import os
        if os.path.exists(onnx_path):
            size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
            print(f"✓ 文件大小: {size_mb:.2f} MB")
    except Exception as e:
        print(f"✗ ONNX 导出失败: {e}")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    print("\n下一步：")
    print("1. 使用摄像头测试姿态检测")
    print("2. 转换 ONNX → OM (Ascend)")
    print("3. 集成 RealSense D435 深度相机")

    return model

def test_with_camera(model):
    """使用摄像头测试实时姿态检测"""
    print("\n启动摄像头测试（按 'q' 退出）...")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ 无法打开摄像头")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO 推理
        results = model(frame, verbose=False)

        # 绘制关键点
        annotated = results[0].plot()

        # 显示 FPS
        cv2.putText(annotated, f"Press 'q' to quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('YOLOv8-pose Test', annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 运行测试
    model = test_yolov8_pose()

    # 询问是否进行摄像头测试
    print("\n是否使用摄像头测试实时检测？(y/n): ", end='')
    try:
        choice = input().strip().lower()
        if choice == 'y':
            test_with_camera(model)
    except KeyboardInterrupt:
        print("\n\n已取消")
