#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PC 端实时摄像头 YOLOv8 检测脚本

用途：
- 使用你训练好的 ElectroCom61 模型，在本机摄像头上做实时检测；
- 观察检测效果与 FPS，为后续板卡端部署提供参考。
"""

import argparse
import time

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 摄像头实时检测")
    parser.add_argument(
        "--model",
        type=str,
        default="runs/detect/train_electro61/weights/best.pt",
        help="YOLOv8 模型权重路径（.pt）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="推理设备，如 '0'（第0块GPU）或 'cpu'",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="推理输入尺寸（方形）",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="置信度阈值",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="摄像头索引（默认 0）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[INFO] 加载模型: {args.model}, device={args.device}")
    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 {args.camera}")

    prev_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] 读取摄像头失败，重试中...")
            time.sleep(0.05)
            continue

        # Ultralytics YOLO 支持直接传入 numpy 数组
        results = model.predict(
            source=frame,
            device=args.device,
            imgsz=args.imgsz,
            conf=args.conf,
            verbose=False,
        )

        # 取第一帧结果并画框
        annotated = results[0].plot()  # 返回带框的 BGR 图像

        frame_count += 1
        now = time.time()
        dt = now - prev_time
        if dt >= 1.0:
            fps = frame_count / dt
            prev_time = now
            frame_count = 0
            cv2.putText(
                annotated,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )

        cv2.imshow("YOLOv8 ElectroCom61 - Camera", annotated)

        # 按 q 退出
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

