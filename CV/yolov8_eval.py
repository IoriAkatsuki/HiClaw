#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PC 端离线图片/文件夹 YOLOv8 检测脚本

用途：
- 对 ElectroCom61 数据集等任意图片/目录做可视化检测；
- 便于直观查看当前模型在真实数据上的效果。
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 图片/文件夹检测")
    parser.add_argument(
        "--model",
        type=str,
        default="runs/detect/train_electro61/weights/best.pt",
        help="YOLOv8 模型权重路径（.pt）",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="待检测图片或目录路径",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="推理输入尺寸",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="置信度阈值",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="推理设备，如 '0' 或 'cpu'",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="CV/vis_output",
        help="可视化结果保存目录",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 加载模型: {args.model}")
    model = YOLO(args.model)

    print(f"[INFO] 开始检测: source={args.source}")
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save=True,
        project=str(out_dir),
        name="",
        exist_ok=True,
        verbose=True,
    )

    print(f"[INFO] 检测完成，结果保存在: {out_dir}")


if __name__ == "__main__":
    main()

