#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为 AMCT PTQ 量化准备校准数据集。

从训练数据目录随机采样图片，预处理为 NCHW float32 并保存为 .npy 文件。
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def preprocess(img_path: Path, imgsz: int) -> np.ndarray:
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"无法读取图片: {img_path}")
    img = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))          # HWC → CHW
    img = np.expand_dims(img, axis=0)             # → NCHW (N=1)
    return img


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备 AMCT 校准数据集")
    parser.add_argument("--input-dir", type=Path, default=Path("2026_3_12"), help="图片搜索根目录")
    parser.add_argument("--output-dir", type=Path, default=Path("tools/amct_calibration_data"), help="输出 .npy 目录")
    parser.add_argument("--num-samples", type=int, default=150, help="采样数量")
    parser.add_argument("--imgsz", type=int, default=640, help="预处理尺寸")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    input_dir = (root / args.input_dir).resolve()
    output_dir = (root / args.output_dir).resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    all_images = [p for p in input_dir.rglob("*") if p.suffix.lower() in IMG_SUFFIXES]
    if not all_images:
        raise RuntimeError(f"在 {input_dir} 中未找到任何图片")

    random.seed(args.seed)
    samples = random.sample(all_images, min(args.num_samples, len(all_images)))

    output_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    for i, img_path in enumerate(samples):
        try:
            arr = preprocess(img_path, args.imgsz)
            out_path = output_dir / f"calib_{i:04d}.npy"
            np.save(str(out_path), arr)
            ok += 1
        except Exception as exc:
            print(f"[WARN] 跳过 {img_path.name}: {exc}")

    print(f"[prepare] 完成: {ok}/{len(samples)} 张图片 → {output_dir}")
    print(f"[prepare] 数组格式: (1, 3, {args.imgsz}, {args.imgsz})  dtype=float32")


if __name__ == "__main__":
    main()
