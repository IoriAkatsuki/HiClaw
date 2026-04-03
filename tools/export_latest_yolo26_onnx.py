#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""导出最新 yolo26 权重到 ONNX。"""

from __future__ import annotations

import argparse
import importlib.metadata as importlib_metadata
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENDOR_DIR = ROOT / "2026_3_12" / ".vendor_ultralytics_8419"
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))

_original_version = importlib_metadata.version


def _safe_version(package_name: str) -> str:
    try:
        return _original_version(package_name)
    except importlib_metadata.PackageNotFoundError:
        if package_name == "torchvision":
            # 某些板卡环境只装了 torch，本处仅用于通过 ultralytics 的版本探测。
            return "0.0.0"
        raise


importlib_metadata.version = _safe_version

from ultralytics import YOLO  # noqa: E402


DEFAULT_WEIGHTS = ROOT / "2026_3_12" / "runs" / "train" / "yolo26n_aug_full_8419_gpu" / "weights" / "best.pt"
DEFAULT_OUTPUT = ROOT / "models" / "route_a_yolo26" / "yolo26n_aug_full_8419_gpu.onnx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出 2026_3_12 最新 yolo26 权重到 ONNX。")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="输入 PT 权重路径。")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出 ONNX 路径。")
    parser.add_argument("--imgsz", type=int, default=640, help="导出输入尺寸。")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset。")
    parser.add_argument("--device", default="cpu", help="导出时使用的设备，例如 cpu / 0。")
    parser.add_argument("--simplify", action="store_true", default=True, help="是否简化 ONNX 图。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = args.weights.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if not weights.exists():
        raise FileNotFoundError(f"未找到权重文件: {weights}")

    model = YOLO(str(weights))
    exported = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=args.simplify,
        device=args.device,
    )

    exported_path = Path(str(exported)).resolve()
    if not exported_path.exists():
        raise FileNotFoundError(f"Ultralytics 未生成预期 ONNX 文件: {exported_path}")

    if exported_path != output:
        shutil.copy2(exported_path, output)

    print(f"[export] weights={weights}")
    print(f"[export] onnx={output}")


if __name__ == "__main__":
    main()
