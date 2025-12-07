#!/usr/bin/env python3
"""
评估路线 A YOLO 模型（调用 mindyolo.tools.eval）。
"""
import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="评估 YOLO 模型")
    parser.add_argument("--config", type=Path, default=Path("configs/route_a_yolov5n.yaml"))
    parser.add_argument("--device_target", type=str, default="Ascend", choices=["Ascend", "GPU", "CPU"])
    parser.add_argument("--checkpoint", type=Path, required=True, help="待评估 ckpt 路径")
    args = parser.parse_args()

    cmd = [
        "python",
        "-m",
        "mindyolo.tools.eval",
        "--config",
        str(args.config),
        "--device_target",
        args.device_target,
        "--weights",
        str(args.checkpoint),
    ]

    print(f"[INFO] 调用：{' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
