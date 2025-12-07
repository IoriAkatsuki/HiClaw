#!/usr/bin/env python3
"""
MindYOLO 训练封装（路线 A）。

说明：封装调用 mindyolo 官方训练入口，避免直接修改上游代码。
确保已安装 mindyolo 与 mindspore，且 Ascend/CPU/GPU 环境已配置。
"""
import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="训练路线 A YOLO 模型（调用 mindyolo.tools.train）")
    parser.add_argument("--config", type=Path, default=Path("configs/route_a_yolov5n.yaml"), help="mindyolo 配置文件路径")
    parser.add_argument("--device_target", type=str, default="Ascend", choices=["Ascend", "GPU", "CPU"], help="训练设备")
    parser.add_argument("--epochs", type=int, default=None, help="可覆盖配置中的 epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="可覆盖配置中的 batch_size")
    parser.add_argument("--resume", type=str, default=None, help="断点继续训练 ckpt 路径")
    parser.add_argument("--pretrained", type=str, default=None, help="预训练权重路径")
    args = parser.parse_args()

    cmd = [
        "python",
        "-m",
        "mindyolo.tools.train",
        "--config",
        str(args.config),
        "--device_target",
        args.device_target,
    ]
    if args.epochs:
        cmd += ["--epochs", str(args.epochs)]
    if args.batch_size:
        cmd += ["--batch_size", str(args.batch_size)]
    if args.resume:
        cmd += ["--resume", args.resume]
    if args.pretrained:
        cmd += ["--pretrained", args.pretrained]

    print(f"[INFO] 调用：{' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
