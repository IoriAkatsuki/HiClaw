#!/usr/bin/env python3
"""
将训练好的 ckpt 导出为 MindIR（调用 mindyolo.tools.export 或手动 export）。
优先尝试 mindyolo.tools.export；若未安装，请补充手动导出逻辑。
"""
import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="导出 MindIR")
    parser.add_argument("--config", type=Path, default=Path("configs/route_a_yolov5n.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True, help="ckpt 路径")
    parser.add_argument("--device_target", type=str, default="Ascend", choices=["Ascend", "GPU", "CPU"])
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--file-name", type=str, default="route_a_yolov5n")
    args = parser.parse_args()

    cmd = [
        "python",
        "-m",
        "mindyolo.tools.export",
        "--config",
        str(args.config),
        "--device_target",
        args.device_target,
        "--weights",
        str(args.checkpoint),
        "--img_size",
        str(args.img_size),
        "--file_name",
        args.file_name,
    ]

    print(f"[INFO] 调用：{' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
