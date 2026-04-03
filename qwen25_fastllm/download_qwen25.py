#!/usr/bin/env python3
# 下载 Qwen2.5-1.5B-Instruct 模型到本地 models/ 目录。
# 默认使用 ModelScope（国内更快）；设置环境变量 QWEN_USE_HF=1 可切换到 HuggingFace。

import os
from pathlib import Path


def download_with_modelscope(model_id: str, target_dir: Path):
    from modelscope import snapshot_download

    print(f"[INFO] 使用 ModelScope 下载 {model_id} ...")
    snapshot_download(model_id, cache_dir=str(target_dir))
    print(f"[INFO] 下载完成，已保存到 {target_dir}")


def download_with_hf(model_id: str, target_dir: Path):
    from huggingface_hub import snapshot_download

    print(f"[INFO] 使用 HuggingFace Hub 下载 {model_id} ...")
    snapshot_download(model_id, local_dir=str(target_dir))
    print(f"[INFO] 下载完成，已保存到 {target_dir}")


def main():
    model_id = "qwen/Qwen2.5-1.5B-Instruct"
    base_dir = Path(__file__).resolve().parent
    target_dir = base_dir / "models"

    use_hf = os.getenv("QWEN_USE_HF", "0") == "1"
    target_dir.mkdir(parents=True, exist_ok=True)

    if use_hf:
        download_with_hf(model_id, target_dir)
    else:
        download_with_modelscope(model_id, target_dir)


if __name__ == "__main__":
    main()
