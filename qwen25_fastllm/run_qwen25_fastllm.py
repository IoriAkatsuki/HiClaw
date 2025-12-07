#!/usr/bin/env python3
# 基于 FastLLM 的最小 Qwen2.5-1.5B 推理示例，目标：Orange Pi AI Pro (310B)。
# 运行前：
# 1) `source /usr/local/Ascend/ascend-toolkit/set_env.sh`（可选，当前版本主要走 CPU）
# 2) 在本目录执行过 fastllm 源码编译：
#    cd fastllm_src && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j2
# 3) 运行 download_qwen25.py 下载模型

import os
import sys
from pathlib import Path
from typing import Iterable


def _bootstrap_fastllm(base_dir: Path):
    """
    将 FastLLM 编译产物加入 sys.path。
    目录要求：
    - {base}/fastllm_src/build/tools/ftllm/  内含 libfastllm_tools.so 和 llm.py 等
    """
    ftllm_parent = base_dir / "fastllm_src" / "build" / "tools"
    sys.path.insert(0, str(ftllm_parent))


def stream_print(chunks: Iterable[str]):
    """流式打印，适合 fastllm 的 stream_response。"""
    for c in chunks:
        print(c, end="", flush=True)
    print()


def main():
    base_dir = Path(__file__).resolve().parent
    _bootstrap_fastllm(base_dir)

    try:
        from ftllm import llm  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"导入 ftllm 失败，请确认已在 fastllm_src/build 下完成编译: {exc}")

    default_model = base_dir / "models" / "qwen" / "Qwen2.5-1.5B-Instruct"
    model_path = Path(os.getenv("QWEN_MODEL_PATH", default_model))

    if not model_path.exists():
        raise FileNotFoundError(f"模型路径不存在: {model_path}，请先运行 download_qwen25.py")

    print(f"[INFO] 将加载模型: {model_path}")
    # dtype=\"float16\"：在 24GB 内存上跑 1.5B 完全足够
    model = llm.model(str(model_path), dtype="float16")
    print("[INFO] 模型加载完成，FastLLM 后端已就绪。")  # 当前仓库以 CPU 为主，后续如有 Ascend 分支可在此扩展

    warmup_prompt = "请用一句话介绍一下香橙派 AI Pro 的 NPU。"
    print(f"[INFO] 发送热身请求: {warmup_prompt}")
    response = []
    model.response(warmup_prompt, lambda x: response.append(x))
    print("AI> " + "".join(response))

    while True:
        try:
            user = input("User> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] 已退出。")  # pragma: no cover
            break

        if user.lower() in {"exit", "quit"}:
            print("[INFO] 已退出。")  # pragma: no cover
            break
        if not user:
            continue

        print("AI> ", end="", flush=True)
        model.response(user, lambda x: print(x, end="", flush=True))
        print()


if __name__ == "__main__":
    main()
