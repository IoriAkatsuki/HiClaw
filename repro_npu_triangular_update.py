#!/usr/bin/env python3
"""Ascend NPU triangular_update 最小复现脚本。"""

from __future__ import annotations

import argparse
import importlib
import os
import signal
import sys
import threading
import time
import traceback

import torch


def emit(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {message}", file=sys.stderr, flush=True)


def ensure_torch_npu_loaded() -> None:
    if hasattr(torch, "npu"):
        return
    importlib.import_module("torch_npu")


def synchronize_npu() -> None:
    if hasattr(torch, "npu") and hasattr(torch.npu, "synchronize"):
        torch.npu.synchronize()


def parse_shape(text: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def triangular_update(attn: torch.Tensor) -> torch.Tensor:
    for index in range(1, attn.shape[-1]):
        row = attn[..., index, :index].clone()
        sub = attn[..., :index, :index].clone()
        attn[..., index, :index] = row + (row.unsqueeze(-1) * sub).sum(-2)
    return attn


def install_timeout(seconds: int) -> None:
    def handler(signum, frame):
        emit(f"TIMEOUT after {seconds}s")
        for thread_id, stack in sys._current_frames().items():
            emit(f"--- Thread {thread_id} ---")
            for line in traceback.format_stack(stack):
                for part in line.rstrip().split("\n"):
                    emit(f"  {part}")
        os._exit(124)

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="复现 NPU triangular_update 挂起")
    parser.add_argument("--shape", default="1,1,1,64,64")
    parser.add_argument("--dtype", default="float32", choices=["float16", "float32"])
    parser.add_argument("--timeout", type=int, default=60)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_torch_npu_loaded()
    torch.npu.set_device(0)
    install_timeout(args.timeout)

    dtype = getattr(torch, args.dtype)
    shape = parse_shape(args.shape)
    emit(f"allocating attn shape={shape} dtype={dtype}")
    attn = torch.randn(shape, device="npu:0", dtype=dtype)
    synchronize_npu()
    emit("allocation synchronize ok")

    triangular_update(attn)
    emit("triangular_update launched")
    synchronize_npu()
    emit("triangular_update synchronize ok")
    signal.alarm(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
