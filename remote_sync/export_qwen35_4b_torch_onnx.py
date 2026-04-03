#!/usr/bin/env python3
"""Export Qwen3.5-4B CausalLM to ONNX on board.

This script intentionally uses the legacy ONNX tracer (dynamo=False)
because torch.export currently fails on data-dependent guards for Qwen3.5.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


class ExportWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, position_ids):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=False,
        )
        return outputs[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Qwen3.5-4B ONNX")
    parser.add_argument(
        "--model-path",
        default="/home/HwHiAiUser/ICT/models/qwen3.5-4b",
        help="HF model directory on board",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output ONNX file path",
    )
    parser.add_argument("--seq-len", type=int, default=128, help="Static sequence length")
    parser.add_argument("--batch-size", type=int, default=1, help="Static batch size")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset")
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Model weights dtype while exporting",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model_dtype = torch.float16 if args.dtype == "float16" else torch.float32

    print("[export] model_path:", args.model_path)
    print("[export] output:", str(out_path))
    print("[export] batch:", args.batch_size, "seq:", args.seq_len, "opset:", args.opset)
    print("[export] dtype:", args.dtype)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        dtype=model_dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.config.use_cache = False

    wrapper = ExportWrapper(model).eval()

    input_ids = torch.zeros((args.batch_size, args.seq_len), dtype=torch.long)
    attention_mask = torch.ones((args.batch_size, args.seq_len), dtype=torch.long)
    position_ids = torch.arange(args.seq_len, dtype=torch.long).unsqueeze(0).repeat(args.batch_size, 1)

    torch.onnx.export(
        wrapper,
        (input_ids, attention_mask, position_ids),
        str(out_path),
        input_names=["input_ids", "attention_mask", "position_ids"],
        output_names=["logits"],
        opset_version=args.opset,
        do_constant_folding=False,
        dynamic_axes=None,
        export_params=True,
        dynamo=False,
    )

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError("ONNX export finished but output file is missing or empty")

    print("[export] done, bytes:", out_path.stat().st_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

