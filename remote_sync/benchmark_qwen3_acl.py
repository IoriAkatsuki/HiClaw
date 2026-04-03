#!/usr/bin/env python3
"""
Qwen ACL 基准脚本。

用途：
1. 快速对比优化前后时延（首 token、平均 token、tokens/s）。
2. 输出结构化 JSON 报告用于归档。
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import yaml


def load_runtime_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="/home/HwHiAiUser/ICT/config/llm_runtime.yaml",
        help="运行配置文件路径",
    )
    parser.add_argument(
        "--prompts-file",
        help="JSON 文件，格式: [\"prompt1\", \"prompt2\", ...]",
    )
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--output", help="报告输出路径（JSON）")
    args = parser.parse_args()

    cfg = load_runtime_config(args.config)
    acl_cfg = cfg.get("acl", {})
    gen_cfg_raw = cfg.get("generation_defaults", {})

    # 延迟导入，保证在非 ACL 环境下仍可执行 --help / 参数校验
    from qwen3_acl_inference import GenerationConfig, Qwen3ACLInference

    if args.prompts_file:
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            prompts = json.load(f)
    else:
        prompts = [
            "请一句话说明香橙派 AI Pro 的优势。",
            "请解释边缘推理和云推理的区别。",
            "What is the role of KV cache in LLM inference?",
        ]

    gen_cfg = GenerationConfig(
        max_new_tokens=args.new_tokens,
        temperature=float(gen_cfg_raw.get("temperature", 0.9)),
        top_k=int(gen_cfg_raw.get("top_k", 40)),
        top_p=float(gen_cfg_raw.get("top_p", 0.9)),
        do_sample=bool(gen_cfg_raw.get("do_sample", True)),
        repetition_penalty=float(gen_cfg_raw.get("repetition_penalty", 1.0)),
    )

    engine = None
    try:
        engine = Qwen3ACLInference(
            acl_cfg.get("model_path"),
            acl_cfg.get("tokenizer_path"),
            device_id=int(acl_cfg.get("device_id", 0)),
            max_seq_len=int(acl_cfg.get("max_seq_len", 512)),
        )
        report = engine.benchmark(
            prompts,
            warmup_steps=args.warmup_steps,
            new_tokens=args.new_tokens,
            config=gen_cfg,
        )

        print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))

        output_path = args.output
        if not output_path:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"/home/HwHiAiUser/ICT/benchmark_acl_{ts}.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        engine.save_benchmark_report(report, output_path)
        print(f"[OK] 基准报告已保存: {output_path}")
    finally:
        if engine is not None:
            engine.cleanup()


if __name__ == "__main__":
    main()
