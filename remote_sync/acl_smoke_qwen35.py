#!/usr/bin/env python3
"""ACL smoke test for a generated OM model."""

from __future__ import annotations

import argparse
import json
import sys

from qwen3_acl_inference import GenerationConfig, Qwen3ACLInference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ACL smoke test for Qwen OM model")
    parser.add_argument("--model-path", required=True, help="OM model path")
    parser.add_argument("--tokenizer-path", required=True, help="Tokenizer/model directory")
    parser.add_argument("--max-seq-len", type=int, required=True, help="Static seq len used by OM")
    parser.add_argument("--prompt", default="你好，请简要介绍 Ascend 310B。", help="Smoke prompt")
    parser.add_argument("--max-new-tokens", type=int, default=8, help="Generate token count")
    parser.add_argument("--device-id", type=int, default=0, help="ACL device id")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    engine = None
    try:
        engine = Qwen3ACLInference(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            device_id=args.device_id,
            max_seq_len=args.max_seq_len,
        )
        cfg = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            top_k=0,
            top_p=1.0,
            do_sample=False,
            repetition_penalty=1.0,
        )
        text, total_s = engine.generate_text(args.prompt, config=cfg)
        report = {
            "model_path": args.model_path,
            "tokenizer_path": args.tokenizer_path,
            "max_seq_len": args.max_seq_len,
            "prompt": args.prompt,
            "response": text,
            "total_time_s": total_s,
            "stats": engine.last_generation_stats,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] failed: {exc}", file=sys.stderr)
        raise
    finally:
        if engine is not None:
            engine.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())

