#!/usr/bin/env python3
"""修补 Qwen3.5 ONNX 中不兼容的双输入 Trilu 节点。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="修补 Qwen3.5 ONNX 图中的 Trilu 兼容性问题")
    parser.add_argument("--input", required=True, help="原始 ONNX 路径")
    parser.add_argument("--output", required=True, help="修补后的 ONNX 路径")
    parser.add_argument(
        "--shape-inference",
        action="store_true",
        help="额外执行 shape inference（更耗内存）",
    )
    return parser.parse_args()


def _read_constant_map(model: onnx.ModelProto) -> dict[str, Any]:
    constants: dict[str, Any] = {}

    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name != "value":
                continue
            try:
                constants[node.output[0]] = numpy_helper.to_array(attr.t)
            except Exception:  # noqa: BLE001
                pass
            break

    return constants


def _is_zero_scalar(value: Any) -> bool:
    array = np.asarray(value)
    return array.size == 1 and int(array.reshape(()).item()) == 0


def patch_trilu(model: onnx.ModelProto) -> tuple[onnx.ModelProto, dict[str, Any]]:
    constants = _read_constant_map(model)
    patched_nodes = 0
    skipped_nodes: list[dict[str, Any]] = []

    for node in model.graph.node:
        if node.op_type != "Trilu":
            continue

        if len(node.input) != 2:
            skipped_nodes.append(
                {"name": node.name, "reason": f"unexpected_input_count={len(node.input)}"}
            )
            continue

        upper = 0
        for attr in node.attribute:
            if attr.name == "upper":
                upper = int(helper.get_attribute_value(attr))
                break

        second_input_name = node.input[1]
        second_value = constants.get(second_input_name)
        if upper != 0 or second_value is None or not _is_zero_scalar(second_value):
            skipped_nodes.append(
                {
                    "name": node.name,
                    "reason": "unsupported_trilu_signature",
                    "upper": upper,
                    "second_input": second_input_name,
                    "second_value": None if second_value is None else np.asarray(second_value).tolist(),
                }
            )
            continue

        del node.input[1]
        patched_nodes += 1

    report = {
        "trilu_total": patched_nodes + len(skipped_nodes),
        "trilu_patched": patched_nodes,
        "trilu_skipped": len(skipped_nodes),
        "skipped_nodes": skipped_nodes,
    }
    return model, report


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[patch] input={input_path}")
    print(f"[patch] output={output_path}")

    model = onnx.load(str(input_path), load_external_data=False)
    model, report = patch_trilu(model)

    if report["trilu_patched"] == 0:
        raise RuntimeError(f"未找到可修补的 Trilu 节点: {json.dumps(report, ensure_ascii=False)}")
    if report["trilu_skipped"] != 0:
        raise RuntimeError(f"存在未覆盖的 Trilu 形态: {json.dumps(report, ensure_ascii=False)}")

    if args.shape_inference:
        try:
            inferred_model = shape_inference.infer_shapes(model)
            model = inferred_model
            print("[patch] shape_inference=ok")
        except Exception as exc:  # noqa: BLE001
            print(f"[patch] shape_inference=skip reason={exc}")
            raise

    onnx.save(model, str(output_path))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[patch] done bytes={output_path.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
