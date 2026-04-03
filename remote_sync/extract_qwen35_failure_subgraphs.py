#!/usr/bin/env python3
"""提取 Qwen3.5 失败节点的最小复现子图。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
from onnx import shape_inference, utils

DEFAULT_TARGETS = [
    "/model/model/layers.0/linear_attn/Mul_39",
    "/model/model/layers.0/linear_attn/Mul_45",
    "/model/model/layers.0/linear_attn/Mul_51",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="提取 Qwen3.5 编译失败节点的最小复现子图")
    parser.add_argument("--input", required=True, help="输入 ONNX 路径")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument(
        "--targets",
        nargs="*",
        default=DEFAULT_TARGETS,
        help="失败节点名，默认使用已知的 Mul_39/45/51",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=0,
        help="向上回溯的节点深度，0 表示只保留目标节点本身",
    )
    parser.add_argument(
        "--emit-combined",
        action="store_true",
        help="额外导出一个包含全部目标节点的合并子图",
    )
    return parser.parse_args()


class GraphIndex:
    def __init__(self, model: onnx.ModelProto) -> None:
        self.model = model
        self.node_by_name = {node.name: node for node in model.graph.node}
        self.producer_by_tensor: dict[str, onnx.NodeProto] = {}
        for node in model.graph.node:
            for output_name in node.output:
                if output_name:
                    self.producer_by_tensor[output_name] = node
        self.graph_inputs = {value.name for value in model.graph.input}
        self.initializers = {value.name for value in model.graph.initializer}

    def collect_boundary_inputs(self, target_names: list[str], depth: int) -> tuple[list[str], list[str]]:
        included = set(target_names)
        frontier = list(target_names)
        for _ in range(depth):
            next_frontier: list[str] = []
            for node_name in frontier:
                node = self.node_by_name[node_name]
                for input_name in node.input:
                    producer = self.producer_by_tensor.get(input_name)
                    if producer is None:
                        continue
                    if producer.name not in included:
                        included.add(producer.name)
                        next_frontier.append(producer.name)
            frontier = next_frontier

        boundary_inputs: set[str] = set()
        for node_name in included:
            node = self.node_by_name[node_name]
            for input_name in node.input:
                producer = self.producer_by_tensor.get(input_name)
                if producer is None:
                    if input_name in self.graph_inputs:
                        boundary_inputs.add(input_name)
                    elif input_name in self.initializers:
                        continue
                    else:
                        boundary_inputs.add(input_name)
                elif producer.name not in included:
                    boundary_inputs.add(input_name)

        outputs: list[str] = []
        for node_name in target_names:
            outputs.extend(self.node_by_name[node_name].output)
        return sorted(boundary_inputs), outputs



def sanitize(name: str) -> str:
    return name.strip("/").replace("/", "__")



def load_shape_inferred_model(path: Path) -> onnx.ModelProto:
    model = onnx.load(str(path), load_external_data=False)
    return shape_inference.infer_shapes(model)



def extract_one(model: onnx.ModelProto, index: GraphIndex, target_names: list[str], depth: int, output_path: Path) -> dict:
    inputs, outputs = index.collect_boundary_inputs(target_names, depth)
    extractor = utils.Extractor(model)
    submodel = extractor.extract_model(inputs, outputs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(submodel, str(output_path))
    return {
        "targets": target_names,
        "depth": depth,
        "inputs": inputs,
        "outputs": outputs,
        "node_count": len(submodel.graph.node),
        "initializer_count": len(submodel.graph.initializer),
        "path": str(output_path),
    }



def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_shape_inferred_model(input_path)
    index = GraphIndex(model)

    missing = [name for name in args.targets if name not in index.node_by_name]
    if missing:
        raise SystemExit(f"找不到目标节点: {missing}")

    report: list[dict] = []
    for target_name in args.targets:
        output_path = output_dir / f"{sanitize(target_name)}.depth{args.depth}.onnx"
        report.append(extract_one(model, index, [target_name], args.depth, output_path))

    if args.emit_combined:
        combined_path = output_dir / f"linear_attn_mul_combined.depth{args.depth}.onnx"
        report.append(extract_one(model, index, list(args.targets), args.depth, combined_path))

    report_path = output_dir / f"extract_report.depth{args.depth}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[extract] report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
