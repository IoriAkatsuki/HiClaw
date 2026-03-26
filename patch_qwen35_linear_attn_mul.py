#!/usr/bin/env python3
"""为 Qwen3.5 linear_attn 子图中的 Mul 做 ATC 兼容性预处理。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="修补 Qwen3.5 linear_attn 子图中的 Mul 兼容性问题")
    parser.add_argument("--input", required=True, help="原始 ONNX 路径")
    parser.add_argument("--output", required=True, help="修补后的 ONNX 路径")
    parser.add_argument(
        "--shape-inference",
        action="store_true",
        help="修补前额外执行一次 shape inference",
    )
    return parser.parse_args(argv)


def _collect_value_info(model: onnx.ModelProto) -> dict[str, dict[str, Any]]:
    info: dict[str, dict[str, Any]] = {}
    for init in model.graph.initializer:
        array = numpy_helper.to_array(init)
        info[init.name] = {
            "elem_type": init.data_type,
            "shape": list(array.shape),
        }

    value_infos = list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)
    for item in value_infos:
        tensor_type = item.type.tensor_type
        if tensor_type.elem_type == 0:
            continue
        dims: list[Any] = []
        for dim in tensor_type.shape.dim:
            if dim.dim_param:
                dims.append(dim.dim_param)
            elif dim.dim_value:
                dims.append(dim.dim_value)
            else:
                dims.append(None)
        info[item.name] = {
            "elem_type": tensor_type.elem_type,
            "shape": dims,
        }
    return info


def _collect_constant_values(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    constants: dict[str, np.ndarray] = {}
    for init in model.graph.initializer:
        constants[init.name] = numpy_helper.to_array(init)
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name == "value":
                constants[node.output[0]] = numpy_helper.to_array(attr.t)
                break
    return constants


def _is_float_elem(elem_type: int) -> bool:
    return elem_type in {
        TensorProto.FLOAT,
        TensorProto.FLOAT16,
        TensorProto.BFLOAT16,
        TensorProto.DOUBLE,
    }


def _is_scalar_shape(shape: list[Any]) -> bool:
    return len(shape) == 0


def _make_scalar_int64(name: str, value: int) -> onnx.TensorProto:
    return helper.make_tensor(name=name, data_type=TensorProto.INT64, dims=[1], vals=[value])


def _tensor_info(info_map: dict[str, dict[str, Any]], name: str) -> tuple[int | None, list[Any] | None]:
    info = info_map.get(name)
    if not info:
        return (None, None)
    return (info["elem_type"], info["shape"])


def _sanitize(name: str) -> str:
    return name.replace("/", "_").replace(".", "_")


def _same_shape(shape_a: list[Any] | None, shape_b: list[Any] | None) -> bool:
    if shape_a is None or shape_b is None or len(shape_a) != len(shape_b):
        return False
    return all(a == b for a, b in zip(shape_a, shape_b))


def _build_reshape_5d_mul(
    node: onnx.NodeProto,
    value_info: dict[str, dict[str, Any]],
) -> tuple[list[onnx.NodeProto], list[onnx.TensorProto]]:
    base = _sanitize(node.name or node.output[0])
    first_input = node.input[0]
    second_input = node.input[1]

    shape_node = helper.make_node("Shape", [first_input], [f"{base}_shape"], name=f"{base}_shape")
    gather0 = helper.make_node(
        "Gather",
        [f"{base}_shape", f"{base}_idx0"],
        [f"{base}_dim0"],
        axis=0,
        name=f"{base}_gather0",
    )
    gather1 = helper.make_node(
        "Gather",
        [f"{base}_shape", f"{base}_idx1"],
        [f"{base}_dim1"],
        axis=0,
        name=f"{base}_gather1",
    )
    gather2 = helper.make_node(
        "Gather",
        [f"{base}_shape", f"{base}_idx2"],
        [f"{base}_dim2"],
        axis=0,
        name=f"{base}_gather2",
    )
    gather3 = helper.make_node(
        "Gather",
        [f"{base}_shape", f"{base}_idx3"],
        [f"{base}_dim3"],
        axis=0,
        name=f"{base}_gather3",
    )
    gather4 = helper.make_node(
        "Gather",
        [f"{base}_shape", f"{base}_idx4"],
        [f"{base}_dim4"],
        axis=0,
        name=f"{base}_gather4",
    )
    merge_mid = helper.make_node(
        "Mul",
        [f"{base}_dim1", f"{base}_dim2"],
        [f"{base}_merged_mid"],
        name=f"{base}_merge_mid",
    )
    merged_shape = helper.make_node(
        "Concat",
        [f"{base}_dim0", f"{base}_merged_mid", f"{base}_dim3", f"{base}_dim4"],
        [f"{base}_shape4d"],
        axis=0,
        name=f"{base}_concat_shape4d",
    )
    reshape_a = helper.make_node(
        "Reshape",
        [first_input, f"{base}_shape4d"],
        [f"{base}_a4d"],
        name=f"{base}_reshape_a",
    )
    reshape_b = helper.make_node(
        "Reshape",
        [second_input, f"{base}_shape4d"],
        [f"{base}_b4d"],
        name=f"{base}_reshape_b",
    )
    mul4d = helper.make_node(
        "Mul",
        [f"{base}_a4d", f"{base}_b4d"],
        [f"{base}_mul4d"],
        name=f"{base}_mul4d",
    )
    reshape_back = helper.make_node(
        "Reshape",
        [f"{base}_mul4d", f"{base}_shape"],
        [node.output[0]],
        name=f"{base}_reshape_back",
    )

    initializers = [
        _make_scalar_int64(f"{base}_idx0", 0),
        _make_scalar_int64(f"{base}_idx1", 1),
        _make_scalar_int64(f"{base}_idx2", 2),
        _make_scalar_int64(f"{base}_idx3", 3),
        _make_scalar_int64(f"{base}_idx4", 4),
    ]
    nodes = [
        shape_node,
        gather0,
        gather1,
        gather2,
        gather3,
        gather4,
        merge_mid,
        merged_shape,
        reshape_a,
        reshape_b,
        mul4d,
        reshape_back,
    ]
    return nodes, initializers


def _build_int64_cast_mul(node: onnx.NodeProto) -> tuple[list[onnx.NodeProto], list[onnx.TensorProto]]:
    base = _sanitize(node.name or node.output[0])
    cast_a = helper.make_node(
        "Cast",
        [node.input[0]],
        [f"{base}_a_int32"],
        to=TensorProto.INT32,
        name=f"{base}_cast_a",
    )
    cast_b = helper.make_node(
        "Cast",
        [node.input[1]],
        [f"{base}_b_int32"],
        to=TensorProto.INT32,
        name=f"{base}_cast_b",
    )
    mul_int32 = helper.make_node(
        "Mul",
        [f"{base}_a_int32", f"{base}_b_int32"],
        [f"{base}_mul_int32"],
        name=f"{base}_mul_int32",
    )
    cast_out = helper.make_node(
        "Cast",
        [f"{base}_mul_int32"],
        [node.output[0]],
        to=TensorProto.INT64,
        name=f"{base}_cast_out",
    )
    return [cast_a, cast_b, mul_int32, cast_out], []


def _build_expand_mul(
    node: onnx.NodeProto,
    scalar_input: str,
    tensor_input: str,
) -> tuple[list[onnx.NodeProto], list[onnx.TensorProto]]:
    base = _sanitize(node.name or node.output[0])
    shape_node = helper.make_node("Shape", [tensor_input], [f"{base}_target_shape"], name=f"{base}_shape")
    expand_node = helper.make_node(
        "Expand",
        [scalar_input, f"{base}_target_shape"],
        [f"{base}_expanded_scalar"],
        name=f"{base}_expand",
    )
    mul_node = helper.make_node(
        "Mul",
        [tensor_input, f"{base}_expanded_scalar"],
        [node.output[0]],
        name=f"{base}_mul_expanded",
    )
    return [shape_node, expand_node, mul_node], []


def _build_tiled_add(
    node: onnx.NodeProto,
    broadcast_input: str,
    stable_input: str,
    last_dim: int,
    broadcast_on_rhs: bool,
) -> tuple[list[onnx.NodeProto], list[onnx.TensorProto]]:
    base = _sanitize(node.name or node.output[0])
    repeats_name = f"{base}_tile_repeats"
    tiled_name = f"{base}_tiled"
    tile_node = helper.make_node(
        "Tile",
        [broadcast_input, repeats_name],
        [tiled_name],
        name=f"{base}_tile",
    )
    add_inputs = [stable_input, tiled_name] if broadcast_on_rhs else [tiled_name, stable_input]
    add_node = helper.make_node(
        "Add",
        add_inputs,
        [node.output[0]],
        name=node.name,
    )
    repeats = [1, 1, 1, last_dim]
    initializer = helper.make_tensor(
        name=repeats_name,
        data_type=TensorProto.INT64,
        dims=[4],
        vals=repeats,
    )
    return [tile_node, add_node], [initializer]


def patch_model(model: onnx.ModelProto) -> tuple[onnx.ModelProto, dict[str, Any]]:
    info_map = _collect_value_info(model)
    report = {
        "mul_total": 0,
        "mul_5d_patched": 0,
        "mul_int64_patched": 0,
        "mul_scalar_expand_patched": 0,
        "expand_identity_removed": 0,
        "add_broadcast_tile_patched": 0,
        "mul_skipped": 0,
        "skipped_nodes": [],
    }

    new_nodes: list[onnx.NodeProto] = []
    new_initializers: list[onnx.TensorProto] = []
    replace_tensor: dict[str, str] = {}

    for node in model.graph.node:
        rewritten_inputs = [replace_tensor.get(inp, inp) for inp in node.input]
        if rewritten_inputs != list(node.input):
            del node.input[:]
            node.input.extend(rewritten_inputs)

        if node.op_type == "Expand":
            input_type, input_shape = _tensor_info(info_map, node.input[0])
            output_type, output_shape = _tensor_info(info_map, node.output[0])
            if _same_shape(input_shape, output_shape):
                replace_tensor[node.output[0]] = node.input[0]
                report["expand_identity_removed"] += 1
                continue
            new_nodes.append(node)
            continue

        if node.op_type == "Add":
            lhs_type, lhs_shape = _tensor_info(info_map, node.input[0])
            rhs_type, rhs_shape = _tensor_info(info_map, node.input[1])
            if (
                _is_float_elem(lhs_type or 0)
                and _is_float_elem(rhs_type or 0)
                and lhs_shape is not None
                and rhs_shape is not None
                and len(lhs_shape) == 4
                and len(rhs_shape) == 4
            ):
                if lhs_shape[:-1] == rhs_shape[:-1]:
                    if rhs_shape[-1] == 1 and isinstance(lhs_shape[-1], int) and lhs_shape[-1] > 1:
                        patched_nodes, initializers = _build_tiled_add(
                            node,
                            broadcast_input=node.input[1],
                            stable_input=node.input[0],
                            last_dim=lhs_shape[-1],
                            broadcast_on_rhs=True,
                        )
                        new_nodes.extend(patched_nodes)
                        new_initializers.extend(initializers)
                        report["add_broadcast_tile_patched"] += 1
                        continue
                    if lhs_shape[-1] == 1 and isinstance(rhs_shape[-1], int) and rhs_shape[-1] > 1:
                        patched_nodes, initializers = _build_tiled_add(
                            node,
                            broadcast_input=node.input[0],
                            stable_input=node.input[1],
                            last_dim=rhs_shape[-1],
                            broadcast_on_rhs=False,
                        )
                        new_nodes.extend(patched_nodes)
                        new_initializers.extend(initializers)
                        report["add_broadcast_tile_patched"] += 1
                        continue
            new_nodes.append(node)
            continue

        if node.op_type != "Mul":
            new_nodes.append(node)
            continue

        report["mul_total"] += 1
        lhs_type, lhs_shape = _tensor_info(info_map, node.input[0])
        rhs_type, rhs_shape = _tensor_info(info_map, node.input[1])

        if lhs_type is None or rhs_type is None or lhs_shape is None or rhs_shape is None:
            report["mul_skipped"] += 1
            report["skipped_nodes"].append({"name": node.name, "reason": "missing_value_info"})
            new_nodes.append(node)
            continue

        if (
            _is_float_elem(lhs_type)
            and _is_float_elem(rhs_type)
            and len(lhs_shape) == 5
            and len(rhs_shape) == 5
        ):
            patched_nodes, initializers = _build_reshape_5d_mul(node, info_map)
            new_nodes.extend(patched_nodes)
            new_initializers.extend(initializers)
            report["mul_5d_patched"] += 1
            continue

        if lhs_type == TensorProto.INT64 and rhs_type == TensorProto.INT64:
            patched_nodes, initializers = _build_int64_cast_mul(node)
            new_nodes.extend(patched_nodes)
            new_initializers.extend(initializers)
            report["mul_int64_patched"] += 1
            continue

        if _is_float_elem(lhs_type) and _is_float_elem(rhs_type):
            if _is_scalar_shape(lhs_shape) and len(rhs_shape) >= 1:
                patched_nodes, initializers = _build_expand_mul(node, node.input[0], node.input[1])
                new_nodes.extend(patched_nodes)
                new_initializers.extend(initializers)
                report["mul_scalar_expand_patched"] += 1
                continue
            if _is_scalar_shape(rhs_shape) and len(lhs_shape) >= 1:
                patched_nodes, initializers = _build_expand_mul(node, node.input[1], node.input[0])
                new_nodes.extend(patched_nodes)
                new_initializers.extend(initializers)
                report["mul_scalar_expand_patched"] += 1
                continue

        report["mul_skipped"] += 1
        report["skipped_nodes"].append(
            {
                "name": node.name,
                "reason": "unsupported_signature",
                "lhs_type": lhs_type,
                "lhs_shape": lhs_shape,
                "rhs_type": rhs_type,
                "rhs_shape": rhs_shape,
            }
        )
        new_nodes.append(node)

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    model.graph.initializer.extend(new_initializers)
    return model, report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(input_path), load_external_data=False)
    if args.shape_inference:
        model = shape_inference.infer_shapes(model)

    model, report = patch_model(model)
    onnx.save(model, str(output_path))

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[patch] output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
