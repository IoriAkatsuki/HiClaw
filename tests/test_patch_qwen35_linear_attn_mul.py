import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

import patch_qwen35_linear_attn_mul as patcher


class PatchQwen35LinearAttnMulTest(unittest.TestCase):
    def test_patch_5d_float_mul_inserts_reshape_before_and_after(self):
        model = _make_5d_float_mul_model()

        patched, report = patcher.patch_model(model)

        node_types = [node.op_type for node in patched.graph.node]
        self.assertEqual(report["mul_5d_patched"], 1)
        self.assertIn("Reshape", node_types)
        self.assertIn("Mul", node_types)
        self.assertEqual(patched.graph.output[0].name, "mul_out")
        reshape_nodes = [node for node in patched.graph.node if node.op_type == "Reshape"]
        self.assertEqual(len(reshape_nodes), 3)

    def test_patch_int64_mul_wraps_casts_and_preserves_output_name(self):
        model = _make_int64_mul_model()

        patched, report = patcher.patch_model(model)

        node_types = [node.op_type for node in patched.graph.node]
        self.assertEqual(report["mul_int64_patched"], 1)
        self.assertEqual(node_types.count("Cast"), 3)
        self.assertIn("Mul", node_types)
        self.assertEqual(patched.graph.output[0].name, "shape_mul_out")

    def test_patch_scalar_broadcast_mul_inserts_expand(self):
        model = _make_scalar_broadcast_mul_model()

        patched, report = patcher.patch_model(model)

        node_types = [node.op_type for node in patched.graph.node]
        self.assertEqual(report["mul_scalar_expand_patched"], 1)
        self.assertIn("Expand", node_types)
        self.assertEqual(patched.graph.output[0].name, "scaled")

    def test_cli_writes_model_and_report(self):
        model = _make_int64_mul_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "in.onnx"
            output_path = Path(tmpdir) / "out.onnx"
            onnx.save(model, input_path)

            rc = patcher.main(
                [
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                ]
            )

            self.assertEqual(rc, 0)
            self.assertTrue(output_path.exists())

    def test_patch_redundant_expand_rewires_consumers(self):
        model = _make_redundant_expand_model()

        patched, report = patcher.patch_model(model)

        self.assertEqual(report["expand_identity_removed"], 1)
        node_names = [node.name for node in patched.graph.node]
        self.assertNotIn("/m/model/layers.0/linear_attn/Expand_96", node_names)
        add_node = next(node for node in patched.graph.node if node.name == "consumer_add")
        self.assertEqual(add_node.input[0], "expand_src")

    def test_patch_broadcast_add_inserts_tile_before_add(self):
        model = _make_broadcast_add_model()

        patched, report = patcher.patch_model(model)

        self.assertEqual(report["add_broadcast_tile_patched"], 1)
        node_types = [node.op_type for node in patched.graph.node]
        self.assertIn("Tile", node_types)
        add_node = next(node for node in patched.graph.node if node.name == "/m/model/layers.0/linear_attn/Add_83")
        self.assertTrue(add_node.input[1].endswith("_tiled"))


def _make_5d_float_mul_model() -> onnx.ModelProto:
    a = helper.make_tensor_value_info("a", TensorProto.FLOAT, ["B", "H", "S", 64, 64])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, ["B", "H", "S", 64, 64])
    out = helper.make_tensor_value_info("mul_out", TensorProto.FLOAT, ["B", "H", "S", 64, 64])

    node = helper.make_node(
        "Mul",
        ["a", "b"],
        ["mul_out"],
        name="/m/model/layers.0/linear_attn/Mul_34",
    )
    graph = helper.make_graph([node], "five_d_mul", [a, b], [out])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _make_int64_mul_model() -> onnx.ModelProto:
    a = helper.make_tensor_value_info("shape_vec", TensorProto.INT64, [5])
    scalar = helper.make_tensor_value_info("shape_scalar", TensorProto.INT64, [])
    out = helper.make_tensor_value_info("shape_mul_out", TensorProto.INT64, [5])

    node = helper.make_node(
        "Mul",
        ["shape_vec", "shape_scalar"],
        ["shape_mul_out"],
        name="/m/model/layers.0/linear_attn/Mul_11",
    )
    graph = helper.make_graph([node], "int64_mul", [a, scalar], [out])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _make_scalar_broadcast_mul_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["B", "T", "C", "D"])
    scale = helper.make_tensor(
        name="scale",
        data_type=TensorProto.FLOAT,
        dims=[],
        vals=np.array(0.5, dtype=np.float32).reshape(()),
    )
    out = helper.make_tensor_value_info("scaled", TensorProto.FLOAT, ["B", "T", "C", "D"])

    node = helper.make_node(
        "Mul",
        ["x", "scale"],
        ["scaled"],
        name="/m/model/layers.0/linear_attn/Mul_6",
    )
    graph = helper.make_graph([node], "scalar_expand_mul", [x], [out], initializer=[scale])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _make_redundant_expand_model() -> onnx.ModelProto:
    expand_src = helper.make_tensor_value_info("expand_src", TensorProto.FLOAT, ["B", 16, "S", 17])
    shape_src = helper.make_tensor_value_info("shape_src", TensorProto.FLOAT, ["B", 16, "S", 17])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, ["B", 16, "S", 17])
    expand_out = helper.make_tensor_value_info(
        "/m/model/layers.0/linear_attn/Expand_96_output_0", TensorProto.FLOAT, ["B", 16, "S", 17]
    )
    shape_out = helper.make_tensor_value_info(
        "/m/model/layers.0/linear_attn/Shape_148_output_0", TensorProto.INT64, [4]
    )
    add_out = helper.make_tensor_value_info("consumer_out", TensorProto.FLOAT, ["B", 16, "S", 17])

    shape_node = helper.make_node(
        "Shape",
        ["shape_src"],
        ["/m/model/layers.0/linear_attn/Shape_148_output_0"],
        name="/m/model/layers.0/linear_attn/Shape_148",
    )
    expand_node = helper.make_node(
        "Expand",
        ["expand_src", "/m/model/layers.0/linear_attn/Shape_148_output_0"],
        ["/m/model/layers.0/linear_attn/Expand_96_output_0"],
        name="/m/model/layers.0/linear_attn/Expand_96",
    )
    consumer = helper.make_node("Add", ["expand_src", "/m/model/layers.0/linear_attn/Expand_96_output_0"], ["out"], name="consumer_add")
    graph = helper.make_graph(
        [shape_node, expand_node, consumer],
        "redundant_expand",
        [expand_src, shape_src],
        [out],
        value_info=[expand_out, shape_out, add_out],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _make_broadcast_add_model() -> onnx.ModelProto:
    lhs = helper.make_tensor_value_info("lhs", TensorProto.FLOAT, ["B", 16, "S", 17])
    rhs = helper.make_tensor_value_info("rhs", TensorProto.FLOAT, ["B", 16, "S", 1])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, ["B", 16, "S", 17])
    node = helper.make_node(
        "Add",
        ["lhs", "rhs"],
        ["out"],
        name="/m/model/layers.0/linear_attn/Add_83",
    )
    graph = helper.make_graph([node], "broadcast_add", [lhs, rhs], [out])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


if __name__ == "__main__":
    unittest.main()
