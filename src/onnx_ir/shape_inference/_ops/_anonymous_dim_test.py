# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests that anonymous (None) dims on inputs are named by the engine.

The engine calls ``ctx.name_anonymous_dims()`` on graph inputs and on
every node input before running the inference function.  These tests
build a graph, run ``infer_symbolic_shapes``, and assert that outputs
never contain anonymous ``SymbolicDim(None)``.
"""

from __future__ import annotations

import unittest

import numpy as np

import onnx_ir as ir
from onnx_ir.shape_inference import infer_symbolic_shapes


def _make_model(
    inputs: list[ir.Value],
    outputs: list[ir.Value],
    nodes: list[ir.Node],
    initializers: list[ir.Value] | None = None,
) -> ir.Model:
    """Build a single-graph model with opset 17."""
    graph = ir.Graph(
        inputs=inputs,
        outputs=outputs,
        nodes=nodes,
        initializers=initializers or [],
        opset_imports={"": 17},
        name="test",
    )
    return ir.Model(graph, ir_version=8)


def _input(name: str, dtype: ir.DataType, shape: list[int | str | None]) -> ir.Value:
    """Create an input value with possibly anonymous dims."""
    v = ir.Value(name=name, type=ir.TensorType(dtype))
    v.shape = ir.Shape(shape)
    return v


def _const(name: str, data: list[int] | np.ndarray, dtype: type = np.int64) -> ir.Value:
    """Create a constant-backed value (for Reshape shape, Slice starts, etc.)."""
    arr = np.array(data, dtype=dtype)
    tensor = ir.Tensor(arr, name=name)
    v = ir.Value(name=name, const_value=tensor, type=ir.TensorType(ir.DataType.INT64))
    v.shape = ir.Shape(list(arr.shape))
    return v


def _assert_no_none_dims(test: unittest.TestCase, value: ir.Value) -> None:
    """Assert that *value* has a shape with no anonymous SymbolicDim(None)."""
    test.assertIsNotNone(value.shape, f"{value.name} has no shape")
    assert value.shape is not None  # type narrowing
    for i, dim in enumerate(value.shape.dims):
        if isinstance(dim, ir.SymbolicDim):
            test.assertIsNotNone(
                dim.value,
                f"{value.name} dim {i} is anonymous SymbolicDim(None): {value.shape}",
            )


class TestUnaryAnonymousDims(unittest.TestCase):
    def test_relu(self) -> None:
        x = _input("x", ir.DataType.FLOAT, [None, None])
        y = ir.Value(name="y")
        node = ir.Node("", "Relu", inputs=[x], outputs=[y])
        model = _make_model([x], [y], [node])
        infer_symbolic_shapes(model)
        _assert_no_none_dims(self, y)
        self.assertEqual(y.shape.rank(), 2)

    def test_identity_preserves_concrete(self) -> None:
        x = _input("x", ir.DataType.FLOAT, [None, 10, None])
        y = ir.Value(name="y")
        node = ir.Node("", "Identity", inputs=[x], outputs=[y])
        model = _make_model([x], [y], [node])
        infer_symbolic_shapes(model)
        _assert_no_none_dims(self, y)
        self.assertEqual(y.shape[1], 10)

    def test_not(self) -> None:
        x = _input("x", ir.DataType.BOOL, [None, 5])
        y = ir.Value(name="y")
        node = ir.Node("", "Not", inputs=[x], outputs=[y])
        model = _make_model([x], [y], [node])
        infer_symbolic_shapes(model)
        _assert_no_none_dims(self, y)
        self.assertEqual(y.shape[1], 5)


class TestElementwiseAnonymousDims(unittest.TestCase):
    def test_add_both_none(self) -> None:
        a = _input("a", ir.DataType.FLOAT, [None, 4])
        b = _input("b", ir.DataType.FLOAT, [None, 4])
        y = ir.Value(name="y")
        node = ir.Node("", "Add", inputs=[a, b], outputs=[y])
        model = _make_model([a, b], [y], [node])
        infer_symbolic_shapes(model)
        _assert_no_none_dims(self, y)
        self.assertEqual(y.shape[1], 4)

    def test_add_broadcast_none_dims(self) -> None:
        a = _input("a", ir.DataType.FLOAT, [None, 1])
        b = _input("b", ir.DataType.FLOAT, [1, None])
        y = ir.Value(name="y")
        node = ir.Node("", "Add", inputs=[a, b], outputs=[y])
        model = _make_model([a, b], [y], [node])
        infer_symbolic_shapes(model)
        _assert_no_none_dims(self, y)
        self.assertEqual(y.shape.rank(), 2)

    def test_equal(self) -> None:
        a = _input("a", ir.DataType.INT64, [None, None])
        b = _input("b", ir.DataType.INT64, [None, None])
        y = ir.Value(name="y")
        node = ir.Node("", "Equal", inputs=[a, b], outputs=[y])
        model = _make_model([a, b], [y], [node])
        infer_symbolic_shapes(model)
        _assert_no_none_dims(self, y)


class TestReduceAnonymousDims(unittest.TestCase):
    def test_reduce_sum_keepdims(self) -> None:
        x = _input("x", ir.DataType.FLOAT, [None, None, None])
        y = ir.Value(name="y")
        node = ir.Node(
            "",
            "ReduceSum",
            inputs=[x],
            outputs=[y],
            attributes={
                "axes": ir.Attr("axes", ir.AttributeType.INTS, [1]),
                "keepdims": ir.Attr("keepdims", ir.AttributeType.INT, 1),
            },
        )
        model = _make_model([x], [y], [node])
        infer_symbolic_shapes(model)
        _assert_no_none_dims(self, y)
        self.assertEqual(y.shape.rank(), 3)
        self.assertEqual(y.shape[1], 1)

    def test_reduce_max_no_keepdims(self) -> None:
        x = _input("x", ir.DataType.FLOAT, [None, 8, None])
        y = ir.Value(name="y")
        node = ir.Node(
            "",
            "ReduceMax",
            inputs=[x],
            outputs=[y],
            attributes={
                "axes": ir.Attr("axes", ir.AttributeType.INTS, [1]),
                "keepdims": ir.Attr("keepdims", ir.AttributeType.INT, 0),
            },
        )
        model = _make_model([x], [y], [node])
        infer_symbolic_shapes(model)
        _assert_no_none_dims(self, y)
        self.assertEqual(y.shape.rank(), 2)


class TestConcatAnonymousDims(unittest.TestCase):
    def test_concat(self) -> None:
        a = _input("a", ir.DataType.FLOAT, [None, 3])
        b = _input("b", ir.DataType.FLOAT, [None, 5])
        y = ir.Value(name="y")
        node = ir.Node(
            "",
            "Concat",
            inputs=[a, b],
            outputs=[y],
            attributes={"axis": ir.Attr("axis", ir.AttributeType.INT, 1)},
        )
        model = _make_model([a, b], [y], [node])
        infer_symbolic_shapes(model)
        _assert_no_none_dims(self, y)
        self.assertEqual(y.shape[1], 8)


class TestSoftmaxAnonymousDims(unittest.TestCase):
    def test_softmax(self) -> None:
        x = _input("x", ir.DataType.FLOAT, [None, None])
        y = ir.Value(name="y")
        node = ir.Node("", "Softmax", inputs=[x], outputs=[y])
        model = _make_model([x], [y], [node])
        infer_symbolic_shapes(model)
        _assert_no_none_dims(self, y)


class TestDropoutAnonymousDims(unittest.TestCase):
    def test_dropout(self) -> None:
        x = _input("x", ir.DataType.FLOAT, [None, None])
        y = ir.Value(name="y")
        mask = ir.Value(name="mask")
        node = ir.Node("", "Dropout", inputs=[x], outputs=[y, mask])
        model = _make_model([x], [y, mask], [node])
        infer_symbolic_shapes(model)
        _assert_no_none_dims(self, y)
        _assert_no_none_dims(self, mask)


class TestWhereAnonymousDims(unittest.TestCase):
    def test_where(self) -> None:
        cond = _input("cond", ir.DataType.BOOL, [None, 1])
        a = _input("a", ir.DataType.FLOAT, [1, None])
        b = _input("b", ir.DataType.FLOAT, [None, None])
        y = ir.Value(name="y")
        node = ir.Node("", "Where", inputs=[cond, a, b], outputs=[y])
        model = _make_model([cond, a, b], [y], [node])
        infer_symbolic_shapes(model)
        _assert_no_none_dims(self, y)


class TestMatMulAnonymousDims(unittest.TestCase):
    def test_matmul(self) -> None:
        a = _input("a", ir.DataType.FLOAT, [None, 4])
        b = _input("b", ir.DataType.FLOAT, [4, None])
        y = ir.Value(name="y")
        node = ir.Node("", "MatMul", inputs=[a, b], outputs=[y])
        model = _make_model([a, b], [y], [node])
        infer_symbolic_shapes(model)
        _assert_no_none_dims(self, y)
        self.assertEqual(y.shape.rank(), 2)

    def test_matmul_batch(self) -> None:
        a = _input("a", ir.DataType.FLOAT, [None, 3, 4])
        b = _input("b", ir.DataType.FLOAT, [None, 4, 5])
        y = ir.Value(name="y")
        node = ir.Node("", "MatMul", inputs=[a, b], outputs=[y])
        model = _make_model([a, b], [y], [node])
        infer_symbolic_shapes(model)
        _assert_no_none_dims(self, y)
        self.assertEqual(y.shape.rank(), 3)


class TestGatherAnonymousDims(unittest.TestCase):
    def test_gather(self) -> None:
        data = _input("data", ir.DataType.FLOAT, [None, 8])
        indices = _input("indices", ir.DataType.INT64, [None])
        y = ir.Value(name="y")
        node = ir.Node(
            "",
            "Gather",
            inputs=[data, indices],
            outputs=[y],
            attributes={"axis": ir.Attr("axis", ir.AttributeType.INT, 0)},
        )
        model = _make_model([data, indices], [y], [node])
        infer_symbolic_shapes(model)
        _assert_no_none_dims(self, y)
        self.assertEqual(y.shape[1], 8)


class TestReshapeAnonymousDims(unittest.TestCase):
    def test_reshape(self) -> None:
        x = _input("x", ir.DataType.FLOAT, [None, 12])
        shape = _const("shape", [0, 3, 4])
        y = ir.Value(name="y")
        node = ir.Node("", "Reshape", inputs=[x, shape], outputs=[y])
        model = _make_model([x], [y], [node], initializers=[shape])
        infer_symbolic_shapes(model)
        _assert_no_none_dims(self, y)
        self.assertEqual(y.shape.rank(), 3)


class TestExpandAnonymousDims(unittest.TestCase):
    def test_expand(self) -> None:
        x = _input("x", ir.DataType.FLOAT, [None, 1])
        target = _const("target", [1, 4])
        y = ir.Value(name="y")
        node = ir.Node("", "Expand", inputs=[x, target], outputs=[y])
        model = _make_model([x], [y], [node], initializers=[target])
        infer_symbolic_shapes(model)
        _assert_no_none_dims(self, y)
        self.assertEqual(y.shape[1], 4)


class TestConvAnonymousDims(unittest.TestCase):
    def test_conv_none_batch(self) -> None:
        x = _input("x", ir.DataType.FLOAT, [None, 3, 8, 8])
        w = _input("w", ir.DataType.FLOAT, [16, 3, 3, 3])
        y = ir.Value(name="y")
        node = ir.Node(
            "",
            "Conv",
            inputs=[x, w],
            outputs=[y],
            attributes={
                "kernel_shape": ir.Attr("kernel_shape", ir.AttributeType.INTS, [3, 3]),
            },
        )
        model = _make_model([x, w], [y], [node])
        infer_symbolic_shapes(model)
        _assert_no_none_dims(self, y)
        self.assertEqual(y.shape.rank(), 4)
        self.assertEqual(y.shape[1], 16)


class TestGemmAnonymousDims(unittest.TestCase):
    def test_gemm(self) -> None:
        a = _input("a", ir.DataType.FLOAT, [None, 4])
        b = _input("b", ir.DataType.FLOAT, [4, None])
        y = ir.Value(name="y")
        node = ir.Node("", "Gemm", inputs=[a, b], outputs=[y])
        model = _make_model([a, b], [y], [node])
        infer_symbolic_shapes(model)
        _assert_no_none_dims(self, y)
        self.assertEqual(y.shape.rank(), 2)


class TestTransposeAnonymousDims(unittest.TestCase):
    def test_transpose(self) -> None:
        x = _input("x", ir.DataType.FLOAT, [None, 3, None])
        y = ir.Value(name="y")
        node = ir.Node(
            "",
            "Transpose",
            inputs=[x],
            outputs=[y],
            attributes={"perm": ir.Attr("perm", ir.AttributeType.INTS, [2, 1, 0])},
        )
        model = _make_model([x], [y], [node])
        infer_symbolic_shapes(model)
        _assert_no_none_dims(self, y)
        self.assertEqual(y.shape[1], 3)


class TestSqueezeAnonymousDims(unittest.TestCase):
    def test_squeeze(self) -> None:
        x = _input("x", ir.DataType.FLOAT, [None, 1, None])
        y = ir.Value(name="y")
        node = ir.Node(
            "",
            "Squeeze",
            inputs=[x],
            outputs=[y],
            attributes={"axes": ir.Attr("axes", ir.AttributeType.INTS, [1])},
        )
        model = _make_model([x], [y], [node])
        infer_symbolic_shapes(model)
        _assert_no_none_dims(self, y)
        self.assertEqual(y.shape.rank(), 2)

    def test_unsqueeze(self) -> None:
        x = _input("x", ir.DataType.FLOAT, [None, None])
        y = ir.Value(name="y")
        node = ir.Node(
            "",
            "Unsqueeze",
            inputs=[x],
            outputs=[y],
            attributes={"axes": ir.Attr("axes", ir.AttributeType.INTS, [0])},
        )
        model = _make_model([x], [y], [node])
        infer_symbolic_shapes(model)
        _assert_no_none_dims(self, y)
        self.assertEqual(y.shape[0], 1)


class TestSplitAnonymousDims(unittest.TestCase):
    def test_split(self) -> None:
        x = _input("x", ir.DataType.FLOAT, [None, 6])
        y0 = ir.Value(name="y0")
        y1 = ir.Value(name="y1")
        node = ir.Node(
            "",
            "Split",
            inputs=[x],
            outputs=[y0, y1],
            attributes={
                "axis": ir.Attr("axis", ir.AttributeType.INT, 1),
                "num_outputs": ir.Attr("num_outputs", ir.AttributeType.INT, 2),
            },
        )
        model = _make_model([x], [y0, y1], [node])
        infer_symbolic_shapes(model)
        _assert_no_none_dims(self, y0)
        _assert_no_none_dims(self, y1)
        self.assertEqual(y0.shape[1], 3)


class TestFlattenAnonymousDims(unittest.TestCase):
    def test_flatten(self) -> None:
        x = _input("x", ir.DataType.FLOAT, [None, 3, None])
        y = ir.Value(name="y")
        node = ir.Node(
            "",
            "Flatten",
            inputs=[x],
            outputs=[y],
            attributes={"axis": ir.Attr("axis", ir.AttributeType.INT, 1)},
        )
        model = _make_model([x], [y], [node])
        infer_symbolic_shapes(model)
        _assert_no_none_dims(self, y)
        self.assertEqual(y.shape.rank(), 2)


class TestChainedOpsAnonymousDims(unittest.TestCase):
    """Test that anonymous dims are named through a chain of ops."""

    def test_relu_then_add(self) -> None:
        x = _input("x", ir.DataType.FLOAT, [None, 4])
        b = _input("b", ir.DataType.FLOAT, [None, 4])
        relu_out = ir.Value(name="relu_out")
        y = ir.Value(name="y")
        relu_node = ir.Node("", "Relu", inputs=[x], outputs=[relu_out])
        add_node = ir.Node("", "Add", inputs=[relu_out, b], outputs=[y])
        model = _make_model([x, b], [y], [relu_node, add_node])
        infer_symbolic_shapes(model)
        _assert_no_none_dims(self, relu_out)
        _assert_no_none_dims(self, y)

    def test_graph_input_dims_are_named(self) -> None:
        """Graph inputs with None dims should be named after inference."""
        x = _input("x", ir.DataType.FLOAT, [None, None])
        y = ir.Value(name="y")
        node = ir.Node("", "Relu", inputs=[x], outputs=[y])
        model = _make_model([x], [y], [node])
        infer_symbolic_shapes(model)

        # Graph input dims should now be named
        self.assertIsNotNone(x.shape)
        for dim in x.shape.dims:
            self.assertIsInstance(dim, ir.SymbolicDim)
            self.assertIsNotNone(dim.value, f"Dim still anonymous: {dim}")


if __name__ == "__main__":
    unittest.main()
