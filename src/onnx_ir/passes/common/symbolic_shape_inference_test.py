# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests for SymbolicShapeInferencePass."""

from __future__ import annotations

import unittest

import onnx_ir as ir
from onnx_ir.passes.common.symbolic_shape_inference import SymbolicShapeInferencePass


class SymbolicShapeInferencePassTest(unittest.TestCase):
    """Tests for SymbolicShapeInferencePass."""

    def test_add_shape_inference(self):
        x = ir.Value(
            name="x", shape=ir.Shape(["batch", 128]), type=ir.TensorType(ir.DataType.FLOAT)
        )
        y = ir.Value(name="y", shape=ir.Shape([1, 128]), type=ir.TensorType(ir.DataType.FLOAT))
        add_out = ir.Value(name="add_out")
        add_node = ir.Node("", "Add", inputs=[x, y], outputs=[add_out])

        graph = ir.Graph(
            inputs=[x, y], outputs=[add_out], nodes=[add_node], opset_imports={"": 17}
        )
        model = ir.Model(graph, ir_version=8)

        result = SymbolicShapeInferencePass()(model)

        self.assertTrue(result.modified)
        self.assertEqual(str(add_out.shape), "[batch,128]")
        self.assertEqual(add_out.dtype, ir.DataType.FLOAT)

    def test_transpose_shape_inference(self):
        x = ir.Value(
            name="x",
            shape=ir.Shape(["batch", "seq", 256]),
            type=ir.TensorType(ir.DataType.FLOAT),
        )
        trans_out = ir.Value(name="trans_out")
        trans_node = ir.Node(
            "",
            "Transpose",
            inputs=[x],
            outputs=[trans_out],
            attributes={"perm": ir.Attr("perm", ir.AttributeType.INTS, [2, 0, 1])},
        )

        graph = ir.Graph(
            inputs=[x], outputs=[trans_out], nodes=[trans_node], opset_imports={"": 17}
        )
        model = ir.Model(graph, ir_version=8)

        result = SymbolicShapeInferencePass()(model)

        self.assertTrue(result.modified)
        self.assertEqual(str(trans_out.shape), "[256,batch,seq]")
        self.assertEqual(trans_out.dtype, ir.DataType.FLOAT)

    def test_transpose_default_perm(self):
        x = ir.Value(
            name="x", shape=ir.Shape([2, 3, 4]), type=ir.TensorType(ir.DataType.FLOAT)
        )
        trans_out = ir.Value(name="trans_out")
        trans_node = ir.Node("", "Transpose", inputs=[x], outputs=[trans_out])

        graph = ir.Graph(
            inputs=[x], outputs=[trans_out], nodes=[trans_node], opset_imports={"": 17}
        )
        model = ir.Model(graph, ir_version=8)

        result = SymbolicShapeInferencePass()(model)

        self.assertTrue(result.modified)
        self.assertEqual(trans_out.shape, [4, 3, 2])

    def test_chained_ops(self):
        x = ir.Value(
            name="x", shape=ir.Shape(["batch", 128]), type=ir.TensorType(ir.DataType.FLOAT)
        )
        y = ir.Value(name="y", shape=ir.Shape([1, 128]), type=ir.TensorType(ir.DataType.FLOAT))
        add_out = ir.Value(name="add_out")
        add_node = ir.Node("", "Add", inputs=[x, y], outputs=[add_out])

        trans_out = ir.Value(name="trans_out")
        trans_node = ir.Node(
            "",
            "Transpose",
            inputs=[add_out],
            outputs=[trans_out],
            attributes={"perm": ir.Attr("perm", ir.AttributeType.INTS, [1, 0])},
        )

        graph = ir.Graph(
            inputs=[x, y],
            outputs=[trans_out],
            nodes=[add_node, trans_node],
            opset_imports={"": 17},
        )
        model = ir.Model(graph, ir_version=8)

        result = SymbolicShapeInferencePass()(model)

        self.assertTrue(result.modified)
        self.assertEqual(str(add_out.shape), "[batch,128]")
        self.assertEqual(str(trans_out.shape), "[128,batch]")


if __name__ == "__main__":
    unittest.main()
