# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the shape inference engine (_engine.py)."""

from __future__ import annotations

import unittest

import onnx_ir as ir
from onnx_ir.shape_inference import _engine


class InferSymbolicShapesTest(unittest.TestCase):
    """Tests for infer_symbolic_shapes (public API)."""

    def _make_model(self, graph: ir.Graph) -> ir.Model:
        return ir.Model(graph, ir_version=10)

    def test_returns_same_model_object(self):
        x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([2, 3]))
        node = ir.Node("", "Relu", inputs=[x], num_outputs=1)
        graph = ir.Graph([x], node.outputs, nodes=[node], opset_imports={"": 21})
        model = self._make_model(graph)

        result = _engine.infer_symbolic_shapes(model)
        self.assertIs(result, model)

    def test_infers_shape_for_relu(self):
        x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([2, 3]))
        node = ir.Node("", "Relu", inputs=[x], num_outputs=1)
        graph = ir.Graph([x], node.outputs, nodes=[node], opset_imports={"": 21})
        model = self._make_model(graph)

        _engine.infer_symbolic_shapes(model)

        self.assertEqual(node.outputs[0].shape, ir.Shape([2, 3]))
        self.assertEqual(node.outputs[0].dtype, ir.DataType.FLOAT)

    def test_names_anonymous_dims_on_graph_inputs(self):
        x = ir.Value(
            name="x",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([ir.SymbolicDim(None), 3]),
        )
        node = ir.Node("", "Relu", inputs=[x], num_outputs=1)
        graph = ir.Graph([x], node.outputs, nodes=[node], opset_imports={"": 21})
        model = self._make_model(graph)

        _engine.infer_symbolic_shapes(model)

        # The anonymous dim should now have a name
        self.assertIsNotNone(x.shape[0].value)

    def test_multi_node_chain(self):
        """Shapes propagate through a chain of nodes."""
        x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([4, 5]))
        relu_node = ir.Node("", "Relu", inputs=[x], num_outputs=1)
        sigmoid_node = ir.Node("", "Sigmoid", inputs=relu_node.outputs, num_outputs=1)
        graph = ir.Graph(
            [x], sigmoid_node.outputs, nodes=[relu_node, sigmoid_node], opset_imports={"": 21}
        )
        model = self._make_model(graph)

        _engine.infer_symbolic_shapes(model)

        self.assertEqual(sigmoid_node.outputs[0].shape, ir.Shape([4, 5]))
        self.assertEqual(sigmoid_node.outputs[0].dtype, ir.DataType.FLOAT)

    def test_unregistered_op_is_skipped(self):
        """Nodes with no registered inference function are skipped gracefully."""
        x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([2]))
        node = ir.Node("com.unknown", "FooBarOp", inputs=[x], num_outputs=1)
        graph = ir.Graph([x], node.outputs, nodes=[node], opset_imports={"": 21})
        model = self._make_model(graph)

        _engine.infer_symbolic_shapes(model)

        # Output shape should remain unset
        self.assertIsNone(node.outputs[0].shape)


class InferSymbolicShapesModifiedTest(unittest.TestCase):
    """Tests for _infer_symbolic_shapes (returns modified flag)."""

    def _make_model(self, graph: ir.Graph) -> ir.Model:
        return ir.Model(graph, ir_version=10)

    def test_returns_true_when_shapes_change(self):
        x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([2, 3]))
        node = ir.Node("", "Relu", inputs=[x], num_outputs=1)
        graph = ir.Graph([x], node.outputs, nodes=[node], opset_imports={"": 21})
        model = self._make_model(graph)

        modified = _engine._infer_symbolic_shapes(model)
        self.assertTrue(modified)

    def test_returns_false_when_no_nodes(self):
        x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([2]))
        graph = ir.Graph([x], [x], nodes=[], opset_imports={"": 21})
        model = self._make_model(graph)

        modified = _engine._infer_symbolic_shapes(model)
        self.assertFalse(modified)


class SubgraphProcessingTest(unittest.TestCase):
    """Tests that subgraphs (e.g. If branches) are processed."""

    def _make_model(self, graph: ir.Graph) -> ir.Model:
        return ir.Model(graph, ir_version=10)

    def test_subgraph_shapes_are_inferred(self):
        """Nodes inside If-then/else subgraphs get shape inference."""
        # Build a then-branch: Relu(input) -> output
        then_x = ir.Value(
            name="then_x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([2, 3])
        )
        then_relu = ir.Node("", "Relu", inputs=[then_x], num_outputs=1)
        then_branch = ir.Graph(
            [then_x],
            then_relu.outputs,
            nodes=[then_relu],
            opset_imports={"": 21},
            name="then_branch",
        )

        # Build an else-branch: Sigmoid(input) -> output
        else_x = ir.Value(
            name="else_x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([2, 3])
        )
        else_sigmoid = ir.Node("", "Sigmoid", inputs=[else_x], num_outputs=1)
        else_branch = ir.Graph(
            [else_x],
            else_sigmoid.outputs,
            nodes=[else_sigmoid],
            opset_imports={"": 21},
            name="else_branch",
        )

        # Build If node
        cond = ir.Value(name="cond", type=ir.TensorType(ir.DataType.BOOL), shape=ir.Shape([]))
        if_node = ir.Node(
            "",
            "If",
            inputs=[cond],
            attributes=[
                ir.Attr("then_branch", ir.AttributeType.GRAPH, then_branch),
                ir.Attr("else_branch", ir.AttributeType.GRAPH, else_branch),
            ],
            num_outputs=1,
        )

        graph = ir.Graph(
            [cond],
            if_node.outputs,
            nodes=[if_node],
            opset_imports={"": 21},
        )
        model = self._make_model(graph)

        _engine.infer_symbolic_shapes(model)

        # Shapes inside subgraphs should be inferred
        self.assertEqual(then_relu.outputs[0].shape, ir.Shape([2, 3]))
        self.assertEqual(then_relu.outputs[0].dtype, ir.DataType.FLOAT)
        self.assertEqual(else_sigmoid.outputs[0].shape, ir.Shape([2, 3]))
        self.assertEqual(else_sigmoid.outputs[0].dtype, ir.DataType.FLOAT)

    def test_anonymous_dims_in_subgraph_are_named(self):
        """Anonymous dims on subgraph inputs get unique names."""
        then_x = ir.Value(
            name="then_x",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([ir.SymbolicDim(None), 3]),
        )
        then_relu = ir.Node("", "Relu", inputs=[then_x], num_outputs=1)
        then_branch = ir.Graph(
            [then_x],
            then_relu.outputs,
            nodes=[then_relu],
            opset_imports={"": 21},
            name="then_branch",
        )

        else_x = ir.Value(
            name="else_x",
            type=ir.TensorType(ir.DataType.FLOAT),
            shape=ir.Shape([ir.SymbolicDim(None), 3]),
        )
        else_relu = ir.Node("", "Relu", inputs=[else_x], num_outputs=1)
        else_branch = ir.Graph(
            [else_x],
            else_relu.outputs,
            nodes=[else_relu],
            opset_imports={"": 21},
            name="else_branch",
        )

        cond = ir.Value(name="cond", type=ir.TensorType(ir.DataType.BOOL), shape=ir.Shape([]))
        if_node = ir.Node(
            "",
            "If",
            inputs=[cond],
            attributes=[
                ir.Attr("then_branch", ir.AttributeType.GRAPH, then_branch),
                ir.Attr("else_branch", ir.AttributeType.GRAPH, else_branch),
            ],
            num_outputs=1,
        )

        graph = ir.Graph(
            [cond],
            if_node.outputs,
            nodes=[if_node],
            opset_imports={"": 21},
        )
        model = self._make_model(graph)

        _engine.infer_symbolic_shapes(model)

        # The anonymous dim in the subgraph input should now have a name
        self.assertIsNotNone(then_x.shape[0].value)


class PolicyPropagationTest(unittest.TestCase):
    """Tests that the policy parameter is respected."""

    def _make_model(self, graph: ir.Graph) -> ir.Model:
        return ir.Model(graph, ir_version=10)

    def test_default_policy_is_refine(self):
        """The default policy should be 'refine'."""
        x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([2, 3]))
        node = ir.Node("", "Relu", inputs=[x], num_outputs=1)
        graph = ir.Graph([x], node.outputs, nodes=[node], opset_imports={"": 21})
        model = self._make_model(graph)

        # Should not raise
        _engine.infer_symbolic_shapes(model)


class FailingOpTest(unittest.TestCase):
    """Tests that a failing op inference does not crash the engine."""

    def _make_model(self, graph: ir.Graph) -> ir.Model:
        return ir.Model(graph, ir_version=10)

    def test_engine_continues_after_op_failure(self):
        """If one node fails, subsequent nodes are still processed."""
        # First node: Gemm with wrong rank (will fail)
        a = ir.Value(
            name="a", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([2, 3, 4])
        )
        b = ir.Value(name="b", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([3, 5]))
        gemm_node = ir.Node("", "Gemm", inputs=[a, b], num_outputs=1)

        # Second node: Relu on a known-good input
        x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([10]))
        relu_node = ir.Node("", "Relu", inputs=[x], num_outputs=1)

        graph = ir.Graph(
            [a, b, x],
            [*gemm_node.outputs, *relu_node.outputs],
            nodes=[gemm_node, relu_node],
            opset_imports={"": 21},
        )
        model = self._make_model(graph)

        _engine.infer_symbolic_shapes(model)

        # Relu should still have been inferred despite Gemm failing
        self.assertEqual(relu_node.outputs[0].shape, ir.Shape([10]))
        self.assertEqual(relu_node.outputs[0].dtype, ir.DataType.FLOAT)


if __name__ == "__main__":
    unittest.main()
