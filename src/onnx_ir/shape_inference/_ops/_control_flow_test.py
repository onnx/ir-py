# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests for control flow shape inference (If, Loop)."""

from __future__ import annotations

import unittest

import onnx_ir as ir
from onnx_ir.shape_inference import (
    OpUsageError,
    _context,
    infer_symbolic_shapes,
)
from onnx_ir.shape_inference._ops._control_flow import _merge_shapes
from onnx_ir.shape_inference._ops._testing import (
    run_shape_inference_with_values,
)


def _make_model(graph: ir.Graph) -> ir.Model:
    return ir.Model(graph, ir_version=10)


def _make_ctx() -> _context.ShapeInferenceContext:
    return _context.ShapeInferenceContext(policy="override")


# ---------------------------------------------------------------------------
# _merge_shapes helper
# ---------------------------------------------------------------------------
class MergeShapesTest(unittest.TestCase):
    def test_same_shapes(self):
        s = _merge_shapes(_make_ctx(), ir.Shape([2, 3]), ir.Shape([2, 3]))
        self.assertEqual(s, ir.Shape([2, 3]))

    def test_different_dims_become_symbolic(self):
        s = _merge_shapes(_make_ctx(), ir.Shape([2, 3]), ir.Shape([2, 5]))
        self.assertIsNotNone(s)
        self.assertEqual(s.rank(), 2)
        self.assertEqual(s[0], 2)
        # Dimension 1 differs → unknown
        self.assertIsInstance(s[1], ir.SymbolicDim)

    def test_different_ranks_return_none(self):
        s = _merge_shapes(_make_ctx(), ir.Shape([2, 3]), ir.Shape([2, 3, 4]))
        self.assertIsNone(s)

    def test_none_input_returns_none(self):
        ctx = _make_ctx()
        self.assertIsNone(_merge_shapes(ctx, None, ir.Shape([2])))
        self.assertIsNone(_merge_shapes(ctx, ir.Shape([2]), None))

    def test_symbolic_same_name(self):
        s = _merge_shapes(_make_ctx(), ir.Shape(["N", 3]), ir.Shape(["N", 3]))
        self.assertEqual(s, ir.Shape(["N", 3]))

    def test_symbolic_different_names(self):
        s = _merge_shapes(_make_ctx(), ir.Shape(["N", 3]), ir.Shape(["M", 3]))
        self.assertIsNotNone(s)
        self.assertIsInstance(s[0], ir.SymbolicDim)
        self.assertEqual(s[1], 3)


# ---------------------------------------------------------------------------
# If operator
# ---------------------------------------------------------------------------
class IfShapeInferenceTest(unittest.TestCase):
    def _run_if(
        self,
        then_out_shape: ir.Shape | None,
        else_out_shape: ir.Shape | None,
        then_dtype: ir.DataType = ir.DataType.FLOAT,
        else_dtype: ir.DataType = ir.DataType.FLOAT,
    ) -> ir.Value:
        """Build an If node with the given branch output shapes and run inference."""
        # Then branch
        then_out = ir.Value(
            name="then_y",
            type=ir.TensorType(then_dtype) if then_dtype else None,
            shape=then_out_shape,
        )
        then_graph = ir.Graph([], [then_out], nodes=[], opset_imports={"": 21}, name="then")

        # Else branch
        else_out = ir.Value(
            name="else_y",
            type=ir.TensorType(else_dtype) if else_dtype else None,
            shape=else_out_shape,
        )
        else_graph = ir.Graph([], [else_out], nodes=[], opset_imports={"": 21}, name="else")

        cond = ir.Value(name="cond", type=ir.TensorType(ir.DataType.BOOL), shape=ir.Shape([]))

        if_node = ir.Node(
            "",
            "If",
            inputs=[cond],
            attributes=[
                ir.Attr("then_branch", ir.AttributeType.GRAPH, then_graph),
                ir.Attr("else_branch", ir.AttributeType.GRAPH, else_graph),
            ],
            num_outputs=1,
        )

        graph = ir.Graph([cond], if_node.outputs, nodes=[if_node], opset_imports={"": 21})
        model = _make_model(graph)
        infer_symbolic_shapes(model)
        return if_node.outputs[0]

    def test_same_shapes(self):
        out = self._run_if(ir.Shape([2, 3]), ir.Shape([2, 3]))
        self.assertEqual(out.shape, ir.Shape([2, 3]))
        self.assertEqual(out.dtype, ir.DataType.FLOAT)

    def test_different_dims_merged(self):
        out = self._run_if(ir.Shape([2, 3]), ir.Shape([2, 5]))
        self.assertIsNotNone(out.shape)
        self.assertEqual(out.shape.rank(), 2)
        self.assertEqual(out.shape[0], 2)
        self.assertIsInstance(out.shape[1], ir.SymbolicDim)

    def test_different_ranks_gives_none_shape(self):
        out = self._run_if(ir.Shape([2, 3]), ir.Shape([2, 3, 4]))
        self.assertIsNone(out.shape)
        self.assertEqual(out.dtype, ir.DataType.FLOAT)

    def test_one_branch_none_shape(self):
        out = self._run_if(ir.Shape([2, 3]), None)
        self.assertEqual(out.shape, ir.Shape([2, 3]))

    def test_both_branches_none_shape(self):
        out = self._run_if(None, None)
        self.assertIsNone(out.shape)
        self.assertEqual(out.dtype, ir.DataType.FLOAT)

    def test_symbolic_dims(self):
        out = self._run_if(ir.Shape(["N", 3]), ir.Shape(["N", 3]))
        self.assertEqual(out.shape, ir.Shape(["N", 3]))

    def test_none_cond_raises(self):
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "If",
                [None],
                attributes=[
                    ir.Attr(
                        "then_branch",
                        ir.AttributeType.GRAPH,
                        ir.Graph([], [], nodes=[], name="then"),
                    ),
                    ir.Attr(
                        "else_branch",
                        ir.AttributeType.GRAPH,
                        ir.Graph([], [], nodes=[], name="else"),
                    ),
                ],
                opset_version=21,
            )

    def test_missing_then_branch_raises(self):
        """Missing then_branch attribute raises OpUsageError."""
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "If",
                [ir.Value(name="cond", type=ir.TensorType(ir.DataType.BOOL))],
                attributes=[
                    ir.Attr(
                        "else_branch",
                        ir.AttributeType.GRAPH,
                        ir.Graph([], [], nodes=[], name="else"),
                    ),
                ],
                opset_version=21,
            )

    def test_missing_else_branch_raises(self):
        """Missing else_branch attribute raises OpUsageError."""
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "If",
                [ir.Value(name="cond", type=ir.TensorType(ir.DataType.BOOL))],
                attributes=[
                    ir.Attr(
                        "then_branch",
                        ir.AttributeType.GRAPH,
                        ir.Graph([], [], nodes=[], name="then"),
                    ),
                ],
                opset_version=21,
            )

    def test_subgraph_shapes_propagated_via_engine(self):
        """End-to-end: ops inside subgraphs get inference via engine recursion."""
        # Then branch with a Relu node
        then_x = ir.Value(
            name="then_x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([4, 5])
        )
        then_relu = ir.Node("", "Relu", inputs=[then_x], num_outputs=1)
        then_graph = ir.Graph(
            [then_x],
            then_relu.outputs,
            nodes=[then_relu],
            opset_imports={"": 21},
            name="then",
        )

        # Else branch with a Sigmoid node
        else_x = ir.Value(
            name="else_x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([4, 5])
        )
        else_sigmoid = ir.Node("", "Sigmoid", inputs=[else_x], num_outputs=1)
        else_graph = ir.Graph(
            [else_x],
            else_sigmoid.outputs,
            nodes=[else_sigmoid],
            opset_imports={"": 21},
            name="else",
        )

        cond = ir.Value(name="cond", type=ir.TensorType(ir.DataType.BOOL), shape=ir.Shape([]))
        if_node = ir.Node(
            "",
            "If",
            inputs=[cond],
            attributes=[
                ir.Attr("then_branch", ir.AttributeType.GRAPH, then_graph),
                ir.Attr("else_branch", ir.AttributeType.GRAPH, else_graph),
            ],
            num_outputs=1,
        )

        graph = ir.Graph([cond], if_node.outputs, nodes=[if_node], opset_imports={"": 21})
        model = _make_model(graph)
        infer_symbolic_shapes(model)

        # The If output should have the merged shape
        self.assertEqual(if_node.outputs[0].shape, ir.Shape([4, 5]))
        self.assertEqual(if_node.outputs[0].dtype, ir.DataType.FLOAT)

    def test_multiple_outputs(self):
        """If with two outputs from each branch."""
        then_y0 = ir.Value(
            name="then_y0", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([2, 3])
        )
        then_y1 = ir.Value(
            name="then_y1", type=ir.TensorType(ir.DataType.INT64), shape=ir.Shape([5])
        )
        then_graph = ir.Graph(
            [], [then_y0, then_y1], nodes=[], opset_imports={"": 21}, name="then"
        )

        else_y0 = ir.Value(
            name="else_y0", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([2, 3])
        )
        else_y1 = ir.Value(
            name="else_y1", type=ir.TensorType(ir.DataType.INT64), shape=ir.Shape([5])
        )
        else_graph = ir.Graph(
            [], [else_y0, else_y1], nodes=[], opset_imports={"": 21}, name="else"
        )

        cond = ir.Value(name="cond", type=ir.TensorType(ir.DataType.BOOL), shape=ir.Shape([]))
        if_node = ir.Node(
            "",
            "If",
            inputs=[cond],
            attributes=[
                ir.Attr("then_branch", ir.AttributeType.GRAPH, then_graph),
                ir.Attr("else_branch", ir.AttributeType.GRAPH, else_graph),
            ],
            num_outputs=2,
        )
        graph = ir.Graph(
            [cond], list(if_node.outputs), nodes=[if_node], opset_imports={"": 21}
        )
        model = _make_model(graph)
        infer_symbolic_shapes(model)

        self.assertEqual(if_node.outputs[0].shape, ir.Shape([2, 3]))
        self.assertEqual(if_node.outputs[0].dtype, ir.DataType.FLOAT)
        self.assertEqual(if_node.outputs[1].shape, ir.Shape([5]))
        self.assertEqual(if_node.outputs[1].dtype, ir.DataType.INT64)


# ---------------------------------------------------------------------------
# Loop operator
# ---------------------------------------------------------------------------
class LoopShapeInferenceTest(unittest.TestCase):
    def test_loop_carried_dependency_shape(self):
        """Loop-carried outputs keep the body output shape."""
        # Body: inputs=[iter_num, cond, x], outputs=[cond_out, y]
        iter_num = ir.Value(
            name="i", type=ir.TensorType(ir.DataType.INT64), shape=ir.Shape([])
        )
        body_cond = ir.Value(
            name="cond_in", type=ir.TensorType(ir.DataType.BOOL), shape=ir.Shape([])
        )
        body_x = ir.Value(
            name="body_x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([2, 3])
        )
        body_relu = ir.Node("", "Relu", inputs=[body_x], num_outputs=1)
        body_cond_out = ir.Value(
            name="cond_out", type=ir.TensorType(ir.DataType.BOOL), shape=ir.Shape([])
        )
        body_graph = ir.Graph(
            [iter_num, body_cond, body_x],
            [body_cond_out, body_relu.outputs[0]],
            nodes=[body_relu],
            opset_imports={"": 21},
            name="body",
        )

        # Loop node: inputs=[max_trip_count, cond, x_init]
        max_trip = ir.Value(
            name="max", type=ir.TensorType(ir.DataType.INT64), shape=ir.Shape([])
        )
        cond = ir.Value(name="cond", type=ir.TensorType(ir.DataType.BOOL), shape=ir.Shape([]))
        x_init = ir.Value(
            name="x_init", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([2, 3])
        )
        loop_node = ir.Node(
            "",
            "Loop",
            inputs=[max_trip, cond, x_init],
            attributes=[ir.Attr("body", ir.AttributeType.GRAPH, body_graph)],
            num_outputs=1,  # 1 loop-carried output, 0 scan outputs
        )

        graph = ir.Graph(
            [max_trip, cond, x_init],
            loop_node.outputs,
            nodes=[loop_node],
            opset_imports={"": 21},
        )
        model = _make_model(graph)
        infer_symbolic_shapes(model)

        self.assertEqual(loop_node.outputs[0].shape, ir.Shape([2, 3]))
        self.assertEqual(loop_node.outputs[0].dtype, ir.DataType.FLOAT)

    def test_scan_output_has_trip_dim_prepended(self):
        """Scan outputs get a trip-count dim prepended."""
        # Body: inputs=[iter_num, cond], outputs=[cond_out, scan_val]
        iter_num = ir.Value(
            name="i", type=ir.TensorType(ir.DataType.INT64), shape=ir.Shape([])
        )
        body_cond = ir.Value(
            name="cond_in", type=ir.TensorType(ir.DataType.BOOL), shape=ir.Shape([])
        )
        scan_val = ir.Value(
            name="scan_val", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([4, 5])
        )
        body_cond_out = ir.Value(
            name="cond_out", type=ir.TensorType(ir.DataType.BOOL), shape=ir.Shape([])
        )
        body_graph = ir.Graph(
            [iter_num, body_cond],
            [body_cond_out, scan_val],
            nodes=[],
            opset_imports={"": 21},
            name="body",
        )

        # Loop node: inputs=[max_trip, cond] (no loop-carried deps)
        max_trip = ir.Value(
            name="max", type=ir.TensorType(ir.DataType.INT64), shape=ir.Shape([])
        )
        cond = ir.Value(name="cond", type=ir.TensorType(ir.DataType.BOOL), shape=ir.Shape([]))
        loop_node = ir.Node(
            "",
            "Loop",
            inputs=[max_trip, cond],
            attributes=[ir.Attr("body", ir.AttributeType.GRAPH, body_graph)],
            num_outputs=1,  # 0 loop-carried, 1 scan output
        )

        graph = ir.Graph(
            [max_trip, cond],
            loop_node.outputs,
            nodes=[loop_node],
            opset_imports={"": 21},
        )
        model = _make_model(graph)
        infer_symbolic_shapes(model)

        out = loop_node.outputs[0]
        self.assertIsNotNone(out.shape)
        self.assertEqual(out.shape.rank(), 3)
        # First dim is the trip count (symbolic/unknown)
        self.assertIsInstance(out.shape[0], ir.SymbolicDim)
        self.assertEqual(out.shape[1], 4)
        self.assertEqual(out.shape[2], 5)
        self.assertEqual(out.dtype, ir.DataType.FLOAT)

    def test_loop_carried_and_scan_outputs(self):
        """Loop with both loop-carried and scan outputs."""
        iter_num = ir.Value(
            name="i", type=ir.TensorType(ir.DataType.INT64), shape=ir.Shape([])
        )
        body_cond = ir.Value(
            name="cond_in", type=ir.TensorType(ir.DataType.BOOL), shape=ir.Shape([])
        )
        body_x = ir.Value(
            name="body_x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([3])
        )
        body_cond_out = ir.Value(
            name="cond_out", type=ir.TensorType(ir.DataType.BOOL), shape=ir.Shape([])
        )
        body_x_out = ir.Value(
            name="body_x_out", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([3])
        )
        scan_out = ir.Value(
            name="scan_out", type=ir.TensorType(ir.DataType.INT64), shape=ir.Shape([7])
        )
        body_graph = ir.Graph(
            [iter_num, body_cond, body_x],
            [body_cond_out, body_x_out, scan_out],
            nodes=[],
            opset_imports={"": 21},
            name="body",
        )

        max_trip = ir.Value(
            name="max", type=ir.TensorType(ir.DataType.INT64), shape=ir.Shape([])
        )
        cond = ir.Value(name="cond", type=ir.TensorType(ir.DataType.BOOL), shape=ir.Shape([]))
        x_init = ir.Value(
            name="x_init", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([3])
        )
        loop_node = ir.Node(
            "",
            "Loop",
            inputs=[max_trip, cond, x_init],
            attributes=[ir.Attr("body", ir.AttributeType.GRAPH, body_graph)],
            num_outputs=2,  # 1 loop-carried, 1 scan
        )

        graph = ir.Graph(
            [max_trip, cond, x_init],
            list(loop_node.outputs),
            nodes=[loop_node],
            opset_imports={"": 21},
        )
        model = _make_model(graph)
        infer_symbolic_shapes(model)

        # Output 0: loop-carried, shape [3]
        self.assertEqual(loop_node.outputs[0].shape, ir.Shape([3]))
        self.assertEqual(loop_node.outputs[0].dtype, ir.DataType.FLOAT)

        # Output 1: scan, shape [trip, 7]
        self.assertIsNotNone(loop_node.outputs[1].shape)
        self.assertEqual(loop_node.outputs[1].shape.rank(), 2)
        self.assertIsInstance(loop_node.outputs[1].shape[0], ir.SymbolicDim)
        self.assertEqual(loop_node.outputs[1].shape[1], 7)
        self.assertEqual(loop_node.outputs[1].dtype, ir.DataType.INT64)

    def test_no_body_attr_raises(self):
        """Missing body attribute raises OpUsageError (caught by engine)."""
        max_trip = ir.Value(
            name="max", type=ir.TensorType(ir.DataType.INT64), shape=ir.Shape([])
        )
        cond = ir.Value(name="cond", type=ir.TensorType(ir.DataType.BOOL), shape=ir.Shape([]))
        loop_node = ir.Node(
            "",
            "Loop",
            inputs=[max_trip, cond],
            attributes=[],
            num_outputs=1,
        )
        graph = ir.Graph(
            [max_trip, cond],
            loop_node.outputs,
            nodes=[loop_node],
            opset_imports={"": 21},
        )
        model = _make_model(graph)
        # Engine catches OpUsageError; output remains None
        infer_symbolic_shapes(model)
        self.assertIsNone(loop_node.outputs[0].shape)

    def test_body_output_none_shape_scan(self):
        """Scan output when body output has no shape → scan output is None."""
        iter_num = ir.Value(
            name="i", type=ir.TensorType(ir.DataType.INT64), shape=ir.Shape([])
        )
        body_cond = ir.Value(
            name="cond_in", type=ir.TensorType(ir.DataType.BOOL), shape=ir.Shape([])
        )
        body_cond_out = ir.Value(
            name="cond_out", type=ir.TensorType(ir.DataType.BOOL), shape=ir.Shape([])
        )
        scan_val = ir.Value(name="scan_val", type=ir.TensorType(ir.DataType.FLOAT))
        body_graph = ir.Graph(
            [iter_num, body_cond],
            [body_cond_out, scan_val],
            nodes=[],
            opset_imports={"": 21},
            name="body",
        )

        max_trip = ir.Value(
            name="max", type=ir.TensorType(ir.DataType.INT64), shape=ir.Shape([])
        )
        cond = ir.Value(name="cond", type=ir.TensorType(ir.DataType.BOOL), shape=ir.Shape([]))
        loop_node = ir.Node(
            "",
            "Loop",
            inputs=[max_trip, cond],
            attributes=[ir.Attr("body", ir.AttributeType.GRAPH, body_graph)],
            num_outputs=1,
        )
        graph = ir.Graph(
            [max_trip, cond],
            loop_node.outputs,
            nodes=[loop_node],
            opset_imports={"": 21},
        )
        model = _make_model(graph)
        infer_symbolic_shapes(model)

        self.assertIsNone(loop_node.outputs[0].shape)
        self.assertEqual(loop_node.outputs[0].dtype, ir.DataType.FLOAT)

    def test_none_cond_raises(self):
        """None cond input raises OpUsageError."""
        body_graph = ir.Graph(
            [
                ir.Value(name="i", type=ir.TensorType(ir.DataType.INT64), shape=ir.Shape([])),
                ir.Value(name="c", type=ir.TensorType(ir.DataType.BOOL), shape=ir.Shape([])),
            ],
            [ir.Value(name="co", type=ir.TensorType(ir.DataType.BOOL), shape=ir.Shape([]))],
            nodes=[],
            opset_imports={"": 21},
            name="body",
        )
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "Loop",
                [
                    ir.Value(name="max", type=ir.TensorType(ir.DataType.INT64)),
                    None,
                ],
                attributes=[ir.Attr("body", ir.AttributeType.GRAPH, body_graph)],
                opset_version=21,
            )

    def test_too_few_inputs_raises(self):
        """Fewer than 2 inputs raises OpUsageError."""
        body_graph = ir.Graph(
            [ir.Value(name="i", type=ir.TensorType(ir.DataType.INT64), shape=ir.Shape([]))],
            [ir.Value(name="co", type=ir.TensorType(ir.DataType.BOOL), shape=ir.Shape([]))],
            nodes=[],
            opset_imports={"": 21},
            name="body",
        )
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "Loop",
                [ir.Value(name="max", type=ir.TensorType(ir.DataType.INT64))],
                attributes=[ir.Attr("body", ir.AttributeType.GRAPH, body_graph)],
                opset_version=21,
            )

    def test_missing_body_attr_raises(self):
        """Missing body attribute raises OpUsageError directly."""
        with self.assertRaises(OpUsageError):
            run_shape_inference_with_values(
                "",
                "Loop",
                [
                    ir.Value(name="max", type=ir.TensorType(ir.DataType.INT64)),
                    ir.Value(name="cond", type=ir.TensorType(ir.DataType.BOOL)),
                ],
                opset_version=21,
            )

    def test_optional_max_trip_count_none(self):
        """max_trip_count can be None without raising."""
        iter_num = ir.Value(
            name="i", type=ir.TensorType(ir.DataType.INT64), shape=ir.Shape([])
        )
        body_cond = ir.Value(
            name="cond_in", type=ir.TensorType(ir.DataType.BOOL), shape=ir.Shape([])
        )
        body_cond_out = ir.Value(
            name="cond_out", type=ir.TensorType(ir.DataType.BOOL), shape=ir.Shape([])
        )
        body_x = ir.Value(
            name="body_x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([3])
        )
        body_graph = ir.Graph(
            [iter_num, body_cond, body_x],
            [body_cond_out, body_x],
            nodes=[],
            opset_imports={"": 21},
            name="body",
        )

        cond = ir.Value(name="cond", type=ir.TensorType(ir.DataType.BOOL), shape=ir.Shape([]))
        x_init = ir.Value(
            name="x_init", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([3])
        )
        loop_node = ir.Node(
            "",
            "Loop",
            inputs=[None, cond, x_init],
            attributes=[ir.Attr("body", ir.AttributeType.GRAPH, body_graph)],
            num_outputs=1,
        )
        graph = ir.Graph(
            [cond, x_init],
            loop_node.outputs,
            nodes=[loop_node],
            opset_imports={"": 21},
        )
        model = _make_model(graph)
        infer_symbolic_shapes(model)

        self.assertEqual(loop_node.outputs[0].shape, ir.Shape([3]))
        self.assertEqual(loop_node.outputs[0].dtype, ir.DataType.FLOAT)


if __name__ == "__main__":
    unittest.main()
