# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the implicit use analysis pass."""

from __future__ import annotations

import unittest

import onnx_ir as ir
from onnx_ir.passes.common import implicit_use_analysis


class ImplicitUseAnalysisPassTest(unittest.TestCase):
    """Test cases for ImplicitUseAnalysisPass."""

    def test_pass_is_in_place(self):
        """Verify the pass is in-place."""
        pass_instance = implicit_use_analysis.ImplicitUseAnalysisPass()
        self.assertTrue(pass_instance.in_place)

    def test_no_subgraphs_no_modification(self):
        """Test that pass doesn't modify models without subgraphs."""
        # Create a simple model without subgraphs
        input_val = ir.val("input", dtype=ir.DataType.FLOAT, shape=[2, 2])
        add_node = ir.node(
            "Add",
            inputs=[input_val, input_val],
            outputs=[ir.val("output", dtype=ir.DataType.FLOAT, shape=[2, 2])],
        )
        graph = ir.Graph(
            inputs=[input_val],
            outputs=[add_node.outputs[0]],
            nodes=[add_node],
            name="test_graph",
        )
        model = ir.Model(graph, ir_version=10)

        # Run the pass
        pass_instance = implicit_use_analysis.ImplicitUseAnalysisPass()
        result = pass_instance(model)

        # Verify no modification
        self.assertFalse(result.modified)
        self.assertNotIn(
            implicit_use_analysis.ImplicitUseAnalysisPass.METADATA_KEY, graph.meta
        )

    def test_simple_if_node_with_captured_values(self):
        """Test If node with subgraphs that capture values from outer scope."""
        # Create main graph values
        condition = ir.val("condition", dtype=ir.DataType.BOOL, shape=[])
        x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[3])
        y = ir.val("y", dtype=ir.DataType.FLOAT, shape=[3])

        # Create then branch that uses x and y from parent scope
        then_add = ir.node(
            "Add",
            inputs=[x, y],
            outputs=[ir.val("then_out", dtype=ir.DataType.FLOAT, shape=[3])],
        )
        then_graph = ir.Graph(
            inputs=[],
            outputs=[then_add.outputs[0]],
            nodes=[then_add],
            name="then_branch",
        )

        # Create else branch that uses only x from parent scope
        else_identity = ir.node(
            "Identity",
            inputs=[x],
            outputs=[ir.val("else_out", dtype=ir.DataType.FLOAT, shape=[3])],
        )
        else_graph = ir.Graph(
            inputs=[],
            outputs=[else_identity.outputs[0]],
            nodes=[else_identity],
            name="else_branch",
        )

        # Create If node
        if_node = ir.node(
            "If",
            inputs=[condition],
            outputs=[ir.val("result", dtype=ir.DataType.FLOAT, shape=[3])],
            attributes={
                "then_branch": ir.Attr("then_branch", ir.AttributeType.GRAPH, then_graph),
                "else_branch": ir.Attr("else_branch", ir.AttributeType.GRAPH, else_graph),
            },
        )

        # Create main graph
        graph = ir.Graph(
            inputs=[condition, x, y],
            outputs=[if_node.outputs[0]],
            nodes=[if_node],
            name="main_graph",
        )
        model = ir.Model(graph, ir_version=10)

        # Run the pass
        pass_instance = implicit_use_analysis.ImplicitUseAnalysisPass()
        result = pass_instance(model)

        # Verify modification
        self.assertTrue(result.modified)

        # Check then_branch captured values
        then_captured = then_graph.meta[
            implicit_use_analysis.ImplicitUseAnalysisPass.METADATA_KEY
        ]
        self.assertEqual(len(then_captured), 2)
        self.assertIn(x, then_captured)
        self.assertIn(y, then_captured)

        # Check else_branch captured values
        else_captured = else_graph.meta[
            implicit_use_analysis.ImplicitUseAnalysisPass.METADATA_KEY
        ]
        self.assertEqual(len(else_captured), 1)
        self.assertIn(x, else_captured)

    def test_subgraph_with_no_captured_values(self):
        """Test subgraph that doesn't capture any outer values."""
        condition = ir.val("condition", dtype=ir.DataType.BOOL, shape=[])

        # Create then branch with its own inputs (no captures)
        then_input = ir.val("then_input", dtype=ir.DataType.FLOAT, shape=[3])
        then_identity = ir.node(
            "Identity",
            inputs=[then_input],
            outputs=[ir.val("then_out", dtype=ir.DataType.FLOAT, shape=[3])],
        )
        then_graph = ir.Graph(
            inputs=[then_input],
            outputs=[then_identity.outputs[0]],
            nodes=[then_identity],
            name="then_branch",
        )

        # Create else branch with constant (no captures)
        else_const = ir.node(
            "Constant",
            inputs=[],
            outputs=[ir.val("else_out", dtype=ir.DataType.FLOAT, shape=[3])],
        )
        else_graph = ir.Graph(
            inputs=[],
            outputs=[else_const.outputs[0]],
            nodes=[else_const],
            name="else_branch",
        )

        # Create If node
        if_node = ir.node(
            "If",
            inputs=[condition],
            outputs=[ir.val("result", dtype=ir.DataType.FLOAT, shape=[3])],
            attributes={
                "then_branch": ir.Attr("then_branch", ir.AttributeType.GRAPH, then_graph),
                "else_branch": ir.Attr("else_branch", ir.AttributeType.GRAPH, else_graph),
            },
        )

        graph = ir.Graph(
            inputs=[condition],
            outputs=[if_node.outputs[0]],
            nodes=[if_node],
            name="main_graph",
        )
        model = ir.Model(graph, ir_version=10)

        # Run the pass
        pass_instance = implicit_use_analysis.ImplicitUseAnalysisPass()
        result = pass_instance(model)

        # Verify modification (metadata is still added, even if empty)
        self.assertTrue(result.modified)

        # Check captured values are empty
        then_captured = then_graph.meta[
            implicit_use_analysis.ImplicitUseAnalysisPass.METADATA_KEY
        ]
        else_captured = else_graph.meta[
            implicit_use_analysis.ImplicitUseAnalysisPass.METADATA_KEY
        ]
        self.assertEqual(len(then_captured), 0)
        self.assertEqual(len(else_captured), 0)

    def test_nested_subgraphs(self):
        """Test nested subgraphs (If inside If)."""
        # Create main graph values
        condition1 = ir.val("condition1", dtype=ir.DataType.BOOL, shape=[])
        condition2 = ir.val("condition2", dtype=ir.DataType.BOOL, shape=[])
        x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[3])

        # Create innermost graph that uses x from outer outer scope
        inner_identity = ir.node(
            "Identity",
            inputs=[x],
            outputs=[ir.val("inner_out", dtype=ir.DataType.FLOAT, shape=[3])],
        )
        inner_graph = ir.Graph(
            inputs=[],
            outputs=[inner_identity.outputs[0]],
            nodes=[inner_identity],
            name="inner_graph",
        )

        # Create middle If node
        middle_if = ir.node(
            "If",
            inputs=[condition2],
            outputs=[ir.val("middle_out", dtype=ir.DataType.FLOAT, shape=[3])],
            attributes={
                "then_branch": ir.Attr("then_branch", ir.AttributeType.GRAPH, inner_graph),
                "else_branch": ir.Attr("else_branch", ir.AttributeType.GRAPH, inner_graph),
            },
        )

        # Create middle graph containing the middle If
        middle_graph = ir.Graph(
            inputs=[],
            outputs=[middle_if.outputs[0]],
            nodes=[middle_if],
            name="middle_graph",
        )

        # Create outer If node
        outer_if = ir.node(
            "If",
            inputs=[condition1],
            outputs=[ir.val("result", dtype=ir.DataType.FLOAT, shape=[3])],
            attributes={
                "then_branch": ir.Attr("then_branch", ir.AttributeType.GRAPH, middle_graph),
                "else_branch": ir.Attr("else_branch", ir.AttributeType.GRAPH, middle_graph),
            },
        )

        # Create main graph
        graph = ir.Graph(
            inputs=[condition1, condition2, x],
            outputs=[outer_if.outputs[0]],
            nodes=[outer_if],
            name="main_graph",
        )
        model = ir.Model(graph, ir_version=10)

        # Run the pass
        pass_instance = implicit_use_analysis.ImplicitUseAnalysisPass()
        result = pass_instance(model)

        # Verify modification
        self.assertTrue(result.modified)

        # Check innermost graph captures x from main graph
        inner_captured = inner_graph.meta[
            implicit_use_analysis.ImplicitUseAnalysisPass.METADATA_KEY
        ]
        self.assertEqual(len(inner_captured), 1)
        self.assertIn(x, inner_captured)

        # Check middle graph captures x and condition2
        middle_captured = middle_graph.meta[
            implicit_use_analysis.ImplicitUseAnalysisPass.METADATA_KEY
        ]
        self.assertGreater(len(middle_captured), 0)
        # Middle graph should capture x (used by inner graph) and condition2 (used by middle_if)
        self.assertIn(x, middle_captured)
        self.assertIn(condition2, middle_captured)

    def test_loop_node_with_captured_values(self):
        """Test Loop node with body graph that captures values."""
        # Create main graph values
        max_iter = ir.val("max_iter", dtype=ir.DataType.INT64, shape=[])
        cond_in = ir.val("cond_in", dtype=ir.DataType.BOOL, shape=[])
        captured_val = ir.val("captured_val", dtype=ir.DataType.FLOAT, shape=[3])

        # Create loop body that uses captured_val
        iter_num = ir.val("iter_num", dtype=ir.DataType.INT64, shape=[])
        cond = ir.val("cond", dtype=ir.DataType.BOOL, shape=[])

        add_node = ir.node(
            "Add",
            inputs=[iter_num, captured_val],  # Uses captured_val from outer scope
            outputs=[ir.val("body_out", dtype=ir.DataType.FLOAT, shape=[3])],
        )

        loop_body = ir.Graph(
            inputs=[iter_num, cond],
            outputs=[cond, add_node.outputs[0]],
            nodes=[add_node],
            name="loop_body",
        )

        # Create Loop node
        loop_node = ir.node(
            "Loop",
            inputs=[max_iter, cond_in],
            outputs=[ir.val("result", dtype=ir.DataType.FLOAT, shape=[3])],
            attributes={
                "body": ir.Attr("body", ir.AttributeType.GRAPH, loop_body),
            },
        )

        graph = ir.Graph(
            inputs=[max_iter, cond_in, captured_val],
            outputs=[loop_node.outputs[0]],
            nodes=[loop_node],
            name="main_graph",
        )
        model = ir.Model(graph, ir_version=10)

        # Run the pass
        pass_instance = implicit_use_analysis.ImplicitUseAnalysisPass()
        result = pass_instance(model)

        # Verify modification
        self.assertTrue(result.modified)

        # Check loop body captured values
        body_captured = loop_body.meta[
            implicit_use_analysis.ImplicitUseAnalysisPass.METADATA_KEY
        ]
        self.assertEqual(len(body_captured), 1)
        self.assertIn(captured_val, body_captured)

    def test_duplicate_captures_are_deduplicated(self):
        """Test that duplicate captured values are deduplicated."""
        condition = ir.val("condition", dtype=ir.DataType.BOOL, shape=[])
        x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[3])

        # Create subgraph that uses x multiple times
        add1 = ir.node(
            "Add",
            inputs=[x, x],
            outputs=[ir.val("add1_out", dtype=ir.DataType.FLOAT, shape=[3])],
        )
        add2 = ir.node(
            "Add",
            inputs=[add1.outputs[0], x],  # Uses x again
            outputs=[ir.val("add2_out", dtype=ir.DataType.FLOAT, shape=[3])],
        )

        then_graph = ir.Graph(
            inputs=[],
            outputs=[add2.outputs[0]],
            nodes=[add1, add2],
            name="then_branch",
        )

        else_graph = ir.Graph(
            inputs=[],
            outputs=[ir.val("else_out", dtype=ir.DataType.FLOAT, shape=[3])],
            nodes=[],
            name="else_branch",
        )

        if_node = ir.node(
            "If",
            inputs=[condition],
            outputs=[ir.val("result", dtype=ir.DataType.FLOAT, shape=[3])],
            attributes={
                "then_branch": ir.Attr("then_branch", ir.AttributeType.GRAPH, then_graph),
                "else_branch": ir.Attr("else_branch", ir.AttributeType.GRAPH, else_graph),
            },
        )

        graph = ir.Graph(
            inputs=[condition, x],
            outputs=[if_node.outputs[0]],
            nodes=[if_node],
            name="main_graph",
        )
        model = ir.Model(graph, ir_version=10)

        # Run the pass
        pass_instance = implicit_use_analysis.ImplicitUseAnalysisPass()
        result = pass_instance(model)

        # Verify modification
        self.assertTrue(result.modified)

        # Check that x appears only once in captured values
        then_captured = then_graph.meta[
            implicit_use_analysis.ImplicitUseAnalysisPass.METADATA_KEY
        ]
        self.assertEqual(len(then_captured), 1)
        self.assertEqual(then_captured[0], x)

    def test_multiple_subgraphs_in_graphs_attribute(self):
        """Test node with GRAPHS attribute (multiple subgraphs)."""
        # Create a Scan node which has multiple subgraphs
        scan_input = ir.val("scan_input", dtype=ir.DataType.FLOAT, shape=[3])
        captured = ir.val("captured", dtype=ir.DataType.FLOAT, shape=[3])

        # Create body1 that captures a value
        body1_in = ir.val("body1_in", dtype=ir.DataType.FLOAT, shape=[3])
        body1_add = ir.node(
            "Add",
            inputs=[body1_in, captured],
            outputs=[ir.val("body1_out", dtype=ir.DataType.FLOAT, shape=[3])],
        )
        body1 = ir.Graph(
            inputs=[body1_in],
            outputs=[body1_add.outputs[0]],
            nodes=[body1_add],
            name="body1",
        )

        # Create body2 that also captures the same value
        body2_in = ir.val("body2_in", dtype=ir.DataType.FLOAT, shape=[3])
        body2_mul = ir.node(
            "Mul",
            inputs=[body2_in, captured],
            outputs=[ir.val("body2_out", dtype=ir.DataType.FLOAT, shape=[3])],
        )
        body2 = ir.Graph(
            inputs=[body2_in],
            outputs=[body2_mul.outputs[0]],
            nodes=[body2_mul],
            name="body2",
        )

        # Create a custom node with GRAPHS attribute
        custom_node = ir.node(
            "CustomOp",
            inputs=[scan_input],
            outputs=[ir.val("result", dtype=ir.DataType.FLOAT, shape=[3])],
            attributes={
                "bodies": ir.Attr("bodies", ir.AttributeType.GRAPHS, [body1, body2]),
            },
        )

        graph = ir.Graph(
            inputs=[scan_input, captured],
            outputs=[custom_node.outputs[0]],
            nodes=[custom_node],
            name="main_graph",
        )
        model = ir.Model(graph, ir_version=10)

        # Run the pass
        pass_instance = implicit_use_analysis.ImplicitUseAnalysisPass()
        result = pass_instance(model)

        # Verify modification
        self.assertTrue(result.modified)

        # Check both bodies captured the value
        body1_captured = body1.meta[implicit_use_analysis.ImplicitUseAnalysisPass.METADATA_KEY]
        body2_captured = body2.meta[implicit_use_analysis.ImplicitUseAnalysisPass.METADATA_KEY]

        self.assertEqual(len(body1_captured), 1)
        self.assertIn(captured, body1_captured)

        self.assertEqual(len(body2_captured), 1)
        self.assertIn(captured, body2_captured)

    def test_metadata_key_is_accessible(self):
        """Test that the METADATA_KEY constant is properly set."""
        self.assertEqual(
            implicit_use_analysis.ImplicitUseAnalysisPass.METADATA_KEY,
            "pkg.onnx_ir.ImplicitUseAnalysisPass.values",
        )


if __name__ == "__main__":
    unittest.main()
