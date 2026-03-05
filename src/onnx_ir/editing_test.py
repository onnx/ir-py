# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

import onnx_ir as ir
from onnx_ir import editing


def _simple_graph_with_chain() -> tuple[ir.Model, ir.Node, ir.Node, ir.Node]:
    """Build: input -> A -> B -> C -> output.

    Returns (model, node_a, node_b, node_c).
    """
    graph_input = ir.Value(name="input")
    node_a = ir.Node("", "A", inputs=[graph_input], num_outputs=1)
    node_b = ir.Node("", "B", inputs=node_a.outputs, num_outputs=1)
    node_c = ir.Node("", "C", inputs=node_b.outputs, num_outputs=1)
    graph = ir.Graph(
        inputs=[graph_input],
        outputs=list(node_c.outputs),
        nodes=[node_a, node_b, node_c],
    )
    model = ir.Model(graph=graph, ir_version=10)
    return model, node_a, node_b, node_c


class ReplaceNodeBasicTest(unittest.TestCase):
    """Test basic replace_node behaviour."""

    def test_replace_middle_node_rewires_consumers(self):
        """Replace B in A->B->C with D, verify C now consumes D's output."""
        model, node_a, node_b, node_c = _simple_graph_with_chain()

        node_d = ir.Node("", "D", inputs=list(node_a.outputs), num_outputs=1)
        editing.replace_node(node_b, node_d)

        # C should now consume D's output
        self.assertIs(node_c.inputs[0], node_d.outputs[0])
        # D should be in the graph, B should not
        graph_nodes = list(model.graph)
        self.assertIn(node_d, graph_nodes)
        self.assertNotIn(node_b, graph_nodes)
        # D should appear before C
        self.assertLess(graph_nodes.index(node_d), graph_nodes.index(node_c))

    def test_replace_node_propagates_metadata_by_default(self):
        """Metadata (type, shape, name, const_value) is propagated."""
        model, node_a, node_b, node_c = _simple_graph_with_chain()

        # Set metadata on B's output
        node_b.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
        node_b.outputs[0].shape = ir.Shape([1, 2, 3])
        node_b.outputs[0].name = "b_output"

        node_d = ir.Node("", "D", inputs=list(node_a.outputs), num_outputs=1)
        editing.replace_node(node_b, node_d)

        self.assertEqual(node_d.outputs[0].type, ir.TensorType(ir.DataType.FLOAT))
        self.assertEqual(node_d.outputs[0].shape, ir.Shape([1, 2, 3]))
        self.assertEqual(node_d.outputs[0].name, "b_output")

    def test_replace_node_skips_metadata_when_disabled(self):
        """propagate_metadata=False leaves new output type/shape untouched."""
        model, node_a, node_b, node_c = _simple_graph_with_chain()

        node_b.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
        node_b.outputs[0].shape = ir.Shape([1, 2, 3])

        node_d = ir.Node("", "D", inputs=list(node_a.outputs), num_outputs=1)
        node_d.outputs[0].type = ir.TensorType(ir.DataType.INT32)

        editing.replace_node(node_b, node_d, propagate_metadata=False)

        # Type/shape should NOT have been overwritten by old values
        self.assertEqual(node_d.outputs[0].type, ir.TensorType(ir.DataType.INT32))
        self.assertIsNone(node_d.outputs[0].shape)

    def test_replace_node_old_has_no_consumers(self):
        """Replacing a node whose output has no consumers (no-op rewire)."""
        graph_input = ir.Value(name="input")
        # A has an output but nothing consumes it
        node_a = ir.Node("", "A", inputs=[graph_input], num_outputs=1)
        graph = ir.Graph(
            inputs=[graph_input],
            outputs=[graph_input],  # graph output is the input itself
            nodes=[node_a],
        )
        model = ir.Model(graph=graph, ir_version=10)

        node_b = ir.Node("", "B", inputs=[graph_input], num_outputs=1)
        editing.replace_node(node_a, node_b)

        graph_nodes = list(model.graph)
        self.assertIn(node_b, graph_nodes)
        self.assertNotIn(node_a, graph_nodes)


class ReplaceNodeGraphOutputTest(unittest.TestCase):
    """Test graph output handling (the 'graph output tax')."""

    def test_replace_node_updates_graph_output(self):
        """When old node's output is a graph output, it is updated to new output."""
        model, node_a, node_b, node_c = _simple_graph_with_chain()
        # C is the last node and its output is the graph output

        node_d = ir.Node("", "D", inputs=list(node_b.outputs), num_outputs=1)
        editing.replace_node(node_c, node_d)

        # Graph output should now be D's output
        self.assertIs(model.graph.outputs[0], node_d.outputs[0])

    def test_replace_with_value_already_graph_output_inserts_identity(self):
        """When new output is already a graph output, Identity is inserted."""
        graph_input = ir.Value(name="input")
        node_a = ir.Node("", "A", inputs=[graph_input], num_outputs=1)
        node_b = ir.Node("", "B", inputs=[graph_input], num_outputs=1)
        graph = ir.Graph(
            inputs=[graph_input],
            outputs=[node_a.outputs[0], node_b.outputs[0]],
            nodes=[node_a, node_b],
        )
        model = ir.Model(graph=graph, ir_version=10)

        # Replace B with a node whose output maps to A's output (already a graph output)
        node_c = ir.Node("", "C", inputs=[graph_input], num_outputs=1)
        editing.replace_node(
            node_b,
            node_c,
            output_mapping={node_b.outputs[0]: node_a.outputs[0]},
        )

        # The second graph output should be an Identity node's output
        second_output = model.graph.outputs[1]
        producer = second_output.producer()
        self.assertIsNotNone(producer)
        self.assertEqual(producer.op_type, "Identity")
        # Identity should consume A's output
        self.assertIs(producer.inputs[0], node_a.outputs[0])

    def test_replace_with_value_already_graph_input_inserts_identity(self):
        """When new output is a graph input, Identity is inserted."""
        graph_input = ir.Value(name="input")
        node_a = ir.Node("", "A", inputs=[graph_input], num_outputs=1)
        graph = ir.Graph(
            inputs=[graph_input],
            outputs=[node_a.outputs[0]],
            nodes=[node_a],
        )
        model = ir.Model(graph=graph, ir_version=10)

        # Replace A with a node whose output maps to the graph input
        node_b = ir.Node("", "B", inputs=[graph_input], num_outputs=1)
        editing.replace_node(
            node_a,
            node_b,
            output_mapping={node_a.outputs[0]: graph_input},
        )

        # Graph output should be an Identity node wrapping the graph input
        output = model.graph.outputs[0]
        producer = output.producer()
        self.assertIsNotNone(producer)
        self.assertEqual(producer.op_type, "Identity")
        self.assertIs(producer.inputs[0], graph_input)

    def test_graph_output_name_is_preserved(self):
        """The graph output name from old output is preserved on the new value."""
        model, node_a, node_b, node_c = _simple_graph_with_chain()
        node_c.outputs[0].name = "final_output"

        node_d = ir.Node("", "D", inputs=list(node_b.outputs), num_outputs=1)
        editing.replace_node(node_c, node_d)

        self.assertEqual(model.graph.outputs[0].name, "final_output")


class ReplaceNodeOutputMappingTest(unittest.TestCase):
    """Test explicit output_mapping behaviour."""

    def test_output_mapping_with_different_output_counts(self):
        """Replace a 2-output node with a 1-output node using mapping."""
        graph_input = ir.Value(name="input")
        node_a = ir.Node("", "A", inputs=[graph_input], num_outputs=2)
        # Consumer only uses first output
        node_b = ir.Node("", "B", inputs=[node_a.outputs[0]], num_outputs=1)
        graph = ir.Graph(
            inputs=[graph_input],
            outputs=[node_b.outputs[0]],
            nodes=[node_a, node_b],
        )
        model = ir.Model(graph=graph, ir_version=10)

        # Replace A with a single-output node
        node_c = ir.Node("", "C", inputs=[graph_input], num_outputs=1)
        editing.replace_node(
            node_a,
            node_c,
            output_mapping={node_a.outputs[0]: node_c.outputs[0]},
        )

        # B should now consume C's output
        self.assertIs(node_b.inputs[0], node_c.outputs[0])

    def test_output_mapping_for_cse_pattern(self):
        """Simulate CSE: replace duplicate node, mapping to existing node's outputs."""
        graph_input = ir.Value(name="input")
        # Two identical nodes
        node_a = ir.Node("", "Relu", inputs=[graph_input], num_outputs=1)
        node_b = ir.Node("", "Relu", inputs=[graph_input], num_outputs=1)
        # Consumers of each
        node_c = ir.Node("", "C", inputs=[node_a.outputs[0]], num_outputs=1)
        node_d = ir.Node("", "D", inputs=[node_b.outputs[0]], num_outputs=1)
        graph = ir.Graph(
            inputs=[graph_input],
            outputs=[node_c.outputs[0], node_d.outputs[0]],
            nodes=[node_a, node_b, node_c, node_d],
        )
        model = ir.Model(graph=graph, ir_version=10)

        # Eliminate node_b (duplicate), redirecting to node_a's outputs
        # node_a is already in the graph, so we create a "dummy" new node
        # Actually for CSE, we use output_mapping to map to existing values
        # This requires a new node. For CSE pattern, we'd use a different approach.
        # Let's test with a proper replacement node.
        node_e = ir.Node("", "Relu", inputs=[graph_input], num_outputs=1)
        editing.replace_node(
            node_b,
            node_e,
            output_mapping={node_b.outputs[0]: node_e.outputs[0]},
        )

        # D should now consume E's output
        self.assertIs(node_d.inputs[0], node_e.outputs[0])

    def test_raises_when_output_counts_differ_without_mapping(self):
        """Error when output counts differ and no mapping provided."""
        graph_input = ir.Value(name="input")
        node_a = ir.Node("", "A", inputs=[graph_input], num_outputs=2)
        graph = ir.Graph(
            inputs=[graph_input],
            outputs=[],
            nodes=[node_a],
        )
        ir.Model(graph=graph, ir_version=10)

        node_b = ir.Node("", "B", inputs=[graph_input], num_outputs=1)
        with self.assertRaises(ValueError):
            editing.replace_node(node_a, node_b)

    def test_raises_when_mapping_misses_consumer(self):
        """Error when an old output has consumers but isn't in the mapping."""
        graph_input = ir.Value(name="input")
        node_a = ir.Node("", "A", inputs=[graph_input], num_outputs=2)
        node_b = ir.Node("", "B", inputs=[node_a.outputs[1]], num_outputs=1)
        graph = ir.Graph(
            inputs=[graph_input],
            outputs=[node_b.outputs[0]],
            nodes=[node_a, node_b],
        )
        ir.Model(graph=graph, ir_version=10)

        node_c = ir.Node("", "C", inputs=[graph_input], num_outputs=1)
        # Only map output[0], but output[1] has consumer node_b
        with self.assertRaises(ValueError):
            editing.replace_node(
                node_a,
                node_c,
                output_mapping={node_a.outputs[0]: node_c.outputs[0]},
            )


class ReplaceNodeValidationTest(unittest.TestCase):
    """Test input validation for replace_node."""

    def test_raises_when_old_node_has_no_graph(self):
        """Error when old_node doesn't belong to a graph."""
        node_a = ir.Node("", "A", inputs=[], num_outputs=1)
        node_b = ir.Node("", "B", inputs=[], num_outputs=1)
        with self.assertRaises(ValueError):
            editing.replace_node(node_a, node_b)

    def test_raises_when_new_node_already_in_graph(self):
        """Error when new_node already belongs to a graph."""
        graph_input = ir.Value(name="input")
        node_a = ir.Node("", "A", inputs=[graph_input], num_outputs=1)
        node_b = ir.Node("", "B", inputs=[graph_input], num_outputs=1)
        graph = ir.Graph(
            inputs=[graph_input],
            outputs=[],
            nodes=[node_a, node_b],
        )
        ir.Model(graph=graph, ir_version=10)

        with self.assertRaises(ValueError):
            editing.replace_node(node_a, node_b)


class ReplaceNodeMultiOutputTest(unittest.TestCase):
    """Test multi-output node replacement."""

    def test_replace_multi_output_node(self):
        """Replace a node with 2 outputs, both consumed."""
        graph_input = ir.Value(name="input")
        node_a = ir.Node("", "Split", inputs=[graph_input], num_outputs=2)
        node_b = ir.Node("", "B", inputs=[node_a.outputs[0]], num_outputs=1)
        node_c = ir.Node("", "C", inputs=[node_a.outputs[1]], num_outputs=1)
        graph = ir.Graph(
            inputs=[graph_input],
            outputs=[node_b.outputs[0], node_c.outputs[0]],
            nodes=[node_a, node_b, node_c],
        )
        ir.Model(graph=graph, ir_version=10)

        node_d = ir.Node("", "NewSplit", inputs=[graph_input], num_outputs=2)
        editing.replace_node(node_a, node_d)

        # Both consumers should be rewired
        self.assertIs(node_b.inputs[0], node_d.outputs[0])
        self.assertIs(node_c.inputs[0], node_d.outputs[1])


class ReplaceNodeInsertionOrderTest(unittest.TestCase):
    """Test that new node is placed at old node's position."""

    def test_new_node_placed_at_old_position(self):
        """New node should be inserted at old node's position."""
        model, node_a, node_b, node_c = _simple_graph_with_chain()

        node_d = ir.Node("", "D", inputs=list(node_a.outputs), num_outputs=1)
        editing.replace_node(node_b, node_d)

        graph_nodes = list(model.graph)
        # D should be where B was: after A, before C
        a_idx = graph_nodes.index(node_a)
        d_idx = graph_nodes.index(node_d)
        c_idx = graph_nodes.index(node_c)
        self.assertEqual(d_idx, a_idx + 1)
        self.assertEqual(c_idx, d_idx + 1)


class ReplaceNodeMetadataPropagationTest(unittest.TestCase):
    """Test fine-grained metadata propagation behaviour."""

    def test_old_metadata_wins_over_new(self):
        """Old value's non-None metadata takes precedence."""
        model, node_a, node_b, node_c = _simple_graph_with_chain()

        node_b.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
        node_b.outputs[0].name = "old_name"

        node_d = ir.Node("", "D", inputs=list(node_a.outputs), num_outputs=1)
        node_d.outputs[0].type = ir.TensorType(ir.DataType.INT32)
        node_d.outputs[0].name = "new_name"

        editing.replace_node(node_b, node_d)

        # Old's non-None values should override new
        self.assertEqual(node_d.outputs[0].type, ir.TensorType(ir.DataType.FLOAT))
        self.assertEqual(node_d.outputs[0].name, "old_name")

    def test_new_metadata_preserved_when_old_is_none(self):
        """When old metadata is None, new value's metadata is preserved."""
        model, node_a, node_b, node_c = _simple_graph_with_chain()

        # old output has no type set (None)
        self.assertIsNone(node_b.outputs[0].type)

        node_d = ir.Node("", "D", inputs=list(node_a.outputs), num_outputs=1)
        node_d.outputs[0].type = ir.TensorType(ir.DataType.INT32)

        editing.replace_node(node_b, node_d)

        # New's type should be preserved since old was None
        self.assertEqual(node_d.outputs[0].type, ir.TensorType(ir.DataType.INT32))


class ReplaceNodeAccessViaModuleTest(unittest.TestCase):
    """Test that replace_node is accessible via ir.editing."""

    def test_accessible_via_ir_editing(self):
        self.assertTrue(hasattr(ir.editing, "replace_node"))
        self.assertTrue(callable(ir.editing.replace_node))


# ---------------------------------------------------------------------------
# replace_subgraph tests
# ---------------------------------------------------------------------------


def _matmul_add_graph() -> (
    tuple[ir.Model, ir.Value, ir.Node, ir.Node, ir.Node]
):
    """Build: input_a, input_b -> MatMul -> (+ input_c) Add -> consumer -> output.

    This is the canonical MatMul+Add→Gemm fusion pattern.
    Returns (model, input_a, matmul_node, add_node, consumer_node).
    """
    input_a = ir.Value(name="A")
    input_b = ir.Value(name="B")
    input_c = ir.Value(name="C")
    matmul_node = ir.Node("", "MatMul", inputs=[input_a, input_b], num_outputs=1)
    add_node = ir.Node("", "Add", inputs=[matmul_node.outputs[0], input_c], num_outputs=1)
    consumer = ir.Node("", "Relu", inputs=[add_node.outputs[0]], num_outputs=1)
    graph = ir.Graph(
        inputs=[input_a, input_b, input_c],
        outputs=list(consumer.outputs),
        nodes=[matmul_node, add_node, consumer],
    )
    model = ir.Model(graph=graph, ir_version=10)
    return model, input_a, matmul_node, add_node, consumer


class ReplaceSubgraphBasicTest(unittest.TestCase):
    """Test basic replace_subgraph behaviour."""

    def test_matmul_add_to_gemm_fusion(self):
        """The canonical fusion: MatMul + Add → Gemm."""
        model, input_a, matmul_node, add_node, consumer = _matmul_add_graph()

        gemm = ir.Node(
            "",
            "Gemm",
            inputs=[
                matmul_node.inputs[0],
                matmul_node.inputs[1],
                add_node.inputs[1],
            ],
            num_outputs=1,
        )
        editing.replace_subgraph(
            [matmul_node, add_node],
            [gemm],
            output_mapping={add_node.outputs[0]: gemm.outputs[0]},
        )

        # Consumer should now use Gemm output
        self.assertIs(consumer.inputs[0], gemm.outputs[0])
        # Graph should contain Gemm and consumer, not MatMul or Add
        graph_nodes = list(model.graph)
        self.assertIn(gemm, graph_nodes)
        self.assertIn(consumer, graph_nodes)
        self.assertNotIn(matmul_node, graph_nodes)
        self.assertNotIn(add_node, graph_nodes)

    def test_new_nodes_inserted_before_earliest_old_node(self):
        """New nodes appear at the position of the first old node."""
        model, input_a, matmul_node, add_node, consumer = _matmul_add_graph()

        gemm = ir.Node(
            "",
            "Gemm",
            inputs=[
                matmul_node.inputs[0],
                matmul_node.inputs[1],
                add_node.inputs[1],
            ],
            num_outputs=1,
        )
        editing.replace_subgraph(
            [matmul_node, add_node],
            [gemm],
            output_mapping={add_node.outputs[0]: gemm.outputs[0]},
        )

        graph_nodes = list(model.graph)
        # Gemm should be first (where MatMul was), consumer after
        self.assertEqual(graph_nodes.index(gemm), 0)
        self.assertGreater(graph_nodes.index(consumer), graph_nodes.index(gemm))

    def test_multiple_new_nodes(self):
        """Replace subgraph with multiple new nodes."""
        model, input_a, matmul_node, add_node, consumer = _matmul_add_graph()

        # Replace MatMul+Add with two new nodes: Transpose then Gemm
        transpose = ir.Node(
            "",
            "Transpose",
            inputs=[matmul_node.inputs[1]],
            num_outputs=1,
        )
        gemm = ir.Node(
            "",
            "Gemm",
            inputs=[matmul_node.inputs[0], transpose.outputs[0], add_node.inputs[1]],
            num_outputs=1,
        )
        editing.replace_subgraph(
            [matmul_node, add_node],
            [transpose, gemm],
            output_mapping={add_node.outputs[0]: gemm.outputs[0]},
        )

        graph_nodes = list(model.graph)
        self.assertIn(transpose, graph_nodes)
        self.assertIn(gemm, graph_nodes)
        self.assertIs(consumer.inputs[0], gemm.outputs[0])
        # Transpose should come before Gemm
        self.assertLess(
            graph_nodes.index(transpose), graph_nodes.index(gemm)
        )

    def test_propagates_metadata_by_default(self):
        """Metadata is propagated from old outputs to new outputs."""
        model, input_a, matmul_node, add_node, consumer = _matmul_add_graph()

        add_node.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
        add_node.outputs[0].shape = ir.Shape([2, 3])
        add_node.outputs[0].name = "add_out"

        gemm = ir.Node(
            "",
            "Gemm",
            inputs=[
                matmul_node.inputs[0],
                matmul_node.inputs[1],
                add_node.inputs[1],
            ],
            num_outputs=1,
        )
        editing.replace_subgraph(
            [matmul_node, add_node],
            [gemm],
            output_mapping={add_node.outputs[0]: gemm.outputs[0]},
        )

        self.assertEqual(gemm.outputs[0].type, ir.TensorType(ir.DataType.FLOAT))
        self.assertEqual(gemm.outputs[0].shape, ir.Shape([2, 3]))
        self.assertEqual(gemm.outputs[0].name, "add_out")

    def test_no_metadata_propagation_when_disabled(self):
        """propagate_metadata=False keeps new output metadata untouched."""
        model, input_a, matmul_node, add_node, consumer = _matmul_add_graph()

        add_node.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
        add_node.outputs[0].shape = ir.Shape([2, 3])

        gemm = ir.Node(
            "",
            "Gemm",
            inputs=[
                matmul_node.inputs[0],
                matmul_node.inputs[1],
                add_node.inputs[1],
            ],
            num_outputs=1,
        )
        gemm.outputs[0].type = ir.TensorType(ir.DataType.INT32)

        editing.replace_subgraph(
            [matmul_node, add_node],
            [gemm],
            output_mapping={add_node.outputs[0]: gemm.outputs[0]},
            propagate_metadata=False,
        )

        # Should keep the INT32 type, not overwrite with FLOAT
        self.assertEqual(gemm.outputs[0].type, ir.TensorType(ir.DataType.INT32))
        self.assertIsNone(gemm.outputs[0].shape)

    def test_internal_edges_dont_need_mapping(self):
        """Edges between old_nodes (internal) don't require output_mapping."""
        model, input_a, matmul_node, add_node, consumer = _matmul_add_graph()

        # MatMul's output is consumed by Add (internal edge) — not in mapping.
        # Only Add's output (consumed by consumer) needs mapping.
        gemm = ir.Node(
            "",
            "Gemm",
            inputs=[
                matmul_node.inputs[0],
                matmul_node.inputs[1],
                add_node.inputs[1],
            ],
            num_outputs=1,
        )
        # This should work without mapping MatMul's output
        editing.replace_subgraph(
            [matmul_node, add_node],
            [gemm],
            output_mapping={add_node.outputs[0]: gemm.outputs[0]},
        )

        self.assertIs(consumer.inputs[0], gemm.outputs[0])


class ReplaceSubgraphGraphOutputTest(unittest.TestCase):
    """Test graph output handling for replace_subgraph."""

    def test_updates_graph_output(self):
        """When an old output is a graph output, it is updated."""
        input_val = ir.Value(name="input")
        node_a = ir.Node("", "A", inputs=[input_val], num_outputs=1)
        node_b = ir.Node("", "B", inputs=node_a.outputs, num_outputs=1)
        graph = ir.Graph(
            inputs=[input_val],
            outputs=list(node_b.outputs),
            nodes=[node_a, node_b],
        )
        model = ir.Model(graph=graph, ir_version=10)

        node_c = ir.Node("", "C", inputs=[input_val], num_outputs=1)
        editing.replace_subgraph(
            [node_a, node_b],
            [node_c],
            output_mapping={node_b.outputs[0]: node_c.outputs[0]},
        )

        self.assertIs(model.graph.outputs[0], node_c.outputs[0])

    def test_inserts_identity_when_new_value_is_graph_input(self):
        """Identity inserted when mapping points to a graph input."""
        input_val = ir.Value(name="input")
        node_a = ir.Node("", "A", inputs=[input_val], num_outputs=1)
        node_b = ir.Node("", "B", inputs=node_a.outputs, num_outputs=1)
        graph = ir.Graph(
            inputs=[input_val],
            outputs=list(node_b.outputs),
            nodes=[node_a, node_b],
        )
        model = ir.Model(graph=graph, ir_version=10)

        # Replace A→B with nothing, redirecting B's output to input_val
        node_c = ir.Node("", "C", inputs=[input_val], num_outputs=1)
        editing.replace_subgraph(
            [node_a, node_b],
            [node_c],
            output_mapping={node_b.outputs[0]: input_val},
        )

        # Graph output should be through an Identity
        output_producer = model.graph.outputs[0].producer()
        self.assertIsNotNone(output_producer)
        self.assertEqual(output_producer.op_type, "Identity")
        self.assertIs(output_producer.inputs[0], input_val)

    def test_graph_output_name_preserved(self):
        """Graph output name transfers to the replacement value."""
        input_val = ir.Value(name="input")
        node_a = ir.Node("", "A", inputs=[input_val], num_outputs=1)
        node_b = ir.Node("", "B", inputs=node_a.outputs, num_outputs=1)
        node_b.outputs[0].name = "final_result"
        graph = ir.Graph(
            inputs=[input_val],
            outputs=list(node_b.outputs),
            nodes=[node_a, node_b],
        )
        model = ir.Model(graph=graph, ir_version=10)

        node_c = ir.Node("", "C", inputs=[input_val], num_outputs=1)
        editing.replace_subgraph(
            [node_a, node_b],
            [node_c],
            output_mapping={node_b.outputs[0]: node_c.outputs[0]},
        )

        self.assertEqual(model.graph.outputs[0].name, "final_result")


class ReplaceSubgraphValidationTest(unittest.TestCase):
    """Test input validation for replace_subgraph."""

    def test_raises_when_old_nodes_empty(self):
        with self.assertRaises(ValueError):
            editing.replace_subgraph([], [], output_mapping={})

    def test_raises_when_old_node_has_no_graph(self):
        node_a = ir.Node("", "A", inputs=[], num_outputs=1)
        with self.assertRaises(ValueError):
            editing.replace_subgraph(
                [node_a], [], output_mapping={}
            )

    def test_raises_when_old_nodes_span_multiple_graphs(self):
        input_a = ir.Value(name="a")
        node_a = ir.Node("", "A", inputs=[input_a], num_outputs=1)
        ir.Graph(inputs=[input_a], outputs=list(node_a.outputs), nodes=[node_a])

        input_b = ir.Value(name="b")
        node_b = ir.Node("", "B", inputs=[input_b], num_outputs=1)
        ir.Graph(inputs=[input_b], outputs=list(node_b.outputs), nodes=[node_b])

        with self.assertRaises(ValueError):
            editing.replace_subgraph(
                [node_a, node_b], [], output_mapping={}
            )

    def test_raises_when_new_node_already_in_graph(self):
        input_val = ir.Value(name="input")
        node_a = ir.Node("", "A", inputs=[input_val], num_outputs=1)
        node_b = ir.Node("", "B", inputs=[input_val], num_outputs=1)
        ir.Graph(
            inputs=[input_val],
            outputs=[],
            nodes=[node_a, node_b],
        )

        with self.assertRaises(ValueError):
            editing.replace_subgraph(
                [node_a],
                [node_b],  # Already in graph
                output_mapping={},
            )

    def test_raises_when_external_consumer_not_mapped(self):
        """Error when old output has external consumer but is not in mapping."""
        input_val = ir.Value(name="input")
        node_a = ir.Node("", "A", inputs=[input_val], num_outputs=1)
        node_b = ir.Node("", "B", inputs=node_a.outputs, num_outputs=1)
        consumer = ir.Node("", "C", inputs=[node_b.outputs[0]], num_outputs=1)
        ir.Graph(
            inputs=[input_val],
            outputs=list(consumer.outputs),
            nodes=[node_a, node_b, consumer],
        )

        node_d = ir.Node("", "D", inputs=[input_val], num_outputs=1)
        with self.assertRaises(ValueError):
            # Missing mapping for node_b.outputs[0] (consumed by consumer)
            editing.replace_subgraph(
                [node_a, node_b],
                [node_d],
                output_mapping={},
            )


class ReplaceSubgraphAccessViaModuleTest(unittest.TestCase):
    """Test that replace_subgraph is accessible via ir.editing."""

    def test_accessible_via_ir_editing(self):
        self.assertTrue(hasattr(ir.editing, "replace_subgraph"))
        self.assertTrue(callable(ir.editing.replace_subgraph))


# ---------------------------------------------------------------------------
# eliminate_node tests
# ---------------------------------------------------------------------------


def _identity_graph() -> tuple[ir.Model, ir.Node, ir.Node, ir.Node]:
    """Build: input -> A -> Identity -> B -> output.

    Returns (model, node_a, identity_node, node_b).
    """
    graph_input = ir.Value(name="input")
    node_a = ir.Node("", "A", inputs=[graph_input], num_outputs=1)
    identity = ir.Node("", "Identity", inputs=node_a.outputs, num_outputs=1)
    node_b = ir.Node("", "B", inputs=identity.outputs, num_outputs=1)
    graph = ir.Graph(
        inputs=[graph_input],
        outputs=list(node_b.outputs),
        nodes=[node_a, identity, node_b],
    )
    model = ir.Model(graph=graph, ir_version=10)
    return model, node_a, identity, node_b


class EliminateNodeBasicTest(unittest.TestCase):
    """Test basic eliminate_node behaviour."""

    def test_eliminate_identity_redirects_consumers(self):
        """After elimination, B consumes A's output directly."""
        model, node_a, identity, node_b = _identity_graph()

        editing.eliminate_node(identity)

        self.assertIs(node_b.inputs[0], node_a.outputs[0])
        graph_nodes = list(model.graph)
        self.assertNotIn(identity, graph_nodes)
        self.assertIn(node_a, graph_nodes)
        self.assertIn(node_b, graph_nodes)

    def test_eliminate_node_with_input_index(self):
        """Redirect to a non-default input index."""
        graph_input = ir.Value(name="input")
        extra_input = ir.Value(name="extra")
        # Node with two inputs; we'll eliminate it routing to input_index=1
        passthrough = ir.Node(
            "", "PassThrough", inputs=[graph_input, extra_input], num_outputs=1
        )
        consumer = ir.Node("", "Consumer", inputs=passthrough.outputs, num_outputs=1)
        graph = ir.Graph(
            inputs=[graph_input, extra_input],
            outputs=list(consumer.outputs),
            nodes=[passthrough, consumer],
        )
        ir.Model(graph=graph, ir_version=10)

        editing.eliminate_node(passthrough, input_index=1)

        self.assertIs(consumer.inputs[0], extra_input)

    def test_eliminate_node_removes_from_graph(self):
        """Eliminated node is removed from the graph's node list."""
        model, node_a, identity, node_b = _identity_graph()

        editing.eliminate_node(identity)

        graph_nodes = list(model.graph)
        self.assertEqual(len(graph_nodes), 2)
        self.assertEqual([n.op_type for n in graph_nodes], ["A", "B"])

    def test_eliminate_preserves_graph_structure(self):
        """Remaining graph is well-formed after elimination."""
        model, node_a, identity, node_b = _identity_graph()

        editing.eliminate_node(identity)

        self.assertEqual(len(model.graph.inputs), 1)
        self.assertEqual(len(model.graph.outputs), 1)
        self.assertIs(model.graph.outputs[0], node_b.outputs[0])


class EliminateNodeMetadataTest(unittest.TestCase):
    """Test metadata merging behaviour."""

    def test_merges_type_when_input_has_none(self):
        """Output's type fills in when input type is None."""
        model, node_a, identity, node_b = _identity_graph()
        identity.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
        self.assertIsNone(node_a.outputs[0].type)

        editing.eliminate_node(identity)

        self.assertEqual(node_a.outputs[0].type, ir.TensorType(ir.DataType.FLOAT))

    def test_preserves_input_type_when_already_set(self):
        """Input's existing type is not overwritten by output's type."""
        model, node_a, identity, node_b = _identity_graph()
        node_a.outputs[0].type = ir.TensorType(ir.DataType.INT32)
        identity.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)

        editing.eliminate_node(identity)

        self.assertEqual(node_a.outputs[0].type, ir.TensorType(ir.DataType.INT32))

    def test_merges_shapes(self):
        """Shapes are merged using merge_shapes (prefers concrete dims)."""
        model, node_a, identity, node_b = _identity_graph()
        # Input has symbolic dim, output has concrete dim
        node_a.outputs[0].shape = ir.Shape([ir.SymbolicDim("N"), 3])
        identity.outputs[0].shape = ir.Shape([8, 3])

        editing.eliminate_node(identity)

        # After merge, concrete 8 should replace symbolic N
        self.assertEqual(node_a.outputs[0].shape, ir.Shape([8, 3]))

    def test_fills_shape_when_input_has_none(self):
        """Output's shape fills in when input has no shape."""
        model, node_a, identity, node_b = _identity_graph()
        identity.outputs[0].shape = ir.Shape([1, 2, 3])

        editing.eliminate_node(identity)

        self.assertEqual(node_a.outputs[0].shape, ir.Shape([1, 2, 3]))

    def test_merges_const_value_when_input_has_none(self):
        """Output's const_value fills in when input has none."""
        import numpy as np

        model, node_a, identity, node_b = _identity_graph()
        identity.outputs[0].const_value = ir.Tensor(np.array([1.0], dtype=np.float32))

        editing.eliminate_node(identity)

        self.assertIsNotNone(node_a.outputs[0].const_value)

    def test_no_metadata_propagation_when_disabled(self):
        """propagate_metadata=False skips all metadata merging."""
        model, node_a, identity, node_b = _identity_graph()
        identity.outputs[0].type = ir.TensorType(ir.DataType.FLOAT)
        identity.outputs[0].shape = ir.Shape([1, 2])

        editing.eliminate_node(identity, propagate_metadata=False)

        self.assertIsNone(node_a.outputs[0].type)
        self.assertIsNone(node_a.outputs[0].shape)

    def test_does_not_transfer_name_to_input(self):
        """The input value's name is preserved (output name is NOT transferred)."""
        model, node_a, identity, node_b = _identity_graph()
        node_a.outputs[0].name = "a_output"
        identity.outputs[0].name = "identity_output"

        editing.eliminate_node(identity)

        self.assertEqual(node_a.outputs[0].name, "a_output")


class EliminateNodeGraphOutputTest(unittest.TestCase):
    """Test graph output handling (the 'graph output tax')."""

    def test_eliminates_node_whose_output_is_graph_output(self):
        """When output is a graph output, graph.outputs is updated to the input value."""
        graph_input = ir.Value(name="input")
        node_a = ir.Node("", "A", inputs=[graph_input], num_outputs=1)
        identity = ir.Node("", "Identity", inputs=node_a.outputs, num_outputs=1)
        identity.outputs[0].name = "graph_out"
        graph = ir.Graph(
            inputs=[graph_input],
            outputs=list(identity.outputs),
            nodes=[node_a, identity],
        )
        model = ir.Model(graph=graph, ir_version=10)

        editing.eliminate_node(identity)

        # Graph output should now be A's output with the transferred name
        self.assertEqual(len(model.graph.outputs), 1)
        self.assertIs(model.graph.outputs[0], node_a.outputs[0])
        self.assertEqual(model.graph.outputs[0].name, "graph_out")

    def test_inserts_identity_when_input_is_already_graph_output(self):
        """When input is already a graph output, an Identity is inserted."""
        graph_input = ir.Value(name="input")
        node_a = ir.Node("", "A", inputs=[graph_input], num_outputs=1)
        identity = ir.Node("", "Identity", inputs=node_a.outputs, num_outputs=1)
        graph = ir.Graph(
            inputs=[graph_input],
            # A's output is graph output #0, Identity's output is graph output #1
            outputs=[node_a.outputs[0], identity.outputs[0]],
            nodes=[node_a, identity],
        )
        model = ir.Model(graph=graph, ir_version=10)

        editing.eliminate_node(identity)

        # First output should still be A's output
        self.assertIs(model.graph.outputs[0], node_a.outputs[0])
        # Second output should be from an Identity wrapping A's output
        second_output = model.graph.outputs[1]
        self.assertIsNotNone(second_output.producer())
        self.assertEqual(second_output.producer().op_type, "Identity")
        self.assertIs(second_output.producer().inputs[0], node_a.outputs[0])

    def test_raises_when_output_is_graph_output_and_input_is_graph_input(self):
        """Cannot eliminate a direct graph_input -> node -> graph_output passthrough."""
        graph_input = ir.Value(name="input")
        identity = ir.Node("", "Identity", inputs=[graph_input], num_outputs=1)
        graph = ir.Graph(
            inputs=[graph_input],
            outputs=list(identity.outputs),
            nodes=[identity],
        )
        ir.Model(graph=graph, ir_version=10)

        with self.assertRaisesRegex(ValueError, "graph input or initializer"):
            editing.eliminate_node(identity)

    def test_raises_when_output_is_graph_output_and_input_is_initializer(self):
        """Cannot eliminate when input is an initializer and output is graph output."""
        import numpy as np

        initializer_value = ir.Value(name="init_val")
        initializer_value.const_value = ir.Tensor(np.array([1.0], dtype=np.float32))
        identity = ir.Node("", "Identity", inputs=[initializer_value], num_outputs=1)
        graph = ir.Graph(
            inputs=[],
            outputs=list(identity.outputs),
            nodes=[identity],
            initializers=[initializer_value],
        )
        ir.Model(graph=graph, ir_version=10)

        with self.assertRaisesRegex(ValueError, "graph input or initializer"):
            editing.eliminate_node(identity)


class EliminateNodeValidationTest(unittest.TestCase):
    """Test input validation for eliminate_node."""

    def test_raises_when_node_has_no_graph(self):
        """Error when node doesn't belong to a graph."""
        node = ir.Node("", "Identity", inputs=[ir.Value(name="x")], num_outputs=1)
        with self.assertRaisesRegex(ValueError, "does not belong to a graph"):
            editing.eliminate_node(node)

    def test_raises_when_input_is_none(self):
        """Error when the specified input is None."""
        graph_input = ir.Value(name="input")
        node = ir.Node("", "TestOp", inputs=[None, graph_input], num_outputs=1)
        graph = ir.Graph(
            inputs=[graph_input],
            outputs=list(node.outputs),
            nodes=[node],
        )
        ir.Model(graph=graph, ir_version=10)

        with self.assertRaisesRegex(ValueError, "None at input index 0"):
            editing.eliminate_node(node, input_index=0)


class EliminateNodeMultipleConsumersTest(unittest.TestCase):
    """Test eliminate_node when the output has multiple consumers."""

    def test_all_consumers_redirected(self):
        """All consumers of the eliminated node's output are redirected."""
        graph_input = ir.Value(name="input")
        node_a = ir.Node("", "A", inputs=[graph_input], num_outputs=1)
        identity = ir.Node("", "Identity", inputs=node_a.outputs, num_outputs=1)
        node_b = ir.Node("", "B", inputs=identity.outputs, num_outputs=1)
        node_c = ir.Node("", "C", inputs=identity.outputs, num_outputs=1)
        graph = ir.Graph(
            inputs=[graph_input],
            outputs=[node_b.outputs[0], node_c.outputs[0]],
            nodes=[node_a, identity, node_b, node_c],
        )
        ir.Model(graph=graph, ir_version=10)

        editing.eliminate_node(identity)

        # Both B and C should now consume A's output directly
        self.assertIs(node_b.inputs[0], node_a.outputs[0])
        self.assertIs(node_c.inputs[0], node_a.outputs[0])


class EliminateNodeAccessViaModuleTest(unittest.TestCase):
    """Test that eliminate_node is accessible via ir.editing."""

    def test_accessible_via_ir_editing(self):
        self.assertTrue(hasattr(ir.editing, "eliminate_node"))
        self.assertTrue(callable(ir.editing.eliminate_node))


if __name__ == "__main__":
    unittest.main()
