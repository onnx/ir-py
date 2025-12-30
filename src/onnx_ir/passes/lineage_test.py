# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests for lineage tracking functionality."""

from __future__ import annotations

import onnx_ir as ir
from onnx_ir.passes import lineage


class TestLineageTracking:
    """Tests for the lineage tracking system."""

    def test_track_lineage_context_manager(self):
        """Test that the track_lineage context manager enables/disables tracking."""
        # Initially disabled
        assert not lineage._tracking_enabled

        # Enabled within context
        with lineage.track_lineage():
            assert lineage._tracking_enabled

        # Disabled after context
        assert not lineage._tracking_enabled

    def test_track_lineage_with_false(self):
        """Test that track_lineage can be explicitly disabled."""
        with lineage.track_lineage(enabled=False):
            assert not lineage._tracking_enabled

    def test_track_lineage_nested(self):
        """Test nested track_lineage contexts."""
        assert not lineage._tracking_enabled

        with lineage.track_lineage():
            assert lineage._tracking_enabled

            with lineage.track_lineage(enabled=False):
                assert not lineage._tracking_enabled

            assert lineage._tracking_enabled

        assert not lineage._tracking_enabled

    def test_tag_when_disabled_is_noop(self):
        """Test that tagging does nothing when tracking is disabled."""
        # Create a simple model
        graph = ir.Graph(
            inputs=[],
            outputs=[],
            nodes=[],
            opset_imports={"": 18},
        )
        model = ir.Model(graph, ir_version=8)

        # Add a node
        x = ir.Value(name="x")
        y = ir.Value(name="y")
        node = ir.Node("", "Identity", inputs=[x], outputs=[y], graph=graph)

        # Tag without enabling tracking
        lineage.tag(model, "test_pass")

        # Should not have lineage info
        assert lineage.LINEAGE_TAG_KEY not in node.metadata_props
        assert lineage.LINEAGE_STEP_KEY not in node.metadata_props

    def test_tag_new_nodes_get_current_step(self):
        """Test that new nodes are tagged with the current step from the counter."""
        graph = ir.Graph(
            inputs=[],
            outputs=[],
            nodes=[],
            opset_imports={"": 18},
        )
        model = ir.Model(graph, ir_version=8)

        x = ir.Value(name="x")
        y = ir.Value(name="y")
        node = ir.Node("", "Identity", inputs=[x], outputs=[y], graph=graph)

        with lineage.track_lineage():
            lineage.tag(model, "first_pass")

        # Counter should be "0" after first tag
        assert model.metadata_props[lineage.LINEAGE_COUNTER_KEY] == "0"
        assert node.metadata_props[lineage.LINEAGE_TAG_KEY] == "first_pass"
        assert node.metadata_props[lineage.LINEAGE_STEP_KEY] == "0"

    def test_tag_does_not_retag_existing_nodes(self):
        """Test that existing nodes are not re-tagged, only new ones."""
        graph = ir.Graph(
            inputs=[],
            outputs=[],
            nodes=[],
            opset_imports={"": 18},
        )
        model = ir.Model(graph, ir_version=8)

        x = ir.Value(name="x")
        y = ir.Value(name="y")
        node = ir.Node("", "Identity", inputs=[x], outputs=[y], graph=graph)

        with lineage.track_lineage():
            lineage.tag(model, "first_pass")
            assert model.metadata_props[lineage.LINEAGE_COUNTER_KEY] == "0"
            assert node.metadata_props[lineage.LINEAGE_STEP_KEY] == "0"

            lineage.tag(model, "second_pass")
            # Counter should increment
            assert model.metadata_props[lineage.LINEAGE_COUNTER_KEY] == "1"
            # But existing node should keep its original tag and step
            assert node.metadata_props[lineage.LINEAGE_TAG_KEY] == "first_pass"
            assert node.metadata_props[lineage.LINEAGE_STEP_KEY] == "0"

            lineage.tag(model, "third_pass")
            assert model.metadata_props[lineage.LINEAGE_COUNTER_KEY] == "2"
            # Still unchanged
            assert node.metadata_props[lineage.LINEAGE_TAG_KEY] == "first_pass"
            assert node.metadata_props[lineage.LINEAGE_STEP_KEY] == "0"

    def test_counter_increments_correctly(self):
        """Test that the global counter increments with each tag call."""
        graph = ir.Graph(
            inputs=[],
            outputs=[],
            nodes=[],
            opset_imports={"": 18},
        )
        model = ir.Model(graph, ir_version=8)

        with lineage.track_lineage():
            # Counter should not be set initially
            assert lineage.LINEAGE_COUNTER_KEY not in model.metadata_props

            # First tag sets it to "0"
            lineage.tag(model, "pass1")
            assert model.metadata_props[lineage.LINEAGE_COUNTER_KEY] == "0"

            # Subsequent tags increment
            lineage.tag(model, "pass2")
            assert model.metadata_props[lineage.LINEAGE_COUNTER_KEY] == "1"

            lineage.tag(model, "pass3")
            assert model.metadata_props[lineage.LINEAGE_COUNTER_KEY] == "2"

            lineage.tag(model, "pass4")
            assert model.metadata_props[lineage.LINEAGE_COUNTER_KEY] == "3"

    def test_tag_values_are_tracked(self):
        """Test that output values are also tagged."""
        graph = ir.Graph(
            inputs=[],
            outputs=[],
            nodes=[],
            opset_imports={"": 18},
        )
        model = ir.Model(graph, ir_version=8)

        x = ir.Value(name="x")
        y = ir.Value(name="y")
        node = ir.Node("", "Identity", inputs=[x], outputs=[y], graph=graph)

        with lineage.track_lineage():
            lineage.tag(model, "first_pass")

        # Output value should be tagged
        assert y.metadata_props[lineage.LINEAGE_TAG_KEY] == "first_pass"
        assert y.metadata_props[lineage.LINEAGE_STEP_KEY] == "0"

    def test_tag_graph_inputs_are_tracked(self):
        """Test that graph inputs are tagged."""
        x = ir.Value(name="x")
        graph = ir.Graph(
            inputs=[x],
            outputs=[],
            nodes=[],
            opset_imports={"": 18},
        )
        model = ir.Model(graph, ir_version=8)

        with lineage.track_lineage():
            lineage.tag(model, "first_pass")

        assert x.metadata_props[lineage.LINEAGE_TAG_KEY] == "first_pass"
        assert x.metadata_props[lineage.LINEAGE_STEP_KEY] == "0"

    def test_tag_initializers_are_tracked(self):
        """Test that initializers are tagged."""
        import numpy as np

        tensor = ir.Tensor(np.array([1.0, 2.0, 3.0]), name="weight")
        weight = ir.Value(name="weight", const_value=tensor)
        graph = ir.Graph(
            inputs=[],
            outputs=[],
            nodes=[],
            initializers=[weight],
            opset_imports={"": 18},
        )
        model = ir.Model(graph, ir_version=8)

        with lineage.track_lineage():
            lineage.tag(model, "first_pass")

        assert weight.metadata_props[lineage.LINEAGE_TAG_KEY] == "first_pass"
        assert weight.metadata_props[lineage.LINEAGE_STEP_KEY] == "0"

    def test_tag_mixed_new_and_existing_nodes(self):
        """Test tagging when some nodes are new and some existed before."""
        graph = ir.Graph(
            inputs=[],
            outputs=[],
            nodes=[],
            opset_imports={"": 18},
        )
        model = ir.Model(graph, ir_version=8)

        # Create initial node
        x = ir.Value(name="x")
        y = ir.Value(name="y")
        node1 = ir.Node("", "Identity", inputs=[x], outputs=[y], graph=graph)

        with lineage.track_lineage():
            # Tag the initial node
            lineage.tag(model, "first_pass")
            assert model.metadata_props[lineage.LINEAGE_COUNTER_KEY] == "0"
            assert node1.metadata_props[lineage.LINEAGE_STEP_KEY] == "0"

            # Add a new node
            z = ir.Value(name="z")
            node2 = ir.Node("", "Identity", inputs=[y], outputs=[z], graph=graph)

            # Tag again
            lineage.tag(model, "second_pass")
            assert model.metadata_props[lineage.LINEAGE_COUNTER_KEY] == "1"

            # Old node should keep its original step
            assert node1.metadata_props[lineage.LINEAGE_TAG_KEY] == "first_pass"
            assert node1.metadata_props[lineage.LINEAGE_STEP_KEY] == "0"

            # New node should have current step with new tag
            assert node2.metadata_props[lineage.LINEAGE_TAG_KEY] == "second_pass"
            assert node2.metadata_props[lineage.LINEAGE_STEP_KEY] == "1"

    def test_tag_same_tag_multiple_times(self):
        """Test calling tag with the same tag name multiple times."""
        graph = ir.Graph(
            inputs=[],
            outputs=[],
            nodes=[],
            opset_imports={"": 18},
        )
        model = ir.Model(graph, ir_version=8)

        x = ir.Value(name="x")
        y = ir.Value(name="y")
        node = ir.Node("", "Identity", inputs=[x], outputs=[y], graph=graph)

        with lineage.track_lineage():
            lineage.tag(model, "my_pass")
            assert model.metadata_props[lineage.LINEAGE_COUNTER_KEY] == "0"
            assert node.metadata_props[lineage.LINEAGE_TAG_KEY] == "my_pass"
            assert node.metadata_props[lineage.LINEAGE_STEP_KEY] == "0"

            lineage.tag(model, "my_pass")
            # Counter increments but node stays the same
            assert model.metadata_props[lineage.LINEAGE_COUNTER_KEY] == "1"
            assert node.metadata_props[lineage.LINEAGE_TAG_KEY] == "my_pass"
            assert node.metadata_props[lineage.LINEAGE_STEP_KEY] == "0"

    def test_tag_with_subgraphs(self):
        """Test that tagging works with subgraphs."""
        # Create a subgraph
        sub_x = ir.Value(name="sub_x")
        sub_y = ir.Value(name="sub_y")
        subgraph = ir.Graph(
            inputs=[sub_x],
            outputs=[sub_y],
            nodes=[],
            opset_imports={"": 18},
        )
        sub_node = ir.Node("", "Identity", inputs=[sub_x], outputs=[sub_y], graph=subgraph)

        # Create main graph with If node containing subgraph
        cond = ir.Value(name="cond")
        result = ir.Value(name="result")
        if_node = ir.Node(
            "",
            "If",
            inputs=[cond],
            outputs=[result],
            attributes=[
                ir.AttrGraph("then_branch", subgraph),
                ir.AttrGraph("else_branch", subgraph),
            ],
        )
        main_graph = ir.Graph(
            inputs=[cond],
            outputs=[result],
            nodes=[if_node],
            opset_imports={"": 18},
        )
        model = ir.Model(main_graph, ir_version=8)

        with lineage.track_lineage():
            lineage.tag(model, "test_pass")

        # Main graph node should be tagged
        assert if_node.metadata_props[lineage.LINEAGE_TAG_KEY] == "test_pass"
        assert if_node.metadata_props[lineage.LINEAGE_STEP_KEY] == "0"

        # Subgraph node should be tagged
        assert sub_node.metadata_props[lineage.LINEAGE_TAG_KEY] == "test_pass"
        assert sub_node.metadata_props[lineage.LINEAGE_STEP_KEY] == "0"

    def test_tag_preserves_other_metadata(self):
        """Test that tagging doesn't affect other metadata."""
        graph = ir.Graph(
            inputs=[],
            outputs=[],
            nodes=[],
            opset_imports={"": 18},
        )
        model = ir.Model(graph, ir_version=8)

        x = ir.Value(name="x")
        y = ir.Value(name="y")
        node = ir.Node("", "Identity", inputs=[x], outputs=[y], graph=graph)

        # Add some custom metadata
        node.metadata_props["custom_key"] = "custom_value"

        with lineage.track_lineage():
            lineage.tag(model, "test_pass")

        # Custom metadata should be preserved
        assert node.metadata_props["custom_key"] == "custom_value"
        # Lineage metadata should be added
        assert node.metadata_props[lineage.LINEAGE_TAG_KEY] == "test_pass"
        assert node.metadata_props[lineage.LINEAGE_STEP_KEY] == "0"

    def test_multiple_outputs_are_all_tagged(self):
        """Test that all output values of a node are tagged."""
        graph = ir.Graph(
            inputs=[],
            outputs=[],
            nodes=[],
            opset_imports={"": 18},
        )
        model = ir.Model(graph, ir_version=8)

        x = ir.Value(name="x")
        y1 = ir.Value(name="y1")
        y2 = ir.Value(name="y2")
        # Create a node with multiple outputs (like Split)
        node = ir.Node("", "Split", inputs=[x], outputs=[y1, y2], graph=graph)

        with lineage.track_lineage():
            lineage.tag(model, "test_pass")

        # All outputs should be tagged
        assert y1.metadata_props[lineage.LINEAGE_TAG_KEY] == "test_pass"
        assert y1.metadata_props[lineage.LINEAGE_STEP_KEY] == "0"
        assert y2.metadata_props[lineage.LINEAGE_TAG_KEY] == "test_pass"
        assert y2.metadata_props[lineage.LINEAGE_STEP_KEY] == "0"

    def test_lineage_workflow_simulation(self):
        """Simulate a realistic workflow with multiple passes."""
        graph = ir.Graph(
            inputs=[],
            outputs=[],
            nodes=[],
            opset_imports={"": 18},
        )
        model = ir.Model(graph, ir_version=8)

        # Initial model setup
        x = ir.Value(name="x")
        y = ir.Value(name="y")
        node1 = ir.Node("", "Relu", inputs=[x], outputs=[y], graph=graph)

        with lineage.track_lineage():
            # Initial tagging
            lineage.tag(model, "load")
            assert model.metadata_props[lineage.LINEAGE_COUNTER_KEY] == "0"
            assert node1.metadata_props[lineage.LINEAGE_STEP_KEY] == "0"

            # First pass: constant folding (no changes)
            lineage.tag(model, "constant_folding")
            assert model.metadata_props[lineage.LINEAGE_COUNTER_KEY] == "1"
            # Existing node unchanged
            assert node1.metadata_props[lineage.LINEAGE_TAG_KEY] == "load"
            assert node1.metadata_props[lineage.LINEAGE_STEP_KEY] == "0"

            # Second pass: fusion (creates new node)
            z = ir.Value(name="z")
            node2 = ir.Node("", "Add", inputs=[y, y], outputs=[z], graph=graph)
            lineage.tag(model, "fusion")
            assert model.metadata_props[lineage.LINEAGE_COUNTER_KEY] == "2"

            # Old node still has its original values
            assert node1.metadata_props[lineage.LINEAGE_TAG_KEY] == "load"
            assert node1.metadata_props[lineage.LINEAGE_STEP_KEY] == "0"

            # New node has current step with fusion tag
            assert node2.metadata_props[lineage.LINEAGE_TAG_KEY] == "fusion"
            assert node2.metadata_props[lineage.LINEAGE_STEP_KEY] == "2"

            # Third pass: optimization (no changes)
            lineage.tag(model, "optimize")
            assert model.metadata_props[lineage.LINEAGE_COUNTER_KEY] == "3"

            # Both nodes keep their original steps
            assert node1.metadata_props[lineage.LINEAGE_STEP_KEY] == "0"
            assert node2.metadata_props[lineage.LINEAGE_STEP_KEY] == "2"
