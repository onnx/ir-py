# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Utilities for comparing IR graphs."""

from __future__ import annotations

from onnx_ir import _core, _enums

# NOTE(justinchuby): We need to ensure a graph has valid inputs and outputs
# NOTE(justinchuby): A graph may be specified with a set of inputs and outputs


def topologically_equal(
    graph1: _core.Graph, graph2: _core.Graph, *, compare_initializers: bool = False
) -> bool:
    """Return true if the two graphs are topologically equivalent.

    Two graphs are topologically equivalent if they have the same structure:
    - Same number of nodes with matching operations and domains
    - Same connectivity pattern between nodes
    - Same number of inputs and outputs
    - Optionally, same initializers (controlled by compare_initializers parameter)

    The comparison is done by building a mapping between nodes of both graphs
    based on their topological position and verifying that corresponding nodes
    have matching properties.

    Args:
        graph1: The first graph to compare.
        graph2: The second graph to compare.
        compare_initializers: Whether to compare initializers. Defaults to False.

    Returns:
        True if the graphs are topologically equal, False otherwise.
    """
    # Quick checks: number of nodes, inputs, outputs
    if len(list(graph1)) != len(list(graph2)):
        return False

    if len(graph1.inputs) != len(graph2.inputs):
        return False

    if len(graph1.outputs) != len(graph2.outputs):
        return False

    # If both graphs are empty, they are equal
    nodes1 = list(graph1)
    nodes2 = list(graph2)
    if not nodes1 and not nodes2:
        return True

    # Build a mapping from graph1 values to graph2 values
    # Start by mapping inputs
    value_map: dict[_core.Value | None, _core.Value | None] = {None: None}

    # Map graph inputs
    value_map.update(zip(graph1.inputs, graph2.inputs))

    # When comparing initializers, verify they have the same count and properties
    # The actual mapping will be done dynamically as we encounter them in nodes
    if compare_initializers:
        if len(graph1.initializers) != len(graph2.initializers):
            return False

        # Sort initializers by name for consistent comparison
        init_values1 = sorted(graph1.initializers.values(), key=lambda v: v.name or "")
        init_values2 = sorted(graph2.initializers.values(), key=lambda v: v.name or "")

        for v1, v2 in zip(init_values1, init_values2):
            # Check if initializers have the same properties
            if v1.const_value is None or v2.const_value is None:
                if v1.const_value != v2.const_value:
                    return False
            else:
                # Compare shapes
                if v1.const_value.shape != v2.const_value.shape:
                    return False
                # Compare dtypes
                if v1.const_value.dtype != v2.const_value.dtype:
                    return False

            # Map the initializer for later use
            value_map[v1] = v2

    # Traverse both graphs in parallel and compare nodes
    for node1, node2 in zip(nodes1, nodes2):
        # Compare node properties
        if node1.domain != node2.domain:
            return False

        if node1.op_type != node2.op_type:
            return False

        if node1.overload != node2.overload:
            return False

        # Check number of inputs and outputs
        if len(node1.inputs) != len(node2.inputs):
            return False

        if len(node1.outputs) != len(node2.outputs):
            return False

        # Check if inputs match according to our mapping
        for inp1, inp2 in zip(node1.inputs, node2.inputs):
            if inp1 is None and inp2 is None:
                continue
            if inp1 is None or inp2 is None:
                return False

            # If inp1 is already mapped, verify the mapping matches
            if inp1 in value_map:
                if value_map[inp1] != inp2:
                    return False
            else:
                # If not mapped yet, it should be an initializer
                # Map it dynamically based on usage (not by name)
                # When not comparing initializers, we allow this dynamic mapping
                # When comparing initializers, they should have been pre-mapped
                if compare_initializers:
                    # If comparing initializers, they should already be mapped
                    return False

                # Check both are initializers
                if not (inp1.is_initializer() and inp2.is_initializer()):
                    # One is initializer, the other is not - graphs are different
                    return False

                # Map this initializer dynamically based on its usage position
                value_map[inp1] = inp2

        # Map outputs
        value_map.update(zip(node1.outputs, node2.outputs))

        # Compare attributes
        if len(node1.attributes) != len(node2.attributes):
            return False

        # Check attribute names and types match
        if set(node1.attributes.keys()) != set(node2.attributes.keys()):
            return False

        for attr_name in node1.attributes:
            attr1 = node1.attributes[attr_name]
            attr2 = node2.attributes[attr_name]

            if attr1.type != attr2.type:
                return False

            # Compare attribute values
            # For graph attributes, recursively compare
            if attr1.type == _enums.AttributeType.GRAPH:
                if not topologically_equal(
                    attr1.value, attr2.value, compare_initializers=compare_initializers
                ):
                    return False
            elif attr1.type == _enums.AttributeType.GRAPHS:
                if len(attr1.value) != len(attr2.value):
                    return False
                for g1, g2 in zip(attr1.value, attr2.value):
                    if not topologically_equal(
                        g1, g2, compare_initializers=compare_initializers
                    ):
                        return False
            elif attr1.type in (
                _enums.AttributeType.TENSOR,
                _enums.AttributeType.SPARSE_TENSOR,
            ):
                # For tensor attributes, compare shapes and dtypes, not values
                # (similar to initializers when not comparing them)
                if hasattr(attr1.value, "shape") and hasattr(attr2.value, "shape"):
                    if attr1.value.shape != attr2.value.shape:
                        return False
                if hasattr(attr1.value, "dtype") and hasattr(attr2.value, "dtype"):
                    if attr1.value.dtype != attr2.value.dtype:
                        return False
            elif attr1.type in (
                _enums.AttributeType.TENSORS,
                _enums.AttributeType.SPARSE_TENSORS,
            ):
                # For tensor list attributes
                if len(attr1.value) != len(attr2.value):
                    return False
                for t1, t2 in zip(attr1.value, attr2.value):
                    if hasattr(t1, "shape") and hasattr(t2, "shape"):
                        if t1.shape != t2.shape:
                            return False
                    if hasattr(t1, "dtype") and hasattr(t2, "dtype"):
                        if t1.dtype != t2.dtype:
                            return False
            else:
                # For scalar and list attributes (INT, FLOAT, STRING, INTS, FLOATS, STRINGS, TYPE_PROTO, TYPE_PROTOS)
                # Compare values directly
                if attr1.value != attr2.value:
                    return False

    # Verify that graph outputs are properly mapped
    for out1, out2 in zip(graph1.outputs, graph2.outputs):
        if out1 not in value_map or value_map[out1] != out2:
            return False

    return True
