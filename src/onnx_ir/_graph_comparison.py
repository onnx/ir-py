# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Utilities for comparing IR graphs."""

from __future__ import annotations

import numpy as np

from onnx_ir import _core, _enums

# NOTE(justinchuby): We need to ensure a graph has valid inputs and outputs
# NOTE(justinchuby): A graph may be specified with a set of inputs and outputs


def topologically_equal(
    graph1: _core.Graph, graph2: _core.Graph, *, tensor_size_limit: int | None = None
) -> bool:
    """Return true if the two graphs are topologically equivalent.

    Two graphs are topologically equivalent if they have the same structure:
    - Same number of nodes with matching operations and domains
    - Same connectivity pattern between nodes
    - Same number of inputs and outputs
    - Matching node attributes (including values for scalar and list types)
    - Tensor attributes and initializers are always compared by shape and dtype
    - Tensor data is compared only if size is within the limit

    The comparison is done by building a mapping between nodes of both graphs
    based on their topological position and verifying that corresponding nodes
    have matching properties.

    Tensor attributes and initializers are always compared by their shape and dtype.
    If tensor_size_limit is specified, tensor data is also compared for tensors
    with size not exceeding the limit.

    Args:
        graph1: The first graph to compare.
        graph2: The second graph to compare.
        tensor_size_limit: Maximum size of tensors to compare data values.
            If None (default), always compare tensor data.
            If an integer, only compare data for tensors with total size <= limit.

    Returns:
        True if the graphs are topologically equal, False otherwise.
    """
    differences: list[str] = []
    result = _compare_graphs(graph1, graph2, tensor_size_limit, differences)
    return result


def assert_topologically_equal(
    graph1: _core.Graph, graph2: _core.Graph, *, tensor_size_limit: int | None = None
) -> None:
    """Assert that two graphs are topologically equivalent.

    Raises an AssertionError with detailed messages if the graphs are not equal.
    If only tensor data differs but topology is the same, continues checking
    and reports all differences together.

    Args:
        graph1: The first graph to compare.
        graph2: The second graph to compare.
        tensor_size_limit: Maximum size of tensors to compare data values.
            If None (default), always compare tensor data.
            If an integer, only compare data for tensors with total size <= limit.

    Raises:
        AssertionError: If the graphs are not topologically equal, with detailed
            error messages describing the differences.
    """
    differences: list[str] = []
    result = _compare_graphs(graph1, graph2, tensor_size_limit, differences)

    if not result or differences:
        error_messages = []

        # Separate topological differences from tensor data differences
        topological_diffs = [d for d in differences if not d.startswith("Tensor data")]
        tensor_data_diffs = [d for d in differences if d.startswith("Tensor data")]

        if topological_diffs:
            error_messages.append("Topological differences found:")
            for diff in topological_diffs:
                error_messages.append(f"  - {diff}")

        if tensor_data_diffs:
            error_messages.append("Tensor data differences found:")
            for diff in tensor_data_diffs:
                error_messages.append(f"  - {diff}")

        raise AssertionError("\n".join(error_messages))


def _should_compare_tensor_data(tensor, tensor_size_limit: int | None) -> bool:
    """Determine if tensor data should be compared based on size limit."""
    if tensor_size_limit is None:
        return True
    if tensor is None:
        return False
    # Calculate total size
    size = np.prod(tensor.shape) if hasattr(tensor, "shape") else 0
    return size <= tensor_size_limit


def _tensors_equal(tensor1, tensor2, tensor_size_limit: int | None) -> bool:
    """Compare two tensors for equality."""
    # Always compare shape and dtype
    if tensor1.shape != tensor2.shape:
        return False
    if tensor1.dtype != tensor2.dtype:
        return False

    # Compare data if within size limit
    if _should_compare_tensor_data(tensor1, tensor_size_limit):
        try:
            return np.array_equal(tensor1.numpy(), tensor2.numpy(), equal_nan=True)
        except Exception:
            # If comparison fails, consider them not equal
            return False

    return True


def _compare_graphs(
    graph1: _core.Graph,
    graph2: _core.Graph,
    tensor_size_limit: int | None,
    differences: list[str],
) -> bool:
    """Internal function to compare two graphs and collect differences."""
    topologically_equal = True

    # Queue to store subgraphs to compare: (graph1, graph2, context_string)
    subgraph_queue = []

    # Quick checks: number of nodes, inputs, outputs
    if len(list(graph1)) != len(list(graph2)):
        differences.append(
            f"Different number of nodes: {len(list(graph1))} vs {len(list(graph2))}"
        )
        return False

    if len(graph1.inputs) != len(graph2.inputs):
        differences.append(
            f"Different number of inputs: {len(graph1.inputs)} vs {len(graph2.inputs)}"
        )
        return False

    if len(graph1.outputs) != len(graph2.outputs):
        differences.append(
            f"Different number of outputs: {len(graph1.outputs)} vs {len(graph2.outputs)}"
        )
        return False

    # If both graphs are empty, they are equal
    nodes1 = list(graph1)
    nodes2 = list(graph2)
    if not nodes1 and not nodes2:
        return True

    # Build a mapping from graph1 values to graph2 values
    value_map: dict[_core.Value | None, _core.Value | None] = {None: None}

    # Map graph inputs
    value_map.update(zip(graph1.inputs, graph2.inputs))

    # Traverse both graphs in parallel and compare nodes
    for node_idx, (node1, node2) in enumerate(zip(nodes1, nodes2)):
        # Compare node properties
        if node1.domain != node2.domain:
            differences.append(
                f"Node {node_idx}: Different domain: '{node1.domain}' vs '{node2.domain}'"
            )
            return False

        if node1.op_type != node2.op_type:
            differences.append(
                f"Node {node_idx}: Different op_type: '{node1.op_type}' vs '{node2.op_type}'"
            )
            return False

        if node1.overload != node2.overload:
            differences.append(
                f"Node {node_idx}: Different overload: '{node1.overload}' vs '{node2.overload}'"
            )
            return False

        # Check number of inputs and outputs
        if len(node1.inputs) != len(node2.inputs):
            differences.append(
                f"Node {node_idx} ({node1.op_type}): Different number of inputs: "
                f"{len(node1.inputs)} vs {len(node2.inputs)}"
            )
            return False

        if len(node1.outputs) != len(node2.outputs):
            differences.append(
                f"Node {node_idx} ({node1.op_type}): Different number of outputs: "
                f"{len(node1.outputs)} vs {len(node2.outputs)}"
            )
            return False

        # Check if inputs match according to our mapping
        for input_idx, (inp1, inp2) in enumerate(zip(node1.inputs, node2.inputs)):
            if inp1 is None and inp2 is None:
                continue
            if inp1 is None or inp2 is None:
                differences.append(
                    f"Node {node_idx} ({node1.op_type}), input {input_idx}: "
                    f"One input is None, the other is not"
                )
                return False

            # If inp1 is already mapped, verify the mapping matches
            if inp1 in value_map:
                if value_map[inp1] != inp2:
                    differences.append(
                        f"Node {node_idx} ({node1.op_type}), input {input_idx}: "
                        f"Value mapping mismatch"
                    )
                    return False
            else:
                # If not mapped yet, it should be an initializer
                # Map it dynamically based on usage (not by name)
                if not (inp1.is_initializer() and inp2.is_initializer()):
                    differences.append(
                        f"Node {node_idx} ({node1.op_type}), input {input_idx}: "
                        f"One is initializer, the other is not"
                    )
                    return False

                # Always compare shapes and dtypes of initializers
                if inp1.const_value is None or inp2.const_value is None:
                    if inp1.const_value != inp2.const_value:
                        differences.append(
                            f"Node {node_idx} ({node1.op_type}), input {input_idx}: "
                            f"Initializer const_value mismatch"
                        )
                        return False
                else:
                    if inp1.const_value.shape != inp2.const_value.shape:
                        differences.append(
                            f"Node {node_idx} ({node1.op_type}), input {input_idx}: "
                            f"Initializer shape mismatch: {inp1.const_value.shape} vs {inp2.const_value.shape}"
                        )
                        return False
                    if inp1.const_value.dtype != inp2.const_value.dtype:
                        differences.append(
                            f"Node {node_idx} ({node1.op_type}), input {input_idx}: "
                            f"Initializer dtype mismatch: {inp1.const_value.dtype} vs {inp2.const_value.dtype}"
                        )
                        return False

                    # Compare tensor data if within size limit
                    if _should_compare_tensor_data(inp1.const_value, tensor_size_limit):
                        if not _tensors_equal(
                            inp1.const_value, inp2.const_value, tensor_size_limit
                        ):
                            # This is a tensor data difference, not topological
                            differences.append(
                                f"Tensor data difference: Node {node_idx} ({node1.op_type}), "
                                f"input {input_idx} initializer data differs"
                            )
                            topologically_equal = (
                                False  # Mark that we found a difference but continue
                            )

                # Map this initializer dynamically based on its usage position
                value_map[inp1] = inp2

        # Map outputs
        value_map.update(zip(node1.outputs, node2.outputs))

        # Compare attributes
        if len(node1.attributes) != len(node2.attributes):
            differences.append(
                f"Node {node_idx} ({node1.op_type}): Different number of attributes: "
                f"{len(node1.attributes)} vs {len(node2.attributes)}"
            )
            return False

        # Check attribute names and types match
        if set(node1.attributes.keys()) != set(node2.attributes.keys()):
            differences.append(
                f"Node {node_idx} ({node1.op_type}): Different attribute names: "
                f"{set(node1.attributes.keys())} vs {set(node2.attributes.keys())}"
            )
            return False

        for attr_name in node1.attributes:
            attr1 = node1.attributes[attr_name]
            attr2 = node2.attributes[attr_name]

            if attr1.type != attr2.type:
                differences.append(
                    f"Node {node_idx} ({node1.op_type}), attribute '{attr_name}': "
                    f"Different types: {attr1.type} vs {attr2.type}"
                )
                return False

            # Compare attribute values
            # For graph attributes, queue for later comparison
            if attr1.type == _enums.AttributeType.GRAPH:
                context = (
                    f"Node {node_idx} ({node1.op_type}), attribute '{attr_name}' subgraph"
                )
                subgraph_queue.append((attr1.value, attr2.value, context))
            elif attr1.type == _enums.AttributeType.GRAPHS:
                if len(attr1.value) != len(attr2.value):
                    differences.append(
                        f"Node {node_idx} ({node1.op_type}), attribute '{attr_name}': "
                        f"Different number of subgraphs: {len(attr1.value)} vs {len(attr2.value)}"
                    )
                    return False
                for g_idx, (g1, g2) in enumerate(zip(attr1.value, attr2.value)):
                    context = f"Node {node_idx} ({node1.op_type}), attribute '{attr_name}' subgraph {g_idx}"
                    subgraph_queue.append((g1, g2, context))
            elif attr1.type in (
                _enums.AttributeType.TENSOR,
                _enums.AttributeType.SPARSE_TENSOR,
            ):
                # For tensor attributes, always compare shapes and dtypes
                if attr1.value.shape != attr2.value.shape:
                    differences.append(
                        f"Node {node_idx} ({node1.op_type}), attribute '{attr_name}': "
                        f"Tensor shape mismatch: {attr1.value.shape} vs {attr2.value.shape}"
                    )
                    return False
                if attr1.value.dtype != attr2.value.dtype:
                    differences.append(
                        f"Node {node_idx} ({node1.op_type}), attribute '{attr_name}': "
                        f"Tensor dtype mismatch: {attr1.value.dtype} vs {attr2.value.dtype}"
                    )
                    return False

                # Compare tensor data if within size limit
                if _should_compare_tensor_data(attr1.value, tensor_size_limit):
                    if not _tensors_equal(attr1.value, attr2.value, tensor_size_limit):
                        differences.append(
                            f"Tensor data difference: Node {node_idx} ({node1.op_type}), "
                            f"attribute '{attr_name}' tensor data differs"
                        )
                        topologically_equal = False
            elif attr1.type in (
                _enums.AttributeType.TENSORS,
                _enums.AttributeType.SPARSE_TENSORS,
            ):
                # For tensor list attributes, always compare shapes and dtypes
                if len(attr1.value) != len(attr2.value):
                    differences.append(
                        f"Node {node_idx} ({node1.op_type}), attribute '{attr_name}': "
                        f"Different number of tensors: {len(attr1.value)} vs {len(attr2.value)}"
                    )
                    return False
                for t_idx, (t1, t2) in enumerate(zip(attr1.value, attr2.value)):
                    if t1.shape != t2.shape:
                        differences.append(
                            f"Node {node_idx} ({node1.op_type}), attribute '{attr_name}' tensor {t_idx}: "
                            f"Shape mismatch: {t1.shape} vs {t2.shape}"
                        )
                        return False
                    if t1.dtype != t2.dtype:
                        differences.append(
                            f"Node {node_idx} ({node1.op_type}), attribute '{attr_name}' tensor {t_idx}: "
                            f"Dtype mismatch: {t1.dtype} vs {t2.dtype}"
                        )
                        return False

                    # Compare tensor data if within size limit
                    if _should_compare_tensor_data(t1, tensor_size_limit):
                        if not _tensors_equal(t1, t2, tensor_size_limit):
                            differences.append(
                                f"Tensor data difference: Node {node_idx} ({node1.op_type}), "
                                f"attribute '{attr_name}' tensor {t_idx} data differs"
                            )
                            topologically_equal = False
            else:
                # For scalar and list attributes (INT, FLOAT, STRING, INTS, FLOATS, STRINGS, TYPE_PROTO, TYPE_PROTOS)
                # Compare values directly
                if attr1.value != attr2.value:
                    differences.append(
                        f"Node {node_idx} ({node1.op_type}), attribute '{attr_name}': "
                        f"Value mismatch: {attr1.value} vs {attr2.value}"
                    )
                    return False

    # Verify that graph outputs are properly mapped
    for out_idx, (out1, out2) in enumerate(zip(graph1.outputs, graph2.outputs)):
        if out1 not in value_map or value_map[out1] != out2:
            differences.append(f"Graph output {out_idx}: Mapping mismatch")
            return False

    # Now compare all queued subgraphs
    for subgraph1, subgraph2, context in subgraph_queue:
        sub_diffs: list[str] = []
        sub_result = _compare_graphs(subgraph1, subgraph2, tensor_size_limit, sub_diffs)
        if not sub_result:
            # Topological difference in subgraph
            differences.extend([f"{context}: {d}" for d in sub_diffs])
            return False
        elif sub_diffs:
            # Only tensor data differences in subgraph
            differences.extend([f"{context}: {d}" for d in sub_diffs])
            topologically_equal = False

    return topologically_equal
