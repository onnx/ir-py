# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Utilities for comparing IR graphs."""

from __future__ import annotations

import math

import numpy as np

from onnx_ir import _core, _enums, _protocols


def topologically_equal(
    graph1: _core.Graph, graph2: _core.Graph, *, tensor_size_limit: int | None = None
) -> bool:
    """Return true if the two graphs are topologically equivalent.

    Two graphs are topologically equivalent if they have the same structure:
    - Same number of inputs and outputs
    - Corresponding outputs are produced by equivalent nodes
    - Node equivalence is checked recursively through their inputs
    - Matching node attributes (including values for scalar and list types)
    - Tensor attributes and initializers are always compared by shape and dtype
    - Tensor data is compared only if size is within the limit

    The comparison is done by starting from the outputs and working backwards,
    building a mapping between equivalent values in both graphs. This allows
    comparison of graphs with nodes in different orders as long as they are
    topologically equivalent.

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


def _should_compare_tensor_data(
    tensor: _protocols.TensorProtocol, tensor_size_limit: int | None
) -> bool:
    """Determine if tensor data should be compared based on size limit."""
    if tensor_size_limit is None:
        return True
    if tensor is None:
        return False
    # Calculate total size
    size = math.prod(tensor.shape.numpy())
    return size <= tensor_size_limit


def _tensors_not_equal(
    tensor1: _protocols.TensorProtocol,
    tensor2: _protocols.TensorProtocol,
    tensor_size_limit: int | None,
) -> str:
    """Compare two tensors for equality."""
    # Always compare shape and dtype
    if tensor1.shape != tensor2.shape:
        return f"tensor 1 shape {tensor1.shape} != tensor 2 shape {tensor2.shape}"
    if tensor1.dtype != tensor2.dtype:
        return f"tensor 1 dtype {tensor1.dtype} != tensor 2 dtype {tensor2.dtype}"

    # Compare data if within size limit
    if _should_compare_tensor_data(tensor1, tensor_size_limit):
        try:
            np.testing.assert_array_almost_equal(tensor1.numpy(), tensor2.numpy())
        except Exception as e:
            # If comparison fails, consider them not equal
            return str(e)

    return ""


def _compare_graphs(
    graph1: _core.Graph,
    graph2: _core.Graph,
    tensor_size_limit: int | None,
    differences: list[str],
) -> bool:
    """Internal function to compare two graphs and collect differences.

    Uses backward traversal from outputs to compare graphs topologically.
    """
    are_equal = True

    # Quick checks: number of inputs, outputs
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

    # Mapping from values in graph1 to values in graph2
    # Also tracks which values have been compared
    value_map: dict[_core.Value | None, _core.Value | None] = {None: None}

    # Track which nodes have been visited during comparison
    visited_nodes1: set[_core.Node] = set()
    visited_nodes2: set[_core.Node] = set()

    # Map graph inputs first - also check type and shape if known
    for inp1, inp2 in zip(graph1.inputs, graph2.inputs):
        value_map[inp1] = inp2
        # Check type and shape match for inputs
        if inp1.type is not None and inp2.type is not None:
            if inp1.type != inp2.type:
                differences.append(
                    f"Input type mismatch: {inp1.name} has type {inp1.type}, "
                    f"{inp2.name} has type {inp2.type}"
                )
                return False
        if inp1.shape is not None and inp2.shape is not None:
            if inp1.shape != inp2.shape:
                differences.append(
                    f"Input shape mismatch: {inp1.name} has shape {inp1.shape}, "
                    f"{inp2.name} has shape {inp2.shape}"
                )
                return False

    # Queue for values to compare (backward traversal from outputs)
    # Contains tuples of (value1, value2, context_string)
    value_queue = []

    # Start with graph outputs
    for out_idx, (out1, out2) in enumerate(zip(graph1.outputs, graph2.outputs)):
        value_queue.append((out1, out2, f"output {out_idx}"))

    # Queue for subgraphs to compare
    subgraph_queue = []

    # Process values backward from outputs
    while value_queue:
        val1, val2, context = value_queue.pop(0)

        # Check if already compared
        if val1 in value_map:
            if value_map[val1] != val2:
                differences.append(
                    f"{context}: Value mapping conflict - "
                    f"val1 already mapped to different val2"
                )
                return False
            continue  # Already compared this value pair

        # Map this value pair
        value_map[val1] = val2

        # Check type and shape match if known
        if val1.type is not None and val2.type is not None:
            if val1.type != val2.type:
                differences.append(
                    f"{context}: Type mismatch - {val1.name} has type {val1.type}, "
                    f"{val2.name} has type {val2.type}"
                )
                return False
        if val1.shape is not None and val2.shape is not None:
            if val1.shape != val2.shape:
                differences.append(
                    f"{context}: Shape mismatch - {val1.name} has shape {val1.shape}, "
                    f"{val2.name} has shape {val2.shape}"
                )
                return False

        # Check if values are graph inputs (base case)
        val1_is_input = val1 in graph1.inputs
        val2_is_input = val2 in graph2.inputs

        if val1_is_input != val2_is_input:
            differences.append(
                f"{context}: One value is a graph input, the other is not "
                f"({val1.name} vs {val2.name})"
            )
            return False

        if val1_is_input:
            # Both are inputs, already mapped
            continue

        # Check if values are initializers
        if val1.is_initializer() and val2.is_initializer():
            # Compare initializer properties
            # Initializers should always have const_value available
            if comp_result := _tensors_not_equal(
                val1.const_value,  # type: ignore[arg-type]
                val2.const_value,  # type: ignore[arg-type]
                tensor_size_limit,
            ):
                # If the result explicitly contains "shape" or "dtype", it's a topological difference
                # Otherwise it's a data mismatch (from numpy comparison)
                if "tensor 1 shape" in comp_result:
                    # Shape mismatch is topological
                    differences.append(f"{context}: Initializer shape mismatch: {comp_result}")
                    return False
                elif "tensor 1 dtype" in comp_result:
                    # Dtype mismatch is topological
                    differences.append(f"{context}: Initializer dtype mismatch: {comp_result}")
                    return False
                else:
                    # Data mismatch only
                    differences.append(
                        f"Tensor data difference: {context}: initializer data differs: {comp_result}"
                    )
                    are_equal = False
            continue
        elif val1.is_initializer() != val2.is_initializer():
            differences.append(
                f"{context}: One value is initializer, the other is not "
                f"({val1.name} vs {val2.name})"
            )
            return False

        # Values must be produced by nodes - get the producer nodes
        producer1 = val1.producer()
        producer2 = val2.producer()

        if producer1 is None or producer2 is None:
            if producer1 != producer2:
                differences.append(
                    f"{context}: One value has producer, the other doesn't "
                    f"({val1.name} vs {val2.name})"
                )
                return False
            continue

        # Check that output indices match
        if val1.index() != val2.index():
            differences.append(
                f"{context}: Output index mismatch - {val1.name} is output {val1.index()} "
                f"of its producer, {val2.name} is output {val2.index()} of its producer"
            )
            return False

        # Track visited nodes
        visited_nodes1.add(producer1)
        visited_nodes2.add(producer2)

        # Compare the producer nodes
        node1_desc = f"producer of {context} ({producer1.op_type}, name='{producer1.name}')"
        node2_desc = f"producer of {context} ({producer2.op_type}, name='{producer2.name}')"

        if producer1.domain != producer2.domain:
            differences.append(
                f"{node1_desc}: Different domain: '{producer1.domain}' vs '{producer2.domain}'"
            )
            return False

        if producer1.op_type != producer2.op_type:
            differences.append(
                f"{node1_desc} vs {node2_desc}: Different op_type: '{producer1.op_type}' vs '{producer2.op_type}'"
            )
            return False

        if producer1.overload != producer2.overload:
            differences.append(
                f"{node1_desc}: Different overload: '{producer1.overload}' vs '{producer2.overload}'"
            )
            return False

        # Compare number of inputs and outputs
        if len(producer1.inputs) != len(producer2.inputs):
            differences.append(
                f"{node1_desc}: Different number of inputs: "
                f"{len(producer1.inputs)} vs {len(producer2.inputs)}"
            )
            return False

        if len(producer1.outputs) != len(producer2.outputs):
            differences.append(
                f"{node1_desc}: Different number of outputs: "
                f"{len(producer1.outputs)} vs {len(producer2.outputs)}"
            )
            return False

        # Queue node inputs for comparison
        for input_idx, (inp1, inp2) in enumerate(zip(producer1.inputs, producer2.inputs)):  # type: ignore[assignment]
            if inp1 is None and inp2 is None:
                continue
            if inp1 is None or inp2 is None:
                differences.append(
                    f"{node1_desc}, input {input_idx}: One input is None, the other is not"
                )
                return False

            # Add to queue for backward traversal
            input_context = f"{node1_desc}, input {input_idx}"
            value_queue.append((inp1, inp2, input_context))

        # Compare attributes
        if len(producer1.attributes) != len(producer2.attributes):
            differences.append(
                f"{node1_desc}: Different number of attributes: "
                f"{len(producer1.attributes)} vs {len(producer2.attributes)}"
            )
            return False

        if set(producer1.attributes.keys()) != set(producer2.attributes.keys()):
            differences.append(
                f"{node1_desc}: Different attribute names: "
                f"{set(producer1.attributes.keys())} vs {set(producer2.attributes.keys())}"
            )
            return False

        for attr_name in producer1.attributes:
            attr1 = producer1.attributes[attr_name]
            attr2 = producer2.attributes[attr_name]

            if attr1.type != attr2.type:
                differences.append(
                    f"{node1_desc}, attribute '{attr_name}': "
                    f"Different types: {attr1.type} vs {attr2.type}"
                )
                return False

            # Compare attribute values
            if attr1.type == _enums.AttributeType.GRAPH:
                context_str = f"{node1_desc}, attribute '{attr_name}' subgraph"
                subgraph_queue.append((attr1.value, attr2.value, context_str))
            elif attr1.type == _enums.AttributeType.GRAPHS:
                if len(attr1.value) != len(attr2.value):
                    differences.append(
                        f"{node1_desc}, attribute '{attr_name}': "
                        f"Different number of subgraphs: {len(attr1.value)} vs {len(attr2.value)}"
                    )
                    return False
                for g_idx, (g1, g2) in enumerate(zip(attr1.value, attr2.value)):
                    context_str = f"{node1_desc}, attribute '{attr_name}' subgraph {g_idx}"
                    subgraph_queue.append((g1, g2, context_str))
            elif attr1.type in (
                _enums.AttributeType.TENSOR,
                _enums.AttributeType.SPARSE_TENSOR,
            ):
                if comp_result := _tensors_not_equal(
                    attr1.value, attr2.value, tensor_size_limit
                ):
                    if "tensor 1 shape" in comp_result or "tensor 1 dtype" in comp_result:
                        if "tensor 1 shape" in comp_result:
                            differences.append(
                                f"{node1_desc}, attribute '{attr_name}': Tensor shape mismatch: {comp_result}"
                            )
                        else:
                            differences.append(
                                f"{node1_desc}, attribute '{attr_name}': Tensor dtype mismatch: {comp_result}"
                            )
                        return False
                    else:
                        differences.append(
                            f"Tensor data difference: {node1_desc}, "
                            f"attribute '{attr_name}' tensor data differs: {comp_result}"
                        )
                        are_equal = False
            elif attr1.type in (
                _enums.AttributeType.TENSORS,
                _enums.AttributeType.SPARSE_TENSORS,
            ):
                if len(attr1.value) != len(attr2.value):
                    differences.append(
                        f"{node1_desc}, attribute '{attr_name}': "
                        f"Different number of tensors: {len(attr1.value)} vs {len(attr2.value)}"
                    )
                    return False
                for t_idx, (t1, t2) in enumerate(zip(attr1.value, attr2.value)):
                    if comp_result := _tensors_not_equal(t1, t2, tensor_size_limit):
                        if "tensor 1 shape" in comp_result or "tensor 1 dtype" in comp_result:
                            differences.append(
                                f"{node1_desc}, attribute '{attr_name}' tensor {t_idx}: {comp_result}"
                            )
                            return False
                        else:
                            differences.append(
                                f"Tensor data difference: {node1_desc}, "
                                f"attribute '{attr_name}' tensor {t_idx} data differs: {comp_result}"
                            )
                            are_equal = False
            else:
                # For scalar and list attributes
                if attr1.value != attr2.value:
                    differences.append(
                        f"{node1_desc}, attribute '{attr_name}': "
                        f"Value mismatch: {attr1.value} vs {attr2.value}"
                    )
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
            are_equal = False

    # Check that all nodes have been visited and matched
    # This ensures no unused nodes exist in either graph
    unvisited_nodes1 = set(graph1) - visited_nodes1
    unvisited_nodes2 = set(graph2) - visited_nodes2

    if unvisited_nodes1 or unvisited_nodes2:
        error_parts = []
        if unvisited_nodes1:
            node_names1 = [f"{n.op_type}(name='{n.name}')" for n in unvisited_nodes1]
            error_parts.append(f"Graph 1 has unvisited nodes: {', '.join(node_names1)}")
        if unvisited_nodes2:
            node_names2 = [f"{n.op_type}(name='{n.name}')" for n in unvisited_nodes2]
            error_parts.append(f"Graph 2 has unvisited nodes: {', '.join(node_names2)}")
        differences.append(" | ".join(error_parts))
        return False

    return are_equal
