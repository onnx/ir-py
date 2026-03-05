# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""High-level graph editing operations for the ONNX IR.

This module provides intent-level graph editing "verbs" that handle common
graph transformation patterns in a single call. They absorb complexity like
graph output handling, metadata propagation, and node lifecycle management
so that pass authors can focus on *what* to transform rather than *how*.

Key operations:

- :func:`replace_node`: Replace a node with another, rewiring all consumers.
- :func:`eliminate_node`: Remove a passthrough node (Identity, Cast, etc.),
  redirecting consumers to its input.
- :func:`insert_on_edge`: Insert a node between a value and all its consumers.
- :func:`replace_subgraph`: Replace multiple connected nodes with new nodes.

These functions compose naturally with the existing traversal utilities
(:mod:`onnx_ir.traversal`) and pass infrastructure (:mod:`onnx_ir.passes`).

Example::

    import onnx_ir as ir
    from onnx_ir import editing

    # Replace a Relu with a Gelu
    old_relu = ...  # some node in the graph
    new_gelu = ir.node("Gelu", inputs=old_relu.inputs)
    editing.replace_node(old_relu, new_gelu)

    # Eliminate an identity node
    editing.eliminate_node(identity_node)

.. note::
    All editing operations handle graph outputs automatically. Pass authors
    never need to check :meth:`Value.is_graph_output()` when using these APIs.
"""

from __future__ import annotations

__all__ = [
    "eliminate_node",
    "insert_on_edge",
    "replace_node",
    "replace_subgraph",
]

from collections.abc import Sequence

from onnx_ir import _core


def _propagate_value_metadata(
    old_value: _core.Value, new_value: _core.Value
) -> None:
    """Copy type, shape, const_value, and name from old to new (old wins when not None)."""
    if old_value.type is not None:
        new_value.type = old_value.type
    if old_value.shape is not None:
        new_value.shape = old_value.shape
    if old_value.const_value is not None:
        new_value.const_value = old_value.const_value
    if old_value.name is not None:
        new_value.name = old_value.name


def _handle_graph_output_replacement(
    graph: _core.Graph,
    old_value: _core.Value,
    new_value: _core.Value,
    insertion_point: _core.Node,
) -> None:
    """Replace old_value with new_value in graph outputs.

    If new_value is already a graph output or graph input, an Identity node is
    inserted to avoid creating an invalid duplicate in the output list.
    Otherwise the name is transferred and the output slot is updated directly.
    """
    for idx, graph_output in enumerate(graph.outputs):
        if graph_output is not old_value:
            continue
        if new_value.is_graph_output() or new_value.is_graph_input():
            # Must insert an Identity to avoid duplicate graph output / aliasing
            identity_node = _core.Node(
                "",
                "Identity",
                inputs=[new_value],
                num_outputs=1,
            )
            identity_out = identity_node.outputs[0]
            identity_out.name = old_value.name
            identity_out.type = old_value.type
            identity_out.shape = old_value.shape
            graph.outputs[idx] = identity_out
            graph.insert_before(insertion_point, identity_node)
        else:
            if old_value.name is not None:
                new_value.name = old_value.name
            graph.outputs[idx] = new_value


def replace_node(
    old_node: _core.Node,
    new_node: _core.Node,
    *,
    output_mapping: dict[_core.Value, _core.Value] | None = None,
    propagate_metadata: bool = True,
) -> None:
    """Replace a node in the graph with another node.

    Handles all rewiring automatically:

    1. Maps old output values to new output values (1:1 by position, or
       via explicit *output_mapping*).
    2. When *propagate_metadata* is ``True``, copies type, shape, name, and
       ``const_value`` from old outputs to new outputs (old takes precedence
       when not ``None``).
    3. Redirects all consumers of old outputs to use new outputs.
    4. Updates graph outputs if any old outputs were graph outputs.
    5. Inserts *new_node* at *old_node*'s position in the graph.
    6. Removes *old_node* from the graph.

    When the output counts of *old_node* and *new_node* match,
    outputs are mapped 1:1 by position and no *output_mapping* is needed.
    When they differ, an explicit *output_mapping* must be provided.

    Args:
        old_node: The node to replace. Must belong to a graph.
        new_node: The replacement node. Must not already belong to a graph.
        output_mapping: Explicit mapping from old output values to new output
            values. Required when the number of outputs differs between the
            old and new nodes. When ``None``, outputs are mapped 1:1 by
            position.
        propagate_metadata: If ``True`` (default), propagate type, shape,
            ``const_value``, and name from old outputs to new outputs.

    Raises:
        ValueError: If *old_node* does not belong to a graph.
        ValueError: If *new_node* already belongs to a graph.
        ValueError: If output counts differ and *output_mapping* is not provided.
        ValueError: If *output_mapping* does not cover all old outputs that
            still have consumers.

    Example::

        >>> import onnx_ir as ir
        >>> from onnx_ir import editing
        >>> # Replace a Relu with a Gelu (same number of outputs)
        >>> old_relu = ...  # existing node in graph
        >>> new_gelu = ir.node("Gelu", inputs=old_relu.inputs)
        >>> editing.replace_node(old_relu, new_gelu)
    """
    graph = old_node.graph
    if graph is None:
        raise ValueError(
            f"old_node {old_node!r} does not belong to a graph. "
            "It must be part of a graph to be replaced."
        )
    if new_node.graph is not None:
        raise ValueError(
            f"new_node {new_node!r} already belongs to a graph ({new_node.graph!r}). "
            "The replacement node must not belong to any graph."
        )

    # Build the value mapping (old_output -> new_output)
    if output_mapping is not None:
        mapping = output_mapping
    elif len(old_node.outputs) == len(new_node.outputs):
        mapping = dict(zip(old_node.outputs, new_node.outputs))
    else:
        raise ValueError(
            f"Output count mismatch: old_node has {len(old_node.outputs)} outputs, "
            f"new_node has {len(new_node.outputs)} outputs. "
            "Provide an explicit output_mapping."
        )

    # Validate that all old outputs with consumers or graph-output status are mapped
    for old_output in old_node.outputs:
        if old_output not in mapping:
            if old_output.uses() or old_output.is_graph_output():
                raise ValueError(
                    f"Old output {old_output!r} has consumers or is a graph output "
                    "but is not in output_mapping."
                )

    # 1-2. Propagate metadata and handle graph outputs before rewiring uses
    graph_output_values = frozenset(graph.outputs)
    for old_output, new_output in mapping.items():
        if propagate_metadata:
            _propagate_value_metadata(old_output, new_output)
        if old_output in graph_output_values:
            _handle_graph_output_replacement(graph, old_output, new_output, old_node)

    # 3. Redirect all consumer uses of old outputs to new outputs
    for old_output, new_output in mapping.items():
        if old_output.is_graph_output():
            old_output.replace_all_uses_with(new_output, replace_graph_outputs=True)
        else:
            old_output.replace_all_uses_with(new_output)

    # 4. Insert new_node at old_node's position (before old_node)
    graph.insert_before(old_node, new_node)

    # 5. Remove old_node (safe=True detaches inputs and validates)
    graph.remove(old_node, safe=True)


def eliminate_node(
    node: _core.Node,
    /,
    input_index: int = 0,
    *,
    propagate_metadata: bool = True,
) -> None:
    """Eliminate a passthrough node by redirecting output consumers to its input.

    This is the fundamental operation for identity-like elimination passes.
    It replaces all uses of the node's outputs with its input at the given
    index, handles graph output replacement and name/type/shape transfer,
    then removes the node from the graph.

    Specifically:

    1. When *propagate_metadata* is ``True``, merges shape and type
       information from the output value into the input value.
    2. Replaces all uses of ``node.outputs[0]`` with
       ``node.inputs[input_index]``.
    3. If the output is a graph output, updates the graph output list and
       transfers the output name to the input value.
    4. Removes the node from the graph in safe mode.

    Args:
        node: The node to eliminate. Must belong to a graph.
        input_index: Which input to redirect consumers to. Defaults to ``0``.
        propagate_metadata: If ``True`` (default), merge shape and type from
            the output into the input value.

    Raises:
        ValueError: If *node* does not belong to a graph.
        ValueError: If ``node.inputs[input_index]`` is ``None``.
        ValueError: If the node output is a graph output **and** the input
            is a graph input or initializer (this would create an invalid
            direct passthrough that must be preserved).

    Example::

        >>> import onnx_ir as ir
        >>> from onnx_ir import editing
        >>> # Eliminate an Identity node
        >>> editing.eliminate_node(identity_node)
        >>> # Eliminate a redundant Cast
        >>> editing.eliminate_node(cast_node, input_index=0)
    """
    graph = node.graph
    if graph is None:
        raise ValueError(
            f"Node {node!r} does not belong to a graph. "
            "It must be part of a graph to be eliminated."
        )

    input_value = node.inputs[input_index]
    if input_value is None:
        raise ValueError(
            f"Node {node!r} has None at input index {input_index}. "
            "Cannot redirect consumers to a missing input."
        )

    output_value = node.outputs[0]

    # A graph_input/initializer → passthrough → graph_output path cannot
    # be eliminated because removing the node would create a direct
    # graph_input → graph_output alias, which is an invalid ONNX construct.
    if output_value.is_graph_output() and (
        input_value.is_graph_input() or input_value.is_initializer()
    ):
        raise ValueError(
            f"Cannot eliminate {node!r}: its output is a graph output and "
            f"its input {input_value!r} is a graph input or initializer. "
            "Removing this node would create an invalid direct passthrough. "
            "The caller should skip this node."
        )

    # 1. Merge metadata from the output into the surviving input value.
    #    Shape uses merge_shapes for correctness (prefers concrete dims).
    #    Type and const_value fill in gaps (input takes precedence).
    if propagate_metadata:
        input_value.merge_shapes(output_value.shape)
        if input_value.type is None:
            input_value.type = output_value.type
        if input_value.const_value is None:
            input_value.const_value = output_value.const_value

    # 2–3. Handle graph output replacement (transfers name, inserts Identity
    #       if the input is already a graph output) then redirect all uses.
    if output_value.is_graph_output():
        _handle_graph_output_replacement(graph, output_value, input_value, node)
        output_value.replace_all_uses_with(input_value, replace_graph_outputs=True)
    else:
        output_value.replace_all_uses_with(input_value)

    # 4. Remove the node from the graph.
    graph.remove(node, safe=True)


def insert_on_edge(
    value: _core.Value,
    new_node: _core.Node,
    *,
    output_index: int = 0,
) -> _core.Value:
    """Insert a node on all edges from a value to its consumers.

    After this operation:

    - *new_node* receives *value* as input (already wired by the caller).
    - All previous consumers of *value* now consume
      ``new_node.outputs[output_index]``.
    - *new_node* is inserted into the graph at the appropriate position
      (after the producer of *value*, or at the start of the graph if
      *value* is a graph input).

    The caller is responsible for creating *new_node* with *value* as one
    of its inputs. This function handles rewiring downstream consumers and
    inserting the node into the graph.

    Args:
        value: The value whose consumers should be redirected.
        new_node: The node to insert. Must have *value* as one of its inputs.
        output_index: Which output of *new_node* replaces *value* for
            downstream consumers. Defaults to ``0``.

    Returns:
        The new output value (``new_node.outputs[output_index]``) that
        consumers now use.

    Raises:
        ValueError: If *value* has no producer and is not a graph input
            or initializer.
        ValueError: If *new_node* does not consume *value* as an input.

    Example::

        >>> import onnx_ir as ir
        >>> from onnx_ir import editing
        >>> # Insert a Cast after a value
        >>> x = some_node.outputs[0]
        >>> cast = ir.node("Cast", inputs=[x], attributes={"to": ir.DataType.FLOAT16})
        >>> new_x = editing.insert_on_edge(x, cast)
    """
    # Validate: new_node must consume value as one of its inputs
    if not any(inp is value for inp in new_node.inputs):
        raise ValueError(
            f"new_node ({new_node.op_type!r}) does not have the given value "
            f"as an input. Create new_node with the value as one of its inputs "
            f"before calling insert_on_edge."
        )

    # Validate: new_node must not already belong to a graph
    if new_node.graph is not None:
        raise ValueError(
            f"new_node ({new_node.op_type!r}) already belongs to graph "
            f"{new_node.graph.name!r}. insert_on_edge expects a new node "
            f"not yet added to any graph."
        )

    # Determine the graph to insert into
    graph = value.graph
    if graph is None:
        raise ValueError(
            "Cannot determine which graph to insert into. "
            "The value has no producer and is not a graph input or initializer."
        )

    new_output = new_node.outputs[output_index]

    # Redirect all consumers of value (except new_node itself) to new_output.
    # Snapshot uses first since we mutate during iteration.
    uses_to_redirect = [
        (user_node, idx)
        for user_node, idx in value.uses()
        if user_node is not new_node
    ]
    for user_node, idx in uses_to_redirect:
        user_node.replace_input_with(idx, new_output)

    # Handle graph outputs: if value is a graph output, replace it with new_output
    if value.is_graph_output():
        for i, output in enumerate(graph.outputs):
            if output is value:
                graph.outputs[i] = new_output

    # Insert new_node into the graph at the correct position
    producer = value.producer()
    if producer is not None:
        graph.insert_after(producer, new_node)
    else:
        # value is a graph input or initializer — insert at the beginning
        first_node = next(iter(graph), None)
        if first_node is not None:
            graph.insert_before(first_node, new_node)
        else:
            graph.append(new_node)

    return new_output


def replace_subgraph(
    old_nodes: Sequence[_core.Node],
    new_nodes: Sequence[_core.Node],
    output_mapping: dict[_core.Value, _core.Value],
    *,
    propagate_metadata: bool = True,
) -> None:
    """Replace a subgraph of connected nodes with new nodes.

    This is the general form of :func:`replace_node` for multi-node
    patterns (e.g., fusing MatMul + Add into Gemm).

    The function:

    1. Validates that *old_nodes* belong to the same graph.
    2. When *propagate_metadata* is ``True``, copies type, shape, name,
       and ``const_value`` from old output values to their replacements
       in *output_mapping*.
    3. Redirects all external consumers of mapped old values to new values.
    4. Handles graph output replacement (inserting Identity nodes when
       needed to avoid conflicts).
    5. Inserts *new_nodes* at the position of the earliest *old_node*.
    6. Removes *old_nodes* from the graph in safe mode.

    Args:
        old_nodes: Nodes to remove. Must all belong to the same graph.
        new_nodes: Replacement nodes. Must not already belong to a graph.
        output_mapping: Maps old output values that have external consumers
            to their replacement values from *new_nodes*.
        propagate_metadata: If ``True`` (default), propagate type, shape,
            ``const_value``, and name from old outputs to new outputs.

    Raises:
        ValueError: If *old_nodes* is empty.
        ValueError: If *old_nodes* span multiple graphs.
        ValueError: If any new node already belongs to a graph.
        ValueError: If any old output with external consumers is not
            in *output_mapping*.

    Example::

        >>> import onnx_ir as ir
        >>> from onnx_ir import editing
        >>> # Fuse MatMul + Add into Gemm
        >>> matmul_node = ...
        >>> add_node = ...
        >>> gemm = ir.node("Gemm",
        ...     inputs=[matmul_node.inputs[0], matmul_node.inputs[1], add_node.inputs[1]],
        ...     attributes={"alpha": 1.0, "beta": 1.0})
        >>> editing.replace_subgraph(
        ...     old_nodes=[matmul_node, add_node],
        ...     new_nodes=[gemm],
        ...     output_mapping={add_node.outputs[0]: gemm.outputs[0]})
    """
    if not old_nodes:
        raise ValueError("old_nodes must not be empty.")

    # Validate all old_nodes belong to the same graph
    old_nodes_set = frozenset(old_nodes)
    graph = old_nodes[0].graph
    if graph is None:
        raise ValueError(
            f"Node {old_nodes[0]!r} does not belong to a graph."
        )
    for node in old_nodes[1:]:
        if node.graph is not graph:
            raise ValueError(
                f"All old_nodes must belong to the same graph. "
                f"Node {node!r} belongs to {node.graph!r}, "
                f"expected {graph!r}."
            )

    # Validate new_nodes are not already in a graph
    for node in new_nodes:
        if node.graph is not None:
            raise ValueError(
                f"new_node {node!r} already belongs to a graph ({node.graph!r}). "
                "Replacement nodes must not belong to any graph."
            )

    # Validate that all old outputs with *external* consumers are mapped.
    # Internal edges (consumer is also in old_nodes_set) don't need mapping.
    graph_output_values = frozenset(graph.outputs)
    for node in old_nodes:
        for old_output in node.outputs:
            if old_output in output_mapping:
                continue
            has_external_use = any(
                user not in old_nodes_set for user, _ in old_output.uses()
            )
            if has_external_use or old_output in graph_output_values:
                raise ValueError(
                    f"Old output {old_output!r} has external consumers or is a "
                    "graph output but is not in output_mapping."
                )

    # Find the earliest old_node in graph order as the insertion point
    insertion_point: _core.Node | None = None
    for graph_node in graph:
        if graph_node in old_nodes_set:
            insertion_point = graph_node
            break
    assert insertion_point is not None  # Guaranteed since old_nodes is non-empty and validated

    # 1. Propagate metadata from old outputs to new outputs
    if propagate_metadata:
        for old_value, new_value in output_mapping.items():
            _propagate_value_metadata(old_value, new_value)

    # 2. Handle graph outputs before rewiring uses
    for old_value, new_value in output_mapping.items():
        if old_value in graph_output_values:
            _handle_graph_output_replacement(
                graph, old_value, new_value, insertion_point
            )

    # 3. Redirect all external consumer uses of old outputs to new outputs
    for old_value, new_value in output_mapping.items():
        if old_value.is_graph_output():
            old_value.replace_all_uses_with(new_value, replace_graph_outputs=True)
        else:
            old_value.replace_all_uses_with(new_value)

    # 4. Insert new_nodes at the earliest old node's position
    if new_nodes:
        graph.insert_before(insertion_point, new_nodes)

    # 5. Remove all old_nodes (safe=True detaches inputs and validates)
    graph.remove(old_nodes, safe=True)
