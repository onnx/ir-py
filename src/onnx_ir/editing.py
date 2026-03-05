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
    raise NotImplementedError("replace_node is not yet implemented")


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
    raise NotImplementedError("eliminate_node is not yet implemented")


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
    raise NotImplementedError("insert_on_edge is not yet implemented")


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
    raise NotImplementedError("replace_subgraph is not yet implemented")
