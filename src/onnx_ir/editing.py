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
- :class:`SubgraphHandle`: Immutable, boundary-annotated handle to a subgraph
  with auto-discovered inputs/outputs and :meth:`~SubgraphHandle.replace_with`.
- :class:`GraphCheckpoint`: Snapshot-based transaction with rollback/commit
  support and a context manager for safe graph modifications.

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
    # Tier 2
    "GraphCheckpoint",
    "SubgraphHandle",
]

from collections.abc import Collection, Iterator, Sequence
from typing import Literal

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


# ============================================================================
# Tier 2: SubgraphHandle
# ============================================================================


class SubgraphHandle:
    """An immutable, boundary-annotated handle to a subgraph within a parent Graph.

    Unlike :class:`~onnx_ir.GraphView` (read-only, no parent reference, stores tuples),
    ``SubgraphHandle`` knows its parent graph, auto-discovers boundary values,
    and supports mutation operations that delegate to existing editing functions.

    Terminology:
        - **inputs**: Values consumed by subgraph nodes but produced *outside* the subgraph
          (or are graph inputs/initializers). These are the subgraph's external dependencies.
        - **outputs**: Values produced by subgraph nodes and consumed by at least one node
          *outside* the subgraph, or that appear in the parent graph's output list.
          These are the subgraph's externally-visible results.
        - **internal values**: Values produced and consumed entirely within the subgraph.

    The handle is immutable after construction. Calling :meth:`replace_with`
    consumes the handle (the nodes are removed from the parent graph), and
    using the handle after that raises :class:`RuntimeError`.

    Example::

        import onnx_ir as ir
        from onnx_ir import editing

        # Find a MatMul+Add pattern and analyze it
        handle = editing.SubgraphHandle(graph, [matmul_node, add_node])
        print(handle.inputs)   # e.g., (A, B, bias)
        print(handle.outputs)  # e.g., (add_output,)

        # Replace with fused Gemm
        gemm = ir.node("Gemm", inputs=[*handle.inputs])
        handle.replace_with(
            new_nodes=[gemm],
            output_mapping={add_node.outputs[0]: gemm.outputs[0]},
        )
    """

    def __init__(
        self,
        parent: _core.Graph,
        nodes: Collection[_core.Node],
    ) -> None:
        """Create a SubgraphHandle from a parent graph and a set of nodes.

        Args:
            parent: The graph that contains all the nodes.
            nodes: The nodes in the subgraph. Must all belong to *parent*.

        Raises:
            ValueError: If *nodes* is empty.
            ValueError: If any node does not belong to *parent*.
        """
        if not nodes:
            raise ValueError("nodes must not be empty.")
        node_set = frozenset(nodes)
        for node in node_set:
            if node.graph is not parent:
                raise ValueError(
                    f"Node {node!r} does not belong to the parent graph {parent!r}. "
                    f"All nodes must belong to the parent graph."
                )

        self._parent = parent
        self._nodes = node_set
        self._consumed = False

        # Compute ordered nodes (parent-graph order)
        self._ordered_nodes: tuple[_core.Node, ...] = tuple(
            n for n in parent if n in self._nodes
        )

        # Compute boundaries
        self._inputs: tuple[_core.Value, ...]
        self._outputs: tuple[_core.Value, ...]
        self._internal_values: frozenset[_core.Value]
        self._compute_boundaries()

    def _compute_boundaries(self) -> None:
        """Discover inputs, outputs, and internal values in O(V + E)."""
        node_set = self._nodes

        inputs: list[_core.Value] = []
        outputs: list[_core.Value] = []
        internal: set[_core.Value] = set()
        seen_inputs: set[_core.Value] = set()

        # Discover inputs: any input value whose producer is NOT in the subgraph
        for node in self._ordered_nodes:
            for inp in node.inputs:
                if inp is None or inp in seen_inputs:
                    continue
                producer = inp.producer()
                if producer is None or producer not in node_set:
                    inputs.append(inp)
                    seen_inputs.add(inp)

        # Discover outputs: any output value that has a consumer outside the
        # subgraph, or appears in parent.outputs
        parent_outputs = frozenset(self._parent.outputs)
        for node in self._ordered_nodes:
            for out in node.outputs:
                has_external_consumer = any(
                    user not in node_set for user, _ in out.uses()
                )
                if has_external_consumer or out in parent_outputs:
                    outputs.append(out)
                else:
                    internal.add(out)

        self._inputs = tuple(inputs)
        self._outputs = tuple(outputs)
        self._internal_values = frozenset(internal)

    # ---- Construction alternatives ----

    @classmethod
    def between(
        cls,
        parent: _core.Graph,
        input_values: Sequence[_core.Value],
        output_values: Sequence[_core.Value],
    ) -> SubgraphHandle:
        """Create a SubgraphHandle from boundary values.

        Performs a backward traversal from *output_values* to *input_values*
        to discover all nodes in between. Uses the same algorithm as
        :func:`onnx_ir.convenience.extract` but without cloning.

        Args:
            parent: The graph containing the subgraph.
            input_values: Values that bound the subgraph's "top" edge.
                Traversal stops at producers of these values.
            output_values: Values that bound the subgraph's "bottom" edge.
                Traversal starts from producers of these values.

        Raises:
            ValueError: If the subgraph is not properly bounded (unreachable
                inputs, missing values).
        """
        from onnx_ir._convenience._extractor import _find_subgraph_bounded_by_values

        nodes, _ = _find_subgraph_bounded_by_values(
            parent, input_values, output_values, parent_graph=parent
        )
        return cls(parent, nodes)

    # ---- Read-only properties ----

    def _check_not_consumed(self) -> None:
        """Raise RuntimeError if this handle has been consumed."""
        if self._consumed:
            raise RuntimeError(
                "This SubgraphHandle has already been consumed by a mutation "
                "operation (replace_with). Create a new handle if needed."
            )

    @property
    def parent(self) -> _core.Graph:
        """The parent graph containing this subgraph."""
        return self._parent

    @property
    def nodes(self) -> frozenset[_core.Node]:
        """The nodes in this subgraph (unordered set)."""
        return self._nodes

    @property
    def inputs(self) -> tuple[_core.Value, ...]:
        """Values consumed by subgraph nodes but produced outside.

        Includes graph inputs and initializers consumed by the subgraph.
        Order: deterministic, sorted by first use position in the parent graph.
        """
        return self._inputs

    @property
    def outputs(self) -> tuple[_core.Value, ...]:
        """Values produced by subgraph nodes and consumed externally.

        Includes values that appear in ``parent.outputs``.
        Order: deterministic, sorted by producer position in the parent graph.
        """
        return self._outputs

    @property
    def internal_values(self) -> frozenset[_core.Value]:
        """Values produced and consumed entirely within the subgraph."""
        return self._internal_values

    def __len__(self) -> int:
        """Number of nodes in the subgraph."""
        return len(self._nodes)

    def __contains__(self, node: _core.Node) -> bool:
        """Check if a node is in this subgraph."""
        return node in self._nodes

    def __iter__(self) -> Iterator[_core.Node]:
        """Iterate over nodes in parent-graph order."""
        return iter(self._ordered_nodes)

    # ---- Mutation operations (consume the handle) ----

    def replace_with(
        self,
        new_nodes: Sequence[_core.Node],
        output_mapping: dict[_core.Value, _core.Value],
        *,
        propagate_metadata: bool = True,
    ) -> None:
        """Replace this subgraph with new nodes.

        Delegates to :func:`replace_subgraph` with the handle's nodes.
        After this call, the handle is **consumed** and must not be reused.

        Before delegating, verifies that all nodes are still in the parent
        graph (guards against stale handles after checkpoint rollback).

        Args:
            new_nodes: Replacement nodes.
            output_mapping: Maps old output values to new output values.
            propagate_metadata: If True, propagate type/shape/name/const_value.

        Raises:
            RuntimeError: If the handle has already been consumed.
            RuntimeError: If any node no longer belongs to the parent graph
                (e.g., after a checkpoint rollback invalidated this handle).
        """
        self._check_not_consumed()

        # Liveness check: verify nodes still belong to the parent graph
        for node in self._nodes:
            if node.graph is not self._parent:
                raise RuntimeError(
                    f"Node {node!r} no longer belongs to the parent graph. "
                    "This handle is stale — the graph may have been modified "
                    "by a checkpoint rollback or other operation."
                )

        replace_subgraph(
            old_nodes=list(self._ordered_nodes),
            new_nodes=new_nodes,
            output_mapping=output_mapping,
            propagate_metadata=propagate_metadata,
        )
        self._consumed = True

    def as_graph_view(self) -> _core.GraphView:
        """Return a read-only GraphView of this subgraph.

        Does NOT consume the handle. The GraphView references the same
        Node objects (not copies).

        .. warning::
            The GraphView's lifetime is bounded by this handle's. After
            :meth:`replace_with` consumes the handle, any previously-returned
            GraphView is stale (its nodes have been removed from the graph).

        Raises:
            RuntimeError: If the handle has already been consumed.
        """
        self._check_not_consumed()
        return _core.GraphView(
            inputs=list(self._inputs),
            outputs=list(self._outputs),
            nodes=self._ordered_nodes,
        )


# ============================================================================
# Tier 2: GraphCheckpoint
# ============================================================================


class GraphCheckpoint:
    """A checkpoint for a model's graph, enabling rollback on failure.

    Takes a snapshot of ``model.graph`` at creation time. If the transformation
    fails or produces an invalid result, call :meth:`rollback` to restore the
    graph to its checkpointed state.

    **Cost:** O(V + E) at creation (``graph.clone()``), O(1) at rollback (reference
    swap), O(1) at commit (discard the clone for GC).

    **Reference invalidation:** After :meth:`rollback`, the model's graph is a
    *different object* with different Node and Value instances. Any local
    variables referencing nodes/values from the pre-rollback graph are
    **stale** and must not be used. The context manager pattern naturally
    avoids this issue because rollback exits the ``with`` block.

    **Functions are NOT checkpointed.** Only ``model.graph`` is cloned. If a
    pass modifies model functions, those changes survive rollback.

    Usage as context manager (recommended)::

        with editing.GraphCheckpoint(model) as cp:
            editing.replace_node(old_node, new_node)
            editing.eliminate_node(identity_node)
            if not validate(model):
                cp.rollback()
        # After the block: graph is either transformed (success)
        # or restored (rollback / exception).

    Usage for fine-grained rollback in a loop::

        for matmul, add in find_fusible_pairs(model.graph):
            with editing.GraphCheckpoint(model):
                try:
                    gemm = ir.node("Gemm", inputs=[...])
                    editing.replace_subgraph([matmul, add], [gemm], {...})
                except ValueError:
                    pass  # auto-rollback on exception, try next pair

    Explicit usage (without context manager)::

        cp = editing.GraphCheckpoint(model)
        editing.replace_node(old_node, new_node)
        if not is_valid(model):
            cp.rollback()
        else:
            cp.commit()  # free the snapshot

    Args:
        model: The model whose graph to checkpoint.
    """

    def __init__(self, model: _core.Model) -> None:
        """Create a checkpoint of the model's current graph state.

        Args:
            model: The model to checkpoint. ``model.graph`` is cloned.
        """
        self._model = model
        self._original_graph: _core.Graph | None = model.graph  # Track identity for safety check
        self._saved_graph: _core.Graph | None = model.graph.clone()
        self._active = True

    @property
    def is_active(self) -> bool:
        """True if this checkpoint has not been committed or rolled back."""
        return self._active

    def rollback(self) -> None:
        """Restore the model's graph to the checkpointed state.

        After rollback:
        - ``model.graph`` is the cloned graph from checkpoint creation.
        - All Node/Value references from the pre-rollback graph are stale.
        - The checkpoint is consumed (not reusable).

        Nested checkpoints are supported in LIFO order: inner rollback first,
        then outer rollback restores to the outer checkpoint's saved state.

        Raises:
            RuntimeError: If the checkpoint has already been committed or
                rolled back.
        """
        if not self._active:
            raise RuntimeError(
                "This checkpoint has already been committed or rolled back."
            )
        assert self._saved_graph is not None
        self._model.graph = self._saved_graph
        self._saved_graph = None
        self._original_graph = None
        self._active = False

    def commit(self) -> None:
        """Discard the checkpoint snapshot, freeing memory.

        Call this when the transformation succeeded and rollback is no
        longer needed. If using the context manager, commit is called
        automatically on clean exit.

        Raises:
            RuntimeError: If the checkpoint has already been committed or
                rolled back.
        """
        if not self._active:
            raise RuntimeError(
                "This checkpoint has already been committed or rolled back."
            )
        self._saved_graph = None  # Free the clone for GC
        self._original_graph = None
        self._active = False

    def __enter__(self) -> GraphCheckpoint:
        """Enter the checkpoint context."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> Literal[False]:
        """Exit the checkpoint context.

        - On clean exit (no exception, no prior rollback): calls ``commit()``.
        - On exception (no prior rollback): calls ``rollback()``.
        - If already rolled back or committed: no-op.

        Exceptions are never suppressed (returns ``False``).
        """
        if self._active:
            if exc_type is not None:
                self.rollback()
            else:
                self.commit()
        return False  # Never suppress exceptions
