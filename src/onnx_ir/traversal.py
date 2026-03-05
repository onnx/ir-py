# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Utilities for traversing the IR graph."""

from __future__ import annotations

__all__ = [
    "RecursiveGraphIterator",
    "topological_order",
]

import heapq
from collections.abc import Iterator, Reversible
from typing import Callable, Union

from typing_extensions import Self

from onnx_ir import _core, _enums

GraphLike = Union[_core.Graph, _core.Function, _core.GraphView]


class RecursiveGraphIterator(Iterator[_core.Node], Reversible[_core.Node]):
    def __init__(
        self,
        graph_like: GraphLike,
        *,
        recursive: Callable[[_core.Node], bool] | None = None,
        reverse: bool = False,
        enter_graph: Callable[[GraphLike], None] | None = None,
        exit_graph: Callable[[GraphLike], None] | None = None,
    ):
        """Iterate over the nodes in the graph, recursively visiting subgraphs.

        This iterator allows for traversing the nodes of a graph and its subgraphs
        in a depth-first manner. It supports optional callbacks for entering and exiting
        subgraphs, as well as a callback `recursive` to determine whether to visit subgraphs
        contained within nodes.

        .. versionadded:: 0.1.6
            Added the `enter_graph` and `exit_graph` callbacks.

        Args:
            graph_like: The graph to traverse.
            recursive: A callback that determines whether to recursively visit the subgraphs
                contained in a node. If not provided, all nodes in subgraphs are visited.
            reverse: Whether to iterate in reverse order.
            enter_graph: An optional callback that is called when entering a subgraph.
            exit_graph: An optional callback that is called when exiting a subgraph.
        """
        self._graph = graph_like
        self._recursive = recursive
        self._reverse = reverse
        self._iterator = self._recursive_node_iter(graph_like)
        self._enter_graph = enter_graph
        self._exit_graph = exit_graph

    def __iter__(self) -> Self:
        self._iterator = self._recursive_node_iter(self._graph)
        return self

    def __next__(self) -> _core.Node:
        return next(self._iterator)

    def _recursive_node_iter(
        self, graph: _core.Graph | _core.Function | _core.GraphView
    ) -> Iterator[_core.Node]:
        iterable = reversed(graph) if self._reverse else graph

        if self._enter_graph is not None:
            self._enter_graph(graph)

        for node in iterable:  # type: ignore[union-attr]
            yield node
            if self._recursive is not None and not self._recursive(node):
                continue
            yield from self._iterate_subgraphs(node)

        if self._exit_graph is not None:
            self._exit_graph(graph)

    def _iterate_subgraphs(self, node: _core.Node):
        for attr in node.attributes.values():
            if not isinstance(attr, _core.Attr):
                continue
            if attr.type == _enums.AttributeType.GRAPH:
                if self._enter_graph is not None:
                    self._enter_graph(attr.value)
                yield from RecursiveGraphIterator(
                    attr.value,
                    recursive=self._recursive,
                    reverse=self._reverse,
                    enter_graph=self._enter_graph,
                    exit_graph=self._exit_graph,
                )
                if self._exit_graph is not None:
                    self._exit_graph(attr.value)
            elif attr.type == _enums.AttributeType.GRAPHS:
                graphs = reversed(attr.value) if self._reverse else attr.value
                for graph in graphs:
                    if self._enter_graph is not None:
                        self._enter_graph(graph)
                    yield from RecursiveGraphIterator(
                        graph,
                        recursive=self._recursive,
                        reverse=self._reverse,
                        enter_graph=self._enter_graph,
                        exit_graph=self._exit_graph,
                    )
                    if self._exit_graph is not None:
                        self._exit_graph(graph)

    def __reversed__(self) -> Iterator[_core.Node]:
        return RecursiveGraphIterator(
            self._graph,
            recursive=self._recursive,
            reverse=not self._reverse,
            enter_graph=self._enter_graph,
            exit_graph=self._exit_graph,
        )


def topological_order(
    graph_like: GraphLike,
    *,
    recursive: bool = False,
) -> Iterator[_core.Node]:
    """Iterate over nodes in topological order without modifying the graph.

    Uses Kahn's algorithm with a min-heap to produce a stable topological
    ordering. Nodes appearing earlier in the original graph order are yielded
    first among nodes at the same topological level.

    This is a lazy iterator — nodes are yielded one at a time without
    materializing the full sorted list. For large graphs, this avoids
    allocating the full sorted node list when you only need a single pass.

    When ``recursive`` is False (default), implicit dependencies through
    subgraph attributes (e.g., an ``If`` node's branch referencing an
    outer-scope value) are still respected in the ordering.

    .. versionadded:: 0.2

    Args:
        graph_like: The graph to iterate over.
        recursive: If True, also yields nodes from subgraphs in topological
            order. Subgraph nodes are yielded before their parent node,
            reflecting the implicit data dependency.

    Yields:
        Nodes in topological order.

    Raises:
        ValueError: If the graph contains a cycle.

    Example::

        for node in ir.traversal.topological_order(graph):
            # Process nodes in dependency order without mutating graph
            process(node)
    """
    all_nodes = list(RecursiveGraphIterator(graph_like))
    if not all_nodes:
        return

    if recursive:
        yield from _kahns_algorithm(all_nodes)
    else:
        top_level_set = set(graph_like)
        for node in _kahns_algorithm(all_nodes):
            if node in top_level_set:
                yield node


def _kahns_algorithm(nodes: list[_core.Node]) -> Iterator[_core.Node]:
    """Yield nodes in topological order using Kahn's algorithm.

    Uses a min-heap keyed by original node position for stable ordering.
    Handles subgraph attributes: all nodes in a node's subgraph attributes
    are considered predecessors of that node.
    """
    node_set: set[_core.Node] = set(nodes)
    node_index: dict[_core.Node, int] = {node: i for i, node in enumerate(nodes)}

    # in_degree tracks unprocessed predecessor edges for each node
    in_degree: dict[_core.Node, int] = dict.fromkeys(nodes, 0)
    # successors maps each node to nodes that depend on it
    successors: dict[_core.Node, list[_core.Node]] = {node: [] for node in nodes}

    for node in nodes:
        # Direct input dependencies: producer of each input value
        for input_value in node.inputs:
            if input_value is None:
                continue
            predecessor = input_value.producer()
            if predecessor is None or predecessor not in node_set:
                continue
            successors[predecessor].append(node)
            in_degree[node] += 1

        # Subgraph dependencies: all direct nodes in subgraph attributes
        # are predecessors of the parent node
        for attr in node.attributes.values():
            if not isinstance(attr, _core.Attr):
                continue
            if attr.type == _enums.AttributeType.GRAPH:
                for sub_node in attr.value:
                    if sub_node in node_set:
                        successors[sub_node].append(node)
                        in_degree[node] += 1
            elif attr.type == _enums.AttributeType.GRAPHS:
                for attr_graph in attr.value:
                    for sub_node in attr_graph:
                        if sub_node in node_set:
                            successors[sub_node].append(node)
                            in_degree[node] += 1

    # Min-heap of (original_index, node) for stable ordering
    heap: list[tuple[int, _core.Node]] = [
        (node_index[n], n) for n in nodes if in_degree[n] == 0
    ]
    heapq.heapify(heap)

    sorted_count = 0
    while heap:
        _, node = heapq.heappop(heap)
        sorted_count += 1
        yield node
        for successor in successors[node]:
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                heapq.heappush(heap, (node_index[successor], successor))

    if sorted_count != len(nodes):
        raise ValueError("Graph contains a cycle, topological sort is not possible.")
