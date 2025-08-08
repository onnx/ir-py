# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Utilities for traversing the IR graph."""

from __future__ import annotations

__all__ = [
    "RecursiveGraphIterator",
]

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
