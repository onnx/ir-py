# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Logic for cloning graphs."""

from __future__ import annotations

import functools
import typing
from collections.abc import Callable, Mapping
from typing import TypeVar

from typing_extensions import Concatenate, ParamSpec

from onnx_ir import _core, _enums

P = ParamSpec("P")
R = TypeVar("R")


def _capture_error_context(
    func: Callable[Concatenate[Cloner, P], R],
) -> Callable[Concatenate[Cloner, P], R]:
    """Decorator to capture error context during cloning."""

    @functools.wraps(func)
    def wrapper(self: Cloner, *args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            raise RuntimeError(
                f"In {func.__name__} with args {args!r} and kwargs {kwargs!r}"
            ) from e

    return wrapper


class Cloner:
    """Utilities for creating a copy of IR objects with substitutions for attributes/input values."""

    def __init__(
        self,
        *,
        attr_map: Mapping[str, _core.Attr],
        value_map: dict[_core.Value, _core.Value | None],
        metadata_props: dict[str, str],
        post_process: Callable[[_core.Node], None] = lambda _: None,
        resolve_ref_attrs: bool = False,
        allow_outer_scope_values: bool = False,
    ) -> None:
        """Initializes the cloner.

        Args:
            attr_map: A mapping from attribute names to attributes to substitute, used when
                inlining functions.
            value_map: A mapping from original values to cloned values. If a value is not in
                this map, it is assumed to be a graph input and will be cloned as a new value.
            metadata_props: Metadata properties to add to cloned nodes.
            post_process: A callback invoked after cloning each node, allowing for additional
                processing on the cloned node.
            resolve_ref_attrs: Whether to resolve reference attributes using the attr_map.
                Set to True when inlining functions.
            allow_outer_scope_values: When True, values that are from outer scopes
                (not defined in this graph) will not be cloned. Instead, the cloned
                graph will reference the same outer scope values. This is useful
                when cloning subgraphs that reference values from the outer graph.
                When False (default), values from outer scopes will cause an error if they
                are referenced in the cloned graph.
        """
        self._value_map = value_map
        self._attr_map = attr_map
        self._metadata_props = metadata_props
        self._post_process = post_process
        self._resolve_ref_attrs = resolve_ref_attrs
        self._allow_outer_scope_values = allow_outer_scope_values

    @_capture_error_context
    def _get_value(self, value: _core.Value) -> _core.Value | None:
        return self._value_map[value]

    @_capture_error_context
    def _clone_or_get_value(self, value: _core.Value) -> _core.Value:
        if value in self._value_map:
            known_value = self._value_map[value]
            assert known_value is not None, f"BUG: Value {value} mapped to None in value map"
            return known_value
        # If the value is not in the value map, it must be a graph input.
        # Note: value.producer() may not be None when the value is an input of a GraphView
        new_value = _core.Value(
            name=value.name,
            type=value.type,
            shape=value.shape.copy() if value.shape is not None else None,
            doc_string=value.doc_string,
            const_value=value.const_value,
        )
        if value.metadata_props:
            new_value.metadata_props.update(value.metadata_props)
        if value.meta:
            new_value.meta.update(value.meta)
        self._value_map[value] = new_value
        return new_value

    @_capture_error_context
    def clone_attr(self, key: str, attr: _core.Attr) -> _core.Attr | None:
        if not attr.is_ref():
            if attr.type == _enums.AttributeType.GRAPH:
                graph = self.clone_graph(attr.as_graph())
                return _core.Attr(
                    key, _enums.AttributeType.GRAPH, graph, doc_string=attr.doc_string
                )
            elif attr.type == _enums.AttributeType.GRAPHS:
                graphs = [self.clone_graph(graph) for graph in attr.as_graphs()]
                return _core.Attr(
                    key, _enums.AttributeType.GRAPHS, graphs, doc_string=attr.doc_string
                )
            return attr

        assert attr.is_ref()
        if not self._resolve_ref_attrs:
            return attr

        ref_attr_name = attr.ref_attr_name
        if ref_attr_name is None:
            raise ValueError("Reference attribute must have a name")
        if ref_attr_name in self._attr_map:
            ref_attr = self._attr_map[ref_attr_name]
            if not ref_attr.is_ref():
                return _core.Attr(
                    key, ref_attr.type, ref_attr.value, doc_string=ref_attr.doc_string
                )

            # When inlining into a function, we resolve reference attributes to other reference
            # attributes declared in the parent scope.
            assert ref_attr.ref_attr_name is not None
            return _core.RefAttr(
                key, ref_attr.ref_attr_name, ref_attr.type, doc_string=ref_attr.doc_string
            )
        # Note that if a function has an attribute-parameter X, and a call (node) to the function
        # has no attribute X, all references to X in nodes inside the function body will be
        # removed. This is just the ONNX representation of optional-attributes.
        return None

    @_capture_error_context
    def clone_node(self, node: _core.Node) -> _core.Node:
        new_inputs: list[_core.Value | None] = []
        for input in node.inputs:
            if input is None:
                new_inputs.append(input)
            elif input not in self._value_map:
                # If the node input cannot be found in the value map, it must be an outer-scope
                # value, given that the nodes are sorted topologically.
                if not self._allow_outer_scope_values:
                    graph_name = (
                        input.graph.name or "<anonymous>" if input.graph else "<unknown>"
                    )
                    raise ValueError(
                        f"Value '{input}' used by node '{node}' is an outer-scope value (from graph '{graph_name}'), "
                        "but 'allow_outer_scope_values' is set to False. Consider creating a GraphView and add the value to its "
                        "inputs then clone, or setting 'allow_outer_scope_values' to True to allow referencing outer-scope values."
                    )
                # When preserving outer-scope values, pass them through unchanged instead of cloning.
                new_inputs.append(input)
            else:
                new_inputs.append(self._get_value(input))
        new_attributes = [
            new_value
            for key, value in node.attributes.items()
            if (new_value := self.clone_attr(key, value)) is not None
        ]

        new_metadata = {**self._metadata_props, **node.metadata_props}
        # TODO: For now, node metadata overrides callnode metadata if there is a conflict.
        # Do we need to preserve both?

        new_node = _core.Node(
            node.domain,
            node.op_type,
            new_inputs,
            new_attributes,
            overload=node.overload,
            num_outputs=len(node.outputs),
            version=node.version,
            name=node.name,
            doc_string=node.doc_string,
            metadata_props=new_metadata,
        )
        if node.meta:
            new_node.meta.update(node.meta)

        # Copy output properties
        for output, new_output in zip(node.outputs, new_node.outputs):
            self._value_map[output] = new_output
            new_output.name = output.name
            new_output.shape = output.shape.copy() if output.shape is not None else None
            new_output.type = output.type
            new_output.const_value = output.const_value
            new_output.doc_string = output.doc_string
            if output.metadata_props:
                new_output.metadata_props.update(output.metadata_props)
            if output.meta:
                new_output.meta.update(output.meta)

        self._post_process(new_node)
        return new_node

    @_capture_error_context
    def clone_graph(self, graph: _core.Graph | _core.GraphView) -> _core.Graph:
        """Clones a graph with shared TensorProtocols."""
        input_values = [self._clone_or_get_value(v) for v in graph.inputs]
        initializers = [self._clone_or_get_value(v) for v in graph.initializers.values()]
        nodes = [self.clone_node(node) for node in graph]
        # Looks up already cloned values. Here we know graph outputs will not be None
        output_values = typing.cast(
            list["_core.Value"], [self._get_value(v) for v in graph.outputs]
        )

        new_graph = _core.Graph(
            input_values,
            output_values,
            nodes=nodes,
            initializers=initializers,
            doc_string=graph.doc_string,
            opset_imports=graph.opset_imports.copy(),
            name=graph.name,
        )
        if graph.metadata_props:
            new_graph.metadata_props.update(graph.metadata_props)
        if graph.meta:
            new_graph.meta.update(graph.meta)
        return new_graph
