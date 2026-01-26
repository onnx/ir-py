# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Wrappers for IR classes to enable journaling.

This module provides wrapper functions that enable journaling for ONNX IR classes.
The wrappers are applied when a Journal context is active, and they record operations
to the journal for debugging and analysis purposes.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from onnx_ir import _core
from onnx_ir.journaling import _journaling

if TYPE_CHECKING:
    from onnx_ir._core import Graph, Node, Shape, Value
    from onnx_ir._protocols import TensorProtocol, TypeProtocol

# Store original methods for restoration
original_methods: dict[str, Any] = {}


def _record_if_journal(obj: Any, operation: str, details: str | None = None) -> None:
    """Record an operation if a journal is active."""
    if (journal := _journaling.get_journal()) is not None:
        journal.record(obj, operation, details)


# =============================================================================
# TensorBase wrappers
# =============================================================================


def _tensorbase_init_wrapper(original_init):
    """Wrapper for TensorBase.__init__."""

    def wrapper(
        self,
        name: str | None = None,
        doc_string: str | None = None,
        metadata_props: dict[str, str] | None = None,
    ) -> None:
        original_init(self, name, doc_string, metadata_props)
        _record_if_journal(self, "initialize")

    return wrapper


# =============================================================================
# Node wrappers
# =============================================================================


def _node_init_wrapper(original_init):
    """Wrapper for Node.__init__."""

    def wrapper(
        self,
        domain: str,
        op_type: str,
        inputs: Iterable[Value | None],
        attributes: Iterable[_core.Attr] | dict[str, _core.Attr] = (),
        *,
        overload: str = "",
        num_outputs: int | None = 1,
        outputs: Iterable[Value] | None = None,
        version: int | None = None,
        graph: Graph | None = None,
        name: str | None = None,
        doc_string: str | None = None,
        metadata_props: dict[str, str] | None = None,
    ) -> None:
        original_init(
            self,
            domain,
            op_type,
            inputs,
            attributes,
            overload=overload,
            num_outputs=num_outputs,
            outputs=outputs,
            version=version,
            graph=graph,
            name=name,
            doc_string=doc_string,
            metadata_props=metadata_props,
        )
        _record_if_journal(self, "initialize", details=repr(self))

    return wrapper


def _node_name_setter_wrapper(original_setter):
    """Wrapper for Node.name setter."""

    def wrapper(self, value: str | None) -> None:
        _record_if_journal(self, "set_name", details=f"{self._name!r} -> {value!r}")
        original_setter(self, value)

    return wrapper


def _node_domain_setter_wrapper(original_setter):
    """Wrapper for Node.domain setter."""

    def wrapper(self, value: str) -> None:
        _record_if_journal(self, "set_domain", details=f"{self._domain!r} -> {value!r}")
        original_setter(self, value)

    return wrapper


def _node_version_setter_wrapper(original_setter):
    """Wrapper for Node.version setter."""

    def wrapper(self, value: int | None) -> None:
        _record_if_journal(self, "set_version", details=f"{self._version!r} -> {value!r}")
        original_setter(self, value)

    return wrapper


def _node_op_type_setter_wrapper(original_setter):
    """Wrapper for Node.op_type setter."""

    def wrapper(self, value: str) -> None:
        _record_if_journal(self, "set_op_type", details=f"{self._op_type!r} -> {value!r}")
        original_setter(self, value)

    return wrapper


def _node_overload_setter_wrapper(original_setter):
    """Wrapper for Node.overload setter."""

    def wrapper(self, value: str) -> None:
        _record_if_journal(self, "set_overload", details=f"{self._overload!r} -> {value!r}")
        original_setter(self, value)

    return wrapper


def _node_resize_inputs_wrapper(original_method):
    """Wrapper for Node.resize_inputs."""

    def wrapper(self, new_size: int, /) -> None:
        _record_if_journal(self, "resize_inputs", details=f"{len(self._inputs)} -> {new_size}")
        original_method(self, new_size)

    return wrapper


def _node_prepend_wrapper(original_method):
    """Wrapper for Node.prepend."""

    def wrapper(self, /, nodes: Node | Iterable[Node]) -> None:
        _record_if_journal(self, "prepend", details=repr(nodes))
        original_method(self, nodes)

    return wrapper


def _node_append_wrapper(original_method):
    """Wrapper for Node.append."""

    def wrapper(self, /, nodes: Node | Iterable[Node]) -> None:
        _record_if_journal(self, "append", details=repr(nodes))
        original_method(self, nodes)

    return wrapper


def _node_resize_outputs_wrapper(original_method):
    """Wrapper for Node.resize_outputs."""

    def wrapper(self, new_size: int, /) -> None:
        _record_if_journal(
            self, "resize_outputs", details=f"{len(self._outputs)} -> {new_size}"
        )
        original_method(self, new_size)

    return wrapper


def _node_graph_setter_wrapper(original_setter):
    """Wrapper for Node.graph setter."""

    def wrapper(self, value: Graph | None) -> None:
        _record_if_journal(
            self,
            "set_graph",
            details=f"{(value.name if isinstance(value, _core.Graph) else value)!r}",
        )
        original_setter(self, value)

    return wrapper


# =============================================================================
# Value wrappers
# =============================================================================


def _value_init_wrapper(original_init):
    """Wrapper for Value.__init__."""

    def wrapper(
        self,
        producer: Node | None = None,
        *,
        index: int | None = None,
        name: str | None = None,
        type: TypeProtocol | None = None,
        shape: Shape | None = None,
        const_value: TensorProtocol | None = None,
        doc_string: str | None = None,
    ) -> None:
        original_init(
            self,
            producer,
            index=index,
            name=name,
            type=type,
            shape=shape,
            const_value=const_value,
            doc_string=doc_string,
        )
        _record_if_journal(self, "initialize", repr(self))

    return wrapper


def _value_name_setter_wrapper(original_setter):
    """Wrapper for Value.name setter."""

    def wrapper(self, value: str | None) -> None:
        _record_if_journal(self, "set_name", details=f"{self._name!r} -> {value!r}")
        original_setter(self, value)

    return wrapper


def _value_type_setter_wrapper(original_setter):
    """Wrapper for Value.type setter."""

    def wrapper(self, value: TypeProtocol | None) -> None:
        _record_if_journal(self, "set_type", details=f"{self._type!r} -> {value!r}")
        original_setter(self, value)

    return wrapper


def _value_shape_setter_wrapper(original_setter):
    """Wrapper for Value.shape setter."""

    def wrapper(self, value: Shape | None) -> None:
        _record_if_journal(self, "set_shape", details=f"{self._shape!r} -> {value!r}")
        original_setter(self, value)

    return wrapper


def _value_const_value_setter_wrapper(original_setter):
    """Wrapper for Value.const_value setter."""

    def wrapper(self, value: TensorProtocol | None) -> None:
        _record_if_journal(
            self, "set_const_value", details=f"{self._const_value!r} -> {value!r}"
        )
        original_setter(self, value)

    return wrapper


def _value_replace_all_uses_with_wrapper(original_method):
    """Wrapper for Value.replace_all_uses_with."""

    def wrapper(
        self,
        replacement: Value,
        /,
        *,
        replace_graph_outputs: bool = False,
    ) -> None:
        _record_if_journal(
            self,
            "replace_all_uses_with",
            details=f"replacement={replacement!r}, replace_graph_outputs={replace_graph_outputs}",
        )
        original_method(self, replacement, replace_graph_outputs=replace_graph_outputs)

    return wrapper


def _value_merge_shapes_wrapper(original_method):
    """Wrapper for Value.merge_shapes."""

    def wrapper(self, other: Shape | None, /) -> None:
        _record_if_journal(
            self, "merge_shapes", details=f"original={self._shape!r}, other={other!r}"
        )
        original_method(self, other)

    return wrapper


# =============================================================================
# Graph wrappers
# =============================================================================


def _graph_init_wrapper(original_init):
    """Wrapper for Graph.__init__."""

    def wrapper(
        self,
        inputs: Iterable[Value],
        outputs: Iterable[Value],
        *,
        nodes: Iterable[Node] = (),
        initializers: Iterable[Value] = (),
        doc_string: str | None = None,
        opset_imports: dict[str, int] | None = None,
        name: str | None = None,
        metadata_props: dict[str, str] | None = None,
    ) -> None:
        original_init(
            self,
            inputs,
            outputs,
            nodes=nodes,
            initializers=initializers,
            doc_string=doc_string,
            opset_imports=opset_imports,
            name=name,
            metadata_props=metadata_props,
        )
        _record_if_journal(self, "initialize", details=str(name))

    return wrapper


def _graph_register_initializer_wrapper(original_method):
    """Wrapper for Graph.register_initializer."""

    def wrapper(self, value: Value) -> None:
        _record_if_journal(self, "register_initializer", details=repr(value))
        original_method(self, value)

    return wrapper


def _graph_append_wrapper(original_method):
    """Wrapper for Graph.append."""

    def wrapper(self, node: Node, /) -> None:
        _record_if_journal(self, "append", details=repr(node))
        original_method(self, node)

    return wrapper


def _graph_extend_wrapper(original_method):
    """Wrapper for Graph.extend."""

    def wrapper(self, nodes: Iterable[Node], /) -> None:
        _record_if_journal(self, "extend", details=repr(nodes))
        original_method(self, nodes)

    return wrapper


def _graph_remove_wrapper(original_method):
    """Wrapper for Graph.remove."""

    def wrapper(self, nodes: Node | Iterable[Node], /, safe: bool = False) -> None:
        _record_if_journal(self, "remove", details=f"nodes={nodes!r}, safe={safe}")
        original_method(self, nodes, safe)

    return wrapper


def _graph_insert_after_wrapper(original_method):
    """Wrapper for Graph.insert_after."""

    def wrapper(self, node: Node, new_nodes: Iterable[Node] | Node, /) -> None:
        _record_if_journal(
            self, "insert_after", details=f"node={node!r}, new_nodes={new_nodes!r}"
        )
        original_method(self, node, new_nodes)

    return wrapper


def _graph_insert_before_wrapper(original_method):
    """Wrapper for Graph.insert_before."""

    def wrapper(self, node: Node, new_nodes: Iterable[Node] | Node, /) -> None:
        _record_if_journal(
            self, "insert_before", details=f"node={node!r}, new_nodes={new_nodes!r}"
        )
        original_method(self, node, new_nodes)

    return wrapper


def _graph_sort_wrapper(original_method):
    """Wrapper for Graph.sort."""

    def wrapper(self) -> None:
        _record_if_journal(self, "sort")
        original_method(self)

    return wrapper


# =============================================================================
# Model wrappers
# =============================================================================


def _model_init_wrapper(original_init):
    """Wrapper for Model.__init__."""

    def wrapper(
        self,
        graph: Graph,
        *,
        ir_version: int,
        producer_name: str | None = None,
        producer_version: str | None = None,
        domain: str | None = None,
        model_version: int | None = None,
        doc_string: str | None = None,
        functions: Iterable[Any] = (),
        metadata_props: dict[str, str] | None = None,
    ) -> None:
        original_init(
            self,
            graph,
            ir_version=ir_version,
            producer_name=producer_name,
            producer_version=producer_version,
            domain=domain,
            model_version=model_version,
            doc_string=doc_string,
            functions=functions,
            metadata_props=metadata_props,
        )
        _record_if_journal(self, "initialize", details=repr(self))

    return wrapper


# =============================================================================
# Function wrappers
# =============================================================================


def _function_init_wrapper(original_init):
    """Wrapper for Function.__init__."""

    def wrapper(
        self,
        domain: str,
        name: str,
        overload: str = "",
        *,
        graph: Graph,
        attributes: Any,
    ) -> None:
        original_init(
            self,
            domain,
            name,
            overload,
            graph=graph,
            attributes=attributes,
        )
        _record_if_journal(self, "initialize", details=repr(self))

    return wrapper


def _function_name_setter_wrapper(original_setter):
    """Wrapper for Function.name setter."""

    def wrapper(self, value: str) -> None:
        _record_if_journal(self, "set_name", details=f"{self._name!r} -> {value!r}")
        original_setter(self, value)

    return wrapper


def _function_domain_setter_wrapper(original_setter):
    """Wrapper for Function.domain setter."""

    def wrapper(self, value: str) -> None:
        _record_if_journal(self, "set_domain", details=f"{self._domain!r} -> {value!r}")
        original_setter(self, value)

    return wrapper


def _function_overload_setter_wrapper(original_setter):
    """Wrapper for Function.overload setter."""

    def wrapper(self, value: str) -> None:
        _record_if_journal(self, "set_overload", details=f"{self._overload!r} -> {value!r}")
        original_setter(self, value)

    return wrapper


# =============================================================================
# Attr wrappers
# =============================================================================


def _attr_init_wrapper(original_init):
    """Wrapper for Attr.__init__."""

    def wrapper(
        self,
        name: str,
        type: Any,
        value: Any,
        *,
        ref_attr_name: str | None = None,
        doc_string: str | None = None,
    ) -> None:
        original_init(
            self,
            name,
            type,
            value,
            ref_attr_name=ref_attr_name,
            doc_string=doc_string,
        )
        _record_if_journal(self, "initialize", details=repr(self))

    return wrapper


# =============================================================================
# Store and restore original methods
# =============================================================================


def get_original_methods() -> dict[str, Any]:
    """Obtain original methods for later restoration.

    Returns:
        A dictionary mapping method names to their original implementations.
    """
    original_methods = {
        # TensorBase
        "TensorBase.__init__": _core.TensorBase.__init__,
        # Node
        "Node.__init__": _core.Node.__init__,
        "Node.name.fset": _core.Node.name.fset,
        "Node.domain.fset": _core.Node.domain.fset,
        "Node.version.fset": _core.Node.version.fset,
        "Node.op_type.fset": _core.Node.op_type.fset,
        "Node.overload.fset": _core.Node.overload.fset,
        "Node.resize_inputs": _core.Node.resize_inputs,
        "Node.prepend": _core.Node.prepend,
        "Node.append": _core.Node.append,
        "Node.resize_outputs": _core.Node.resize_outputs,
        "Node.graph.fset": _core.Node.graph.fset,
        # Value
        "Value.__init__": _core.Value.__init__,
        "Value.name.fset": _core.Value.name.fset,
        "Value.type.fset": _core.Value.type.fset,
        "Value.shape.fset": _core.Value.shape.fset,
        "Value.const_value.fset": _core.Value.const_value.fset,
        "Value.replace_all_uses_with": _core.Value.replace_all_uses_with,
        "Value.merge_shapes": _core.Value.merge_shapes,
        # Graph
        "Graph.__init__": _core.Graph.__init__,
        "Graph.register_initializer": _core.Graph.register_initializer,
        "Graph.append": _core.Graph.append,
        "Graph.extend": _core.Graph.extend,
        "Graph.remove": _core.Graph.remove,
        "Graph.insert_after": _core.Graph.insert_after,
        "Graph.insert_before": _core.Graph.insert_before,
        "Graph.sort": _core.Graph.sort,
        # Model
        "Model.__init__": _core.Model.__init__,
        # Function
        "Function.__init__": _core.Function.__init__,
        "Function.name.fset": _core.Function.name.fset,
        "Function.domain.fset": _core.Function.domain.fset,
        "Function.overload.fset": _core.Function.overload.fset,
        # Attr
        "Attr.__init__": _core.Attr.__init__,
    }

    return original_methods


def wrap_ir_classes() -> dict[str, Any]:
    """Wrap IR classes with journaling-enabled versions.

    This function replaces methods on IR classes with wrapped versions that
    record operations to the active journal.
    """
    # Store original methods if not already done
    original_methods = get_original_methods()

    # TensorBase
    _core.TensorBase.__init__ = _tensorbase_init_wrapper(
        original_methods["TensorBase.__init__"]
    )

    # Node
    _core.Node.__init__ = _node_init_wrapper(original_methods["Node.__init__"])
    _core.Node.name = property(
        _core.Node.name.fget,
        _node_name_setter_wrapper(original_methods["Node.name.fset"]),
    )
    _core.Node.domain = property(
        _core.Node.domain.fget,
        _node_domain_setter_wrapper(original_methods["Node.domain.fset"]),
    )
    _core.Node.version = property(
        _core.Node.version.fget,
        _node_version_setter_wrapper(original_methods["Node.version.fset"]),
    )
    _core.Node.op_type = property(
        _core.Node.op_type.fget,
        _node_op_type_setter_wrapper(original_methods["Node.op_type.fset"]),
    )
    _core.Node.overload = property(
        _core.Node.overload.fget,
        _node_overload_setter_wrapper(original_methods["Node.overload.fset"]),
    )
    _core.Node.resize_inputs = _node_resize_inputs_wrapper(
        original_methods["Node.resize_inputs"]
    )
    _core.Node.prepend = _node_prepend_wrapper(original_methods["Node.prepend"])
    _core.Node.append = _node_append_wrapper(original_methods["Node.append"])
    _core.Node.resize_outputs = _node_resize_outputs_wrapper(
        original_methods["Node.resize_outputs"]
    )
    _core.Node.graph = property(
        _core.Node.graph.fget,
        _node_graph_setter_wrapper(original_methods["Node.graph.fset"]),
    )

    # Value
    _core.Value.__init__ = _value_init_wrapper(original_methods["Value.__init__"])
    _core.Value.name = property(
        _core.Value.name.fget,
        _value_name_setter_wrapper(original_methods["Value.name.fset"]),
    )
    _core.Value.type = property(
        _core.Value.type.fget,
        _value_type_setter_wrapper(original_methods["Value.type.fset"]),
    )
    _core.Value.shape = property(
        _core.Value.shape.fget,
        _value_shape_setter_wrapper(original_methods["Value.shape.fset"]),
    )
    _core.Value.const_value = property(
        _core.Value.const_value.fget,
        _value_const_value_setter_wrapper(original_methods["Value.const_value.fset"]),
    )
    _core.Value.replace_all_uses_with = _value_replace_all_uses_with_wrapper(
        original_methods["Value.replace_all_uses_with"]
    )
    _core.Value.merge_shapes = _value_merge_shapes_wrapper(
        original_methods["Value.merge_shapes"]
    )

    # Graph
    _core.Graph.__init__ = _graph_init_wrapper(original_methods["Graph.__init__"])
    _core.Graph.register_initializer = _graph_register_initializer_wrapper(
        original_methods["Graph.register_initializer"]
    )
    _core.Graph.append = _graph_append_wrapper(original_methods["Graph.append"])
    _core.Graph.extend = _graph_extend_wrapper(original_methods["Graph.extend"])
    _core.Graph.remove = _graph_remove_wrapper(original_methods["Graph.remove"])
    _core.Graph.insert_after = _graph_insert_after_wrapper(
        original_methods["Graph.insert_after"]
    )
    _core.Graph.insert_before = _graph_insert_before_wrapper(
        original_methods["Graph.insert_before"]
    )
    _core.Graph.sort = _graph_sort_wrapper(original_methods["Graph.sort"])

    # Model
    _core.Model.__init__ = _model_init_wrapper(original_methods["Model.__init__"])

    # Function
    _core.Function.__init__ = _function_init_wrapper(original_methods["Function.__init__"])
    _core.Function.name = property(
        _core.Function.name.fget,
        _function_name_setter_wrapper(original_methods["Function.name.fset"]),
    )
    _core.Function.domain = property(
        _core.Function.domain.fget,
        _function_domain_setter_wrapper(original_methods["Function.domain.fset"]),
    )
    _core.Function.overload = property(
        _core.Function.overload.fget,
        _function_overload_setter_wrapper(original_methods["Function.overload.fset"]),
    )

    # Attr
    _core.Attr.__init__ = _attr_init_wrapper(original_methods["Attr.__init__"])

    return original_methods


def restore_ir_classes(original_methods: dict[str, Any]) -> None:
    """Restore IR classes to their original implementations.

    This function undoes the wrapping done by wrap_ir_classes().
    """
    # TensorBase
    _core.TensorBase.__init__ = original_methods["TensorBase.__init__"]

    # Node
    _core.Node.__init__ = original_methods["Node.__init__"]
    _core.Node.name = property(
        _core.Node.name.fget,
        original_methods["Node.name.fset"],
    )
    _core.Node.domain = property(
        _core.Node.domain.fget,
        original_methods["Node.domain.fset"],
    )
    _core.Node.version = property(
        _core.Node.version.fget,
        original_methods["Node.version.fset"],
    )
    _core.Node.op_type = property(
        _core.Node.op_type.fget,
        original_methods["Node.op_type.fset"],
    )
    _core.Node.overload = property(
        _core.Node.overload.fget,
        original_methods["Node.overload.fset"],
    )
    _core.Node.resize_inputs = original_methods["Node.resize_inputs"]
    _core.Node.prepend = original_methods["Node.prepend"]
    _core.Node.append = original_methods["Node.append"]
    _core.Node.resize_outputs = original_methods["Node.resize_outputs"]
    _core.Node.graph = property(
        _core.Node.graph.fget,
        original_methods["Node.graph.fset"],
    )

    # Value
    _core.Value.__init__ = original_methods["Value.__init__"]
    _core.Value.name = property(
        _core.Value.name.fget,
        original_methods["Value.name.fset"],
    )
    _core.Value.type = property(
        _core.Value.type.fget,
        original_methods["Value.type.fset"],
    )
    _core.Value.shape = property(
        _core.Value.shape.fget,
        original_methods["Value.shape.fset"],
    )
    _core.Value.const_value = property(
        _core.Value.const_value.fget,
        original_methods["Value.const_value.fset"],
    )
    _core.Value.replace_all_uses_with = original_methods["Value.replace_all_uses_with"]
    _core.Value.merge_shapes = original_methods["Value.merge_shapes"]

    # Graph
    _core.Graph.__init__ = original_methods["Graph.__init__"]
    _core.Graph.register_initializer = original_methods["Graph.register_initializer"]
    _core.Graph.append = original_methods["Graph.append"]
    _core.Graph.extend = original_methods["Graph.extend"]
    _core.Graph.remove = original_methods["Graph.remove"]
    _core.Graph.insert_after = original_methods["Graph.insert_after"]
    _core.Graph.insert_before = original_methods["Graph.insert_before"]
    _core.Graph.sort = original_methods["Graph.sort"]

    # Model
    _core.Model.__init__ = original_methods["Model.__init__"]

    # Function
    _core.Function.__init__ = original_methods["Function.__init__"]
    _core.Function.name = property(
        _core.Function.name.fget,
        original_methods["Function.name.fset"],
    )
    _core.Function.domain = property(
        _core.Function.domain.fget,
        original_methods["Function.domain.fset"],
    )
    _core.Function.overload = property(
        _core.Function.overload.fget,
        original_methods["Function.overload.fset"],
    )

    # Attr
    _core.Attr.__init__ = original_methods["Attr.__init__"]
