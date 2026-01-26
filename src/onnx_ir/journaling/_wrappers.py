# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Wrappers for IR classes to enable journaling.

This module provides wrapper functions that enable journaling for ONNX IR classes.
The wrappers are applied when a Journal context is active, and they record operations
to the journal for debugging and analysis purposes.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar

from typing_extensions import Concatenate, ParamSpec

from onnx_ir import _core
from onnx_ir.journaling import _journaling

_P = ParamSpec("_P")
_T = TypeVar("_T")
_SelfT = TypeVar("_SelfT")
_ValueT = TypeVar("_ValueT")


# =============================================================================
# Generic wrapper factories
# =============================================================================


def _init_wrapper(
    journal: _journaling.Journal,
    original_init: Callable[Concatenate[_SelfT, _P], None],
    *,
    details_func: Callable[[_SelfT], str | None] = lambda self: repr(self),
) -> Callable[Concatenate[_SelfT, _P], None]:
    """Generic wrapper factory for __init__ methods.

    Args:
        journal: The journal to record to.
        original_init: The original __init__ method.
        details_func: A function that takes self and returns details string.
    """

    @functools.wraps(original_init)
    def wrapper(self: _SelfT, *args: _P.args, **kwargs: _P.kwargs) -> None:
        original_init(self, *args, **kwargs)
        journal.record(self, "init", details=details_func(self))

    return wrapper


def _setter_wrapper(
    journal: _journaling.Journal,
    original_setter: Callable[[_SelfT, _ValueT], None],
    property_name: str,
    operation: str,
) -> Callable[[_SelfT, _ValueT], None]:
    """Generic wrapper factory for property setters.

    Args:
        journal: The journal to record to.
        original_setter: The original setter method.
        property_name: The private attribute name (e.g., "_name").
        operation: The operation name for the journal (e.g., "set_name").
    """

    @functools.wraps(original_setter)
    def wrapper(self: _SelfT, value: _ValueT) -> None:
        old_value = getattr(self, property_name)
        journal.record(self, operation, details=f"{old_value!r} -> {value!r}")
        original_setter(self, value)

    return wrapper


def _method_wrapper(
    journal: _journaling.Journal,
    original_method: Callable[Concatenate[_SelfT, _P], _T],
    operation: str,
    details_func: Callable[Concatenate[_SelfT, _P], str | None],
) -> Callable[Concatenate[_SelfT, _P], _T]:
    """Generic wrapper factory for methods.

    Args:
        journal: The journal to record to.
        original_method: The original method.
        operation: The operation name for the journal.
        details_func: A function that takes (self, *args, **kwargs) and returns details.
    """

    @functools.wraps(original_method)
    def wrapper(self: _SelfT, *args: _P.args, **kwargs: _P.kwargs) -> _T:
        journal.record(self, operation, details=details_func(self, *args, **kwargs))
        return original_method(self, *args, **kwargs)

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


def wrap_ir_classes(journal: _journaling.Journal) -> dict[str, Any]:
    """Wrap IR classes with journaling-enabled versions.

    This function replaces methods on IR classes with wrapped versions that
    record operations to the active journal.
    """
    original_methods = get_original_methods()

    # TensorBase
    _core.TensorBase.__init__ = _init_wrapper(
        journal, original_methods["TensorBase.__init__"], details_func=lambda self: None
    )

    # Node
    _core.Node.__init__ = _init_wrapper(journal, original_methods["Node.__init__"])
    _core.Node.name = property(
        _core.Node.name.fget,
        _setter_wrapper(journal, original_methods["Node.name.fset"], "_name", "set_name"),
    )
    _core.Node.domain = property(
        _core.Node.domain.fget,
        _setter_wrapper(
            journal, original_methods["Node.domain.fset"], "_domain", "set_domain"
        ),
    )
    _core.Node.version = property(
        _core.Node.version.fget,
        _setter_wrapper(
            journal, original_methods["Node.version.fset"], "_version", "set_version"
        ),
    )
    _core.Node.op_type = property(
        _core.Node.op_type.fget,
        _setter_wrapper(
            journal, original_methods["Node.op_type.fset"], "_op_type", "set_op_type"
        ),
    )
    _core.Node.overload = property(
        _core.Node.overload.fget,
        _setter_wrapper(
            journal, original_methods["Node.overload.fset"], "_overload", "set_overload"
        ),
    )
    _core.Node.resize_inputs = _method_wrapper(
        journal,
        original_methods["Node.resize_inputs"],
        "resize_inputs",
        lambda self, new_size: f"{len(self._inputs)} -> {new_size}",
    )
    _core.Node.prepend = _method_wrapper(
        journal,
        original_methods["Node.prepend"],
        "prepend",
        lambda self, nodes: repr(nodes),
    )
    _core.Node.append = _method_wrapper(
        journal,
        original_methods["Node.append"],
        "append",
        lambda self, nodes: repr(nodes),
    )
    _core.Node.resize_outputs = _method_wrapper(
        journal,
        original_methods["Node.resize_outputs"],
        "resize_outputs",
        lambda self, new_size: f"{len(self._outputs)} -> {new_size}",
    )
    _core.Node.graph = property(
        _core.Node.graph.fget,
        _method_wrapper(
            journal,
            original_methods["Node.graph.fset"],
            "set_graph",
            lambda self,
            value: f"{(value.name if isinstance(value, _core.Graph) else value)!r}",
        ),
    )

    # Value
    _core.Value.__init__ = _init_wrapper(journal, original_methods["Value.__init__"])
    _core.Value.name = property(
        _core.Value.name.fget,
        _setter_wrapper(journal, original_methods["Value.name.fset"], "_name", "set_name"),
    )
    _core.Value.type = property(
        _core.Value.type.fget,
        _setter_wrapper(journal, original_methods["Value.type.fset"], "_type", "set_type"),
    )
    _core.Value.shape = property(
        _core.Value.shape.fget,
        _setter_wrapper(journal, original_methods["Value.shape.fset"], "_shape", "set_shape"),
    )
    _core.Value.const_value = property(
        _core.Value.const_value.fget,
        _setter_wrapper(
            journal,
            original_methods["Value.const_value.fset"],
            "_const_value",
            "set_const_value",
        ),
    )
    _core.Value.replace_all_uses_with = _method_wrapper(
        journal,
        original_methods["Value.replace_all_uses_with"],
        "replace_all_uses_with",
        lambda self, replacement, *, replace_graph_outputs=False: (
            f"replacement={replacement!r}, replace_graph_outputs={replace_graph_outputs}"
        ),
    )
    _core.Value.merge_shapes = _method_wrapper(
        journal,
        original_methods["Value.merge_shapes"],
        "merge_shapes",
        lambda self, other: f"original={self._shape!r}, other={other!r}",
    )

    # Graph
    _core.Graph.__init__ = _init_wrapper(
        journal, original_methods["Graph.__init__"], details_func=lambda self: str(self.name)
    )
    _core.Graph.register_initializer = _method_wrapper(
        journal,
        original_methods["Graph.register_initializer"],
        "register_initializer",
        lambda self, value: repr(value),
    )
    _core.Graph.append = _method_wrapper(
        journal,
        original_methods["Graph.append"],
        "append",
        lambda self, node: repr(node),
    )
    _core.Graph.extend = _method_wrapper(
        journal,
        original_methods["Graph.extend"],
        "extend",
        lambda self, nodes: repr(nodes),
    )
    _core.Graph.remove = _method_wrapper(
        journal,
        original_methods["Graph.remove"],
        "remove",
        lambda self, nodes, safe=False: f"nodes={nodes!r}, safe={safe}",
    )
    _core.Graph.insert_after = _method_wrapper(
        journal,
        original_methods["Graph.insert_after"],
        "insert_after",
        lambda self, node, new_nodes: f"node={node!r}, new_nodes={new_nodes!r}",
    )
    _core.Graph.insert_before = _method_wrapper(
        journal,
        original_methods["Graph.insert_before"],
        "insert_before",
        lambda self, node, new_nodes: f"node={node!r}, new_nodes={new_nodes!r}",
    )
    _core.Graph.sort = _method_wrapper(
        journal,
        original_methods["Graph.sort"],
        "sort",
        lambda self: None,
    )

    # Model
    _core.Model.__init__ = _init_wrapper(journal, original_methods["Model.__init__"])

    # Function
    _core.Function.__init__ = _init_wrapper(journal, original_methods["Function.__init__"])
    _core.Function.name = property(
        _core.Function.name.fget,
        _setter_wrapper(journal, original_methods["Function.name.fset"], "_name", "set_name"),
    )
    _core.Function.domain = property(
        _core.Function.domain.fget,
        _setter_wrapper(
            journal, original_methods["Function.domain.fset"], "_domain", "set_domain"
        ),
    )
    _core.Function.overload = property(
        _core.Function.overload.fget,
        _setter_wrapper(
            journal, original_methods["Function.overload.fset"], "_overload", "set_overload"
        ),
    )

    # Attr
    _core.Attr.__init__ = _init_wrapper(journal, original_methods["Attr.__init__"])

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
