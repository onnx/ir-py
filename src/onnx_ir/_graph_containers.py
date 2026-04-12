# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Tracked containers for graph."""

# pylint: disable=protected-access

from __future__ import annotations

__all__ = [
    "GraphInputs",
    "GraphOutputs",
]

import collections
import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import SupportsIndex, TypeVar

import onnx_ir
from onnx_ir import _core, _protocols

T = TypeVar("T")

logger = logging.getLogger(__name__)


class _GraphIO(collections.UserList["_core.Value"]):
    """The inputs and outputs of a Graph."""

    def __init__(self, graph: _core.Graph, initlist=None):
        self._graph = graph
        # Use a ref counter to track the number of references to each value
        # in the input/output list. This is used to determine when to unset the graph
        # reference in the value.
        # Even though a duplicated value is invalid in inputs and not recommended in outputs,
        # it is still possible to have duplicated inputs/outputs in an ONNX graph so we
        # need to properly handle this case and maintain the graph reference properly.
        self._ref_counter: collections.Counter[_core.Value] = collections.Counter()
        if initlist is not None:
            initlist = tuple(initlist)  # Create a copy in case initlist is a generator
            for value in initlist:
                self._set_graph(value)
        super().__init__(initlist)
        self._check_invariance()

    def _check_invariance(self) -> None:
        """Check the invariance of the graph."""
        raise NotImplementedError

    def _set_graph(self, value: _core.Value) -> None:
        """Set the graph for the value."""
        raise NotImplementedError

    def _maybe_unset_graph(self, value: _core.Value) -> None:
        """Unset the graph for the value."""
        raise NotImplementedError

    def append(self, item: _core.Value) -> None:
        """Add a new input to the graph."""
        # Perform checks first in _set_graph before modifying the data structure
        self._set_graph(item)
        super().append(item)
        self._check_invariance()

    def extend(self, other) -> None:
        """Extend the list of inputs or outputs."""
        other = tuple(other)
        for item in other:
            self._set_graph(item)
        super().extend(other)

    def insert(self, i: int, item: _core.Value) -> None:
        """Insert an input/output to the graph."""
        super().insert(i, item)
        self._set_graph(item)
        self._check_invariance()

    def pop(self, i: int = -1) -> _core.Value:
        """Remove an input/output from the graph."""
        value = super().pop(i)
        self._maybe_unset_graph(value)
        self._check_invariance()
        return value

    def remove(self, item: _core.Value) -> None:
        """Remove an input/output from the graph."""
        super().remove(item)
        self._maybe_unset_graph(item)
        self._check_invariance()

    def clear(self) -> None:
        """Clear the list."""
        for value in self.data:
            self._maybe_unset_graph(value)
        super().clear()

    def copy(self) -> list[_core.Value]:
        """Return a shallow copy of the list."""
        # This is a shallow copy, so the values are not copied, just the references
        return self.data.copy()

    def __setitem__(self, i, item) -> None:
        """Replace an input/output to the node."""
        if isinstance(item, Iterable) and isinstance(i, slice):
            # Modify a slice of the list
            for value in self.data[i]:
                self._maybe_unset_graph(value)
            for value in item:
                self._set_graph(value)
            super().__setitem__(i, item)
            self._check_invariance()
            return
        elif isinstance(i, SupportsIndex):
            # Replace a single item
            self._maybe_unset_graph(self.data[i])
            self._set_graph(item)
            super().__setitem__(i, item)
            self._check_invariance()
            return

        raise TypeError(f"Invalid types for __setitem__: {type(i)} and {type(item)}")

    def __getitem__(self, i):
        """Get an input/output from the graph."""
        return self.data[i]

    def _unimplemented(self, *_args, **_kwargs):
        """Unimplemented method."""
        raise RuntimeError("Method is not supported")

    __add__ = _unimplemented
    __radd__ = _unimplemented
    __iadd__ = _unimplemented
    __mul__ = _unimplemented
    __rmul__ = _unimplemented


class GraphInputs(_GraphIO):
    """The inputs of a Graph."""

    def _check_invariance(self) -> None:
        """Check the invariance of the graph."""
        if not onnx_ir.DEBUG:
            return
        for value in self.data:
            if value._graph is self._graph:
                continue
            raise ValueError(
                f"Invariance error: Value '{value}' is not an input of the graph: {self._graph!r}"
            )

    def _set_graph(self, value: _core.Value) -> None:
        """Set the graph for the value."""
        if value._graph is not None and value._graph is not self._graph:
            raise ValueError(
                f"Value '{value}' is already owned by a different graph. Please remove the value from the previous graph first"
            )
        if value.producer() is not None:
            raise ValueError(
                f"Value '{value}' is produced by a node and cannot be an input to the graph. Please create new Values for graph inputs"
            )
        self._ref_counter[value] += 1
        value._is_graph_input = True
        value._graph = self._graph

    def _maybe_unset_graph(self, value: _core.Value) -> None:
        """Unset the graph for the value."""
        assert value._graph is self._graph, "Bug: value does not belong to the graph"
        self._ref_counter[value] -= 1
        if self._ref_counter[value] > 0:
            # The value is still used by another graph input
            return
        value._is_graph_input = False
        if value._owned_by_graph():
            # Keep the graph reference if the value is still an input or an initializer
            return
        value._graph = None


class GraphOutputs(_GraphIO):
    """The outputs of a Graph."""

    def _check_invariance(self) -> None:
        """Check the invariance of the graph."""
        if not onnx_ir.DEBUG:
            return
        for value in self.data:
            if value._graph is self._graph:
                continue
            raise ValueError(
                f"Invariance error: Value '{value}' is not an output of the graph: {self._graph!r}"
            )

    def _set_graph(self, value: _core.Value) -> None:
        """Set the graph for the value."""
        if value._graph is not None and value._graph is not self._graph:
            raise ValueError(
                f"Value '{value}' is already an output of a different graph. Please remove the value from the previous graph first"
            )
        self._ref_counter[value] += 1
        value._is_graph_output = True
        value._graph = self._graph

    def _maybe_unset_graph(self, value: _core.Value) -> None:
        """Unset the graph for the value."""
        assert value._graph is self._graph, "Bug: value does not belong to the graph"
        self._ref_counter[value] -= 1
        if self._ref_counter[value] > 0:
            # The value is still used by another graph input
            return
        value._is_graph_output = False
        if value._owned_by_graph():
            # Keep the graph reference if the value is still an input or an initializer
            return
        value._graph = None


class GraphInitializers:
    """The initializers of a Graph, stored by identity with backward-compatible dict-like API.

    Internal storage uses ``list[Value]`` for ordered iteration and ``set[Value]``
    for O(1) identity-based membership checks (Value uses default identity-based
    ``__hash__``/``__eq__``). A lazy ``_name_cache`` (``dict[str, Value]``) is
    rebuilt on demand for name-based access.

    Primary mutation API (identity-based):
        - ``add(value)`` — add a Value by identity
        - ``remove(value)`` — remove a Value by identity (raises if absent)
        - ``discard(value)`` — remove a Value by identity if present

    Backward-compatible dict-like API (name-based):
        - ``__getitem__(name)``, ``__setitem__(name, value)``, ``__delitem__(name)``
        - ``__contains__(key)`` — accepts both ``str`` (name) and ``Value`` (identity)
        - ``__iter__``, ``keys()``, ``values()``, ``items()``, ``__len__``
        - ``pop(name)``, ``get(name)``, ``update(dict)``
        - ``clear()``
    """

    def __init__(self, graph: _core.Graph, dict=None, /, **kwargs):
        self._graph = graph
        self._values: list[_core.Value] = []
        self._identity_set: set[_core.Value] = set()
        self._name_cache: dict[str, _core.Value] | None = None

        # Collect initial data from dict and kwargs
        initial_values: list[_core.Value] = []
        if dict is not None:
            if isinstance(dict, Iterable) and not isinstance(dict, Mapping):
                raise TypeError(
                    f"Expected a mapping, not {type(dict)}"
                )
            initial_values.extend(dict.values() if isinstance(dict, Mapping) else dict)
        if kwargs:
            initial_values.extend(kwargs.values())

        for value in initial_values:
            self._maybe_set_graph(value)
            if value not in self._identity_set:
                self._values.append(value)
                self._identity_set.add(value)

    def _invalidate_name_cache(self) -> None:
        """Invalidate the lazy name cache. Must be called on any mutation."""
        self._name_cache = None

    def _build_name_cache(self) -> dict[str, _core.Value]:
        """Build and return the name cache from current values."""
        if self._name_cache is None:
            cache: dict[str, _core.Value] = {}
            for value in self._values:
                if value.name:
                    cache[value.name] = value
            self._name_cache = cache
        return self._name_cache

    def _maybe_set_graph(self, value: _core.Value) -> None:
        """Set the graph for the value."""
        if value._graph is not None and value._graph is not self._graph:
            raise ValueError(
                f"Value '{value}' is already an initializer of a different graph. Please remove the value from the previous graph first"
            )
        value._is_initializer = True
        value._graph = self._graph

    def _maybe_unset_graph(self, value: _core.Value) -> None:
        """Unset the graph for the value."""
        assert value._graph is self._graph, "Bug: value does not belong to the graph"
        value._is_initializer = False
        if value._owned_by_graph():
            return
        value._graph = None

    # === Identity-based mutation API ===

    def add(self, value: _core.Value) -> None:
        """Add an initializer to the graph by identity.

        If the value is already present (by identity), this is a no-op.
        For backward compatibility, this validates the value's name via ``__setitem__``.
        """
        # Delegate to __setitem__ for full validation (name checks, type checks)
        self[value.name] = value  # type: ignore[index]

    def remove(self, value: _core.Value) -> None:
        """Remove an initializer by identity. Raises ValueError if not present."""
        if value not in self._identity_set:
            raise ValueError(f"Value '{value}' is not an initializer of this graph")
        self._maybe_unset_graph(value)
        self._identity_set.discard(value)
        self._values.remove(value)
        self._invalidate_name_cache()

    def discard(self, value: _core.Value) -> None:
        """Remove an initializer by identity if present. No-op if not present."""
        if value not in self._identity_set:
            return
        self._maybe_unset_graph(value)
        self._identity_set.discard(value)
        self._values.remove(value)
        self._invalidate_name_cache()

    # === Backward-compatible dict-like API ===

    def __setitem__(self, key: str, value: _core.Value) -> None:
        """Set an initializer for the graph."""
        if not isinstance(value, _core.Value):
            raise TypeError(f"value must be a Value object, not {type(value)}")
        if not isinstance(key, str):
            raise TypeError(f"Value name must be a string, not {type(key)}")
        if key == "":
            raise ValueError("Value name cannot be an empty string")
        if not value.name:
            logger.info("Value %s does not have a name, setting it to '%s'", value, key)
            value.name = key
        elif key != value.name:
            raise ValueError(
                f"Key '{key}' does not match the name of the value '{value.name}'. Please use the value.name as the key."
            )
        if value.producer() is not None:
            raise ValueError(
                f"Value '{value}' is produced by a node and cannot be a graph initializer"
            )
        # If a different value with the same name already exists, remove it
        cache = self._build_name_cache()
        if key in cache:
            old_value = cache[key]
            if old_value is not value:
                self._maybe_unset_graph(old_value)
                self._identity_set.discard(old_value)
                self._values.remove(old_value)

        # Add the new value by identity (no-op if already present)
        if value not in self._identity_set:
            self._maybe_set_graph(value)
            self._values.append(value)
            self._identity_set.add(value)
        else:
            # Value already present by identity — ensure graph is set
            self._maybe_set_graph(value)
        self._invalidate_name_cache()

    def __delitem__(self, key: str) -> None:
        """Delete an initializer from the graph by name."""
        cache = self._build_name_cache()
        if key not in cache:
            raise KeyError(key)
        value = cache[key]
        self._maybe_unset_graph(value)
        self._identity_set.discard(value)
        self._values.remove(value)
        self._invalidate_name_cache()

    def __getitem__(self, key: str) -> _core.Value:
        """Get an initializer by name."""
        cache = self._build_name_cache()
        if key not in cache:
            raise KeyError(key)
        return cache[key]

    def __contains__(self, key: object) -> bool:
        """Check membership. Accepts str (name lookup) or Value (identity check)."""
        if isinstance(key, _core.Value):
            return key in self._identity_set
        if isinstance(key, str):
            return key in self._build_name_cache()
        return False

    def __iter__(self):
        """Iterate over initializer names (for backward compat with dict interface)."""
        return iter(self._build_name_cache())

    def __len__(self) -> int:
        return len(self._values)

    def __bool__(self) -> bool:
        return len(self._values) > 0

    def __repr__(self) -> str:
        return f"GraphInitializers({self._build_name_cache()!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, GraphInitializers):
            return self._values == other._values
        if isinstance(other, dict):
            return self._build_name_cache() == other
        return NotImplemented

    def keys(self):
        """Return initializer names."""
        return self._build_name_cache().keys()

    def values(self):
        """Return initializer Values in insertion order."""
        return _GraphInitializerValues(self._values)

    def items(self):
        """Return (name, Value) pairs."""
        return self._build_name_cache().items()

    def get(self, key: str, default=None):
        """Get an initializer by name, with optional default."""
        cache = self._build_name_cache()
        return cache.get(key, default)

    def pop(self, key: str, *args):
        """Remove and return an initializer by name."""
        cache = self._build_name_cache()
        if key not in cache:
            if args:
                return args[0]
            raise KeyError(key)
        value = cache[key]
        self._maybe_unset_graph(value)
        self._identity_set.discard(value)
        self._values.remove(value)
        self._invalidate_name_cache()
        return value

    def update(self, dict=None, /, **kwargs):
        """Update initializers from a dict and/or keyword arguments."""
        if dict is not None:
            if isinstance(dict, Mapping):
                for key, value in dict.items():
                    self[key] = value
            elif isinstance(dict, Iterable):
                for key, value in dict:
                    self[key] = value
            else:
                raise TypeError(f"Expected a mapping or iterable, not {type(dict)}")
        for key, value in kwargs.items():
            self[key] = value

    def clear(self) -> None:
        """Remove all initializers from the graph."""
        for value in list(self._values):
            self._maybe_unset_graph(value)
        self._values.clear()
        self._identity_set.clear()
        self._invalidate_name_cache()

    def setdefault(self, key: str, default: _core.Value | None = None) -> _core.Value:
        """Get initializer by name; if absent, set and return default."""
        cache = self._build_name_cache()
        if key in cache:
            return cache[key]
        if default is None:
            raise KeyError(key)
        self[key] = default
        return default


class _GraphInitializerValues:
    """A view over the initializer Values list, behaving like dict_values."""

    def __init__(self, values: list[_core.Value]):
        self._values = values

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __contains__(self, item: object) -> bool:
        if isinstance(item, _core.Value):
            return item in self._values
        return False

    def __repr__(self) -> str:
        return f"GraphInitializerValues({list(self._values)!r})"


class Attributes(collections.UserDict[str, "_core.Attr"]):
    """The attributes of a Node as ``dict[str, Attr]`` with additional access methods."""

    def __init__(
        self, attrs: Iterable[_core.Attr], owner: _core.Node | _core.Function
    ) -> None:
        self._owner = owner
        super().__init__({attr.name: attr for attr in attrs})

    def __setitem__(self, key: str, value: _core.Attr) -> None:
        """Set an attribute for the node."""
        if type(key) is not str:
            raise TypeError(f"Key must be a string, not {type(key)}")
        if not isinstance(value, _core.Attr):
            raise TypeError(f"Value must be an Attr, not {type(value)}")
        super().__setitem__(key, value)

    def add(self, value: _core.Attr) -> None:
        """Add an attribute to the node."""
        self[value.name] = value

    def get_int(self, key: str, default: T = None) -> int | T:  # type: ignore[assignment]
        """Get the integer value of the attribute."""
        if key in self:
            return self[key].as_int()
        return default

    def get_float(self, key: str, default: T = None) -> float | T:  # type: ignore[assignment]
        """Get the float value of the attribute."""
        if key in self:
            return self[key].as_float()
        return default

    def get_string(self, key: str, default: T = None) -> str | T:  # type: ignore[assignment]
        """Get the string value of the attribute."""
        if key in self:
            return self[key].as_string()
        return default

    def get_tensor(self, key: str, default: T = None) -> _protocols.TensorProtocol | T:  # type: ignore[assignment]
        """Get the tensor value of the attribute."""
        if key in self:
            return self[key].as_tensor()
        return default

    def get_graph(self, key: str, default: T = None) -> _core.Graph | T:  # type: ignore[assignment]
        """Get the graph value of the attribute."""
        if key in self:
            return self[key].as_graph()
        return default

    def get_ints(self, key: str, default: T = None) -> Sequence[int] | T:  # type: ignore[assignment]
        """Get the Sequence of integers from the attribute."""
        if key in self:
            return self[key].as_ints()
        return default

    def get_floats(self, key: str, default: T = None) -> Sequence[float] | T:  # type: ignore[assignment]
        """Get the Sequence of floats from the attribute."""
        if key in self:
            return self[key].as_floats()
        return default

    def get_strings(self, key: str, default: T = None) -> Sequence[str] | T:  # type: ignore[assignment]
        """Get the Sequence of strings from the attribute."""
        if key in self:
            return self[key].as_strings()
        return default

    def get_tensors(
        self,
        key: str,
        default: T = None,  # type: ignore[assignment]
    ) -> Sequence[_protocols.TensorProtocol] | T:
        """Get the Sequence of tensors from the attribute."""
        if key in self:
            return self[key].as_tensors()
        return default

    def get_graphs(self, key: str, default: T = None) -> Sequence[_core.Graph] | T:  # type: ignore[assignment]
        """Get the Sequence of graphs from the attribute."""
        if key in self:
            return self[key].as_graphs()
        return default
