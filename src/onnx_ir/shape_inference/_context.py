# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference context and merge policies."""

from __future__ import annotations

__all__ = [
    "ShapeInferenceContext",
    "ShapeInferenceError",
    "ShapeMergePolicy",
]

import dataclasses
import logging
from collections.abc import Mapping, Sequence
from typing import Literal

import sympy

import onnx_ir as ir

logger = logging.getLogger(__name__)

@dataclasses.dataclass(frozen=True)
class ShapeInferenceError:
    """A recorded error from shape inference.

    Attributes:
        node_name: The name of the node (or ``None`` if unnamed).
        op_type: The operator type (e.g. ``"Add"``).
        domain: The operator domain.
        message: Human-readable description of the error.
    """

    node_name: str | None
    op_type: str
    domain: str
    message: str

    def __str__(self) -> str:
        op_id = f"{self.domain}::{self.op_type}" if self.domain else self.op_type
        node_desc = f" (node {self.node_name!r})" if self.node_name else ""
        return f"{op_id}{node_desc}: {self.message}"


ShapeMergePolicy = Literal["skip", "override", "refine", "strict"]
"""Policy for merging inferred shapes/dtypes with existing values.

* ``"skip"``: Don't update if shape/dtype already exists.
* ``"override"``: Always replace with inferred shape/dtype.
* ``"refine"``: Only update if inferred is more specific
    (concrete beats symbolic, named symbolic beats None).
* ``"strict"``: Fail if inferred shape/dtype conflicts with existing.
"""


def _is_more_specific(
    inferred_dim: int | ir.SymbolicDim,
    existing_dim: int | ir.SymbolicDim,
) -> bool:
    """Check if the inferred dimension is more specific than the existing one.

    Specificity order: concrete int > named symbolic > unknown (None)
    """
    # Concrete int is most specific
    if isinstance(inferred_dim, int):
        return not isinstance(existing_dim, int)

    # Named symbolic is more specific than unknown
    if isinstance(inferred_dim, ir.SymbolicDim) and inferred_dim.value is not None:
        if isinstance(existing_dim, ir.SymbolicDim) and existing_dim.value is None:
            return True

    return False


def _dims_conflict(
    dim1: int | ir.SymbolicDim,
    dim2: int | ir.SymbolicDim,
) -> bool:
    """Check if two dimensions conflict (both concrete but different values)."""
    if isinstance(dim1, int) and isinstance(dim2, int):
        return dim1 != dim2
    return False


class ShapeInferenceContext:
    """Context for shape and type inference operations.

    Tracks dimension bindings, constraints, and provides utilities for
    shape inference functions.

    Attributes:
        opset_imports: Mapping from domain to opset version.
        policy: The shape merge policy.
    """

    def __init__(
        self,
        opset_imports: Mapping[str, int] | None = None,
        policy: ShapeMergePolicy = "refine",
    ) -> None:
        """Initialize the shape inference context.

        Args:
            opset_imports: Mapping from ONNX domain to opset version
                (e.g. ``{"": 17}``).  When ``None``, defaults to ``{"": 1}``.
            policy: The shape merge policy to use.
        """
        self.opset_imports: Mapping[str, int] = opset_imports or {"": 1}
        self.policy = policy

        # Dimension variable bindings (symbol name -> concrete value or expression)
        self._bindings: dict[str, int | sympy.Expr] = {}
        # Recorded errors from shape inference
        self._errors: list[ShapeInferenceError] = []

    @property
    def opset(self) -> int:
        """Get the default opset version for inference."""
        return self.opset_imports.get("", 1)

    def get_opset_version(self, domain: str) -> int:
        """Get the opset version for a specific domain."""
        if domain in self.opset_imports:
            return self.opset_imports[domain]
        if domain in ("", "ai.onnx"):
            return self.opset
        return 1

    def bind(self, symbol: str, value: int | sympy.Expr) -> None:
        """Bind a symbol to a concrete value or expression.

        Args:
            symbol: The symbol name.
            value: The concrete value or expression to bind.
        """
        if symbol in self._bindings:
            existing = self._bindings[symbol]
            if existing != value:
                logger.warning(
                    "Symbol %s already bound to %s, rebinding to %s",
                    symbol,
                    existing,
                    value,
                )
        self._bindings[symbol] = value

    def get_binding(self, symbol: str) -> int | sympy.Expr | None:
        """Get the binding for a symbol."""
        return self._bindings.get(symbol)

    @property
    def bindings(self) -> Mapping[str, int | sympy.Expr]:
        """Get all current bindings."""
        return self._bindings

    def record_error(self, node: ir.Node, message: str) -> None:
        """Record a shape inference error for a node.

        In strict mode the error is raised immediately as a :class:`ValueError`.
        Otherwise it is appended to an internal list that can be inspected via
        :attr:`errors` after the pass completes.

        Args:
            node: The node that caused the error.
            message: Human-readable description of the problem.

        Raises:
            ValueError: If the merge policy is ``"strict"``.
        """
        error = ShapeInferenceError(
            node_name=node.name,
            op_type=node.op_type,
            domain=node.domain,
            message=message,
        )
        self._errors.append(error)
        if self.policy == "strict":
            raise ValueError(str(error))
        logger.warning("Shape inference error: %s", error)

    @property
    def errors(self) -> Sequence[ShapeInferenceError]:
        """All errors recorded during shape inference."""
        return self._errors

    def set_shape(self, value: ir.Value, shape: ir.Shape) -> bool:
        """Set the shape of a value according to the merge policy.

        Args:
            value: The value to set the shape on.
            shape: The inferred shape.

        Returns:
            True if the shape was updated, False otherwise.

        Raises:
            ValueError: If policy is STRICT and shapes conflict.
        """
        existing = value.shape

        if existing is None:
            value.shape = shape
            return True

        if self.policy == "skip":
            return False

        if self.policy == "override":
            value.shape = shape
            return True

        if self.policy == "strict":
            # Check for conflicts
            if existing.rank() != shape.rank():
                raise ValueError(
                    f"Shape rank mismatch for {value.name}: "
                    f"existing {existing.rank()} vs inferred {shape.rank()}"
                )
            for i, (e_dim, i_dim) in enumerate(zip(existing.dims, shape.dims)):
                if _dims_conflict(e_dim, i_dim):
                    raise ValueError(
                        f"Shape conflict for {value.name} at dim {i}: "
                        f"existing {e_dim} vs inferred {i_dim}"
                    )
            # No conflicts, merge by taking more specific
            return self._refine_shape(value, existing, shape)

        # "refine" policy
        return self._refine_shape(value, existing, shape)

    def _refine_shape(self, value: ir.Value, existing: ir.Shape, inferred: ir.Shape) -> bool:
        """Refine existing shape with inferred shape, keeping more specific dims."""
        if existing.rank() != inferred.rank():
            # Can't refine if ranks differ; keep existing
            return False

        modified = False
        new_dims: list[int | ir.SymbolicDim] = []

        for e_dim, i_dim in zip(existing.dims, inferred.dims):
            if _is_more_specific(i_dim, e_dim):
                new_dims.append(i_dim)
                modified = True
            else:
                new_dims.append(e_dim)

        if modified:
            value.shape = ir.Shape(new_dims)

        return modified

    def set_dtype(self, value: ir.Value, dtype: ir.DataType) -> bool:
        """Set the dtype of a value according to the merge policy.

        Args:
            value: The value to set the dtype on.
            dtype: The inferred dtype.

        Returns:
            True if the dtype was updated, False otherwise.

        Raises:
            ValueError: If policy is STRICT and dtypes conflict.
        """
        existing = value.dtype

        if existing is None:
            value.dtype = dtype
            return True

        if self.policy == "skip":
            return False

        if self.policy == "override":
            value.dtype = dtype
            return True

        if self.policy == "strict":
            if existing != dtype:
                raise ValueError(
                    f"Dtype conflict for {value.name}: existing {existing} vs inferred {dtype}"
                )
            return False

        # "refine" policy - only set if not already set (existing is not None here)
        return False

    def set_shape_and_dtype(
        self,
        value: ir.Value,
        shape: ir.Shape | None = None,
        dtype: ir.DataType | None = None,
    ) -> bool:
        """Set both shape and dtype of a value.

        Convenience method to set both at once.

        Args:
            value: The value to update.
            shape: The inferred shape (or None to skip).
            dtype: The inferred dtype (or None to skip).

        Returns:
            True if either shape or dtype was updated.
        """
        modified = False
        if shape is not None:
            modified = self.set_shape(value, shape) or modified
        if dtype is not None:
            modified = self.set_dtype(value, dtype) or modified
        return modified
