# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference context and merge policies."""

from __future__ import annotations

__all__ = [
    "ShapeInferenceContext",
    "ShapeMergePolicy",
]

import enum
import logging
from collections.abc import Mapping

import sympy

import onnx_ir as ir

logger = logging.getLogger(__name__)


class ShapeMergePolicy(enum.Enum):
    """Policy for merging inferred shapes/dtypes with existing values.

    Attributes:
        SKIP: Don't update if shape/dtype already exists.
        OVERRIDE: Always replace with inferred shape/dtype.
        REFINE: Only update if inferred is more specific
            (concrete beats symbolic, named symbolic beats None).
        STRICT: Fail if inferred shape/dtype conflicts with existing.
    """

    SKIP = "skip"
    OVERRIDE = "override"
    REFINE = "refine"
    STRICT = "strict"


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
        model: The IR model being processed.
        opset: The opset version to use for inference.
        policy: The shape merge policy.
    """

    def __init__(
        self,
        model: ir.Model,
        opset: int | None = None,
        policy: ShapeMergePolicy = ShapeMergePolicy.REFINE,
    ) -> None:
        """Initialize the shape inference context.

        Args:
            model: The IR model to perform inference on.
            opset: The opset version. If None, uses the model's default opset.
            policy: The shape merge policy to use.
        """
        self.model = model
        self._opset = opset
        self.policy = policy

        # Dimension variable bindings (symbol name -> concrete value or expression)
        self._bindings: dict[str, int | sympy.Expr] = {}

        # Track constraint violations for STRICT mode
        self._violations: list[str] = []

    @property
    def opset(self) -> int:
        """Get the opset version for inference."""
        if self._opset is not None:
            return self._opset
        # Get from model's opset imports (dict: domain -> version)
        return self.model.opset_imports.get("", 1)

    def get_opset_version(self, domain: str) -> int:
        """Get the opset version for a specific domain."""
        if domain in self.model.opset_imports:
            return self.model.opset_imports[domain]
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

        if self.policy == ShapeMergePolicy.SKIP:
            return False

        if self.policy == ShapeMergePolicy.OVERRIDE:
            value.shape = shape
            return True

        if self.policy == ShapeMergePolicy.STRICT:
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

        # REFINE policy
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

        if self.policy == ShapeMergePolicy.SKIP:
            return False

        if self.policy == ShapeMergePolicy.OVERRIDE:
            value.dtype = dtype
            return True

        if self.policy == ShapeMergePolicy.STRICT:
            if existing != dtype:
                raise ValueError(
                    f"Dtype conflict for {value.name}: existing {existing} vs inferred {dtype}"
                )
            return False

        # REFINE policy - only set if not already set (existing is not None here)
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

    @property
    def violations(self) -> list[str]:
        """Get any constraint violations encountered."""
        return self._violations.copy()
