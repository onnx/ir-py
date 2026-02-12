# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference context and merge policies."""

from __future__ import annotations

__all__ = [
    "OpUsageError",
    "ShapeInferenceContext",
    "ShapeInferenceError",
    "ShapeMergePolicy",
]

import logging
from collections.abc import Mapping, Sequence
from typing import Literal

import onnx_ir as ir

logger = logging.getLogger(__name__)


class ShapeInferenceError(ValueError):
    """A recorded error from shape inference.

    Can be raised directly (it is a :class:`ValueError` subclass) or stored
    for later inspection via :attr:`ShapeInferenceContext.errors`.

    Attributes:
        node_name: The name of the node (or ``None`` if unnamed).
        op_type: The operator type (e.g. ``"Add"``).
        domain: The operator domain.
        message: Human-readable description of the error.
    """

    def __init__(
        self,
        *,
        node_name: str | None,
        op_type: str,
        domain: str,
        message: str,
    ) -> None:
        self.node_name = node_name
        self.op_type = op_type
        self.domain = domain
        self.message = message
        super().__init__(str(self))

    def __str__(self) -> str:
        op_id = f"{self.domain}::{self.op_type}" if self.domain else self.op_type
        node_desc = f" (node {self.node_name!r})" if self.node_name else ""
        return f"{op_id}{node_desc}: {self.message}"


class OpUsageError(ValueError):
    """Raised when an operator node has invalid structure.

    This indicates the model is malformed: wrong number of inputs, missing
    required inputs, or missing required attributes.
    """

    def __init__(self, node: ir.Node, message: str) -> None:
        self.node = node
        self.message = message
        super().__init__(str(self))

    def __str__(self) -> str:
        op_id = (
            f"{self.node.domain}::{self.node.op_type}"
            if self.node.domain
            else self.node.op_type
        )
        node_desc = f" (node {self.node.name!r})" if self.node.name else ""
        return f"{op_id}{node_desc}: {self.message}"


def check_inputs(node: ir.Node, *names: str) -> tuple[ir.Value, ...]:
    """Validate that required positional inputs exist and are not None.

    Args:
        node: The node to validate.
        *names: Names of required inputs, in positional order.

    Returns:
        A tuple of :class:`ir.Value` for each required input.

    Raises:
        OpUsageError: If the node has fewer inputs than required or
            any required input is ``None``.
    """
    min_count = len(names)
    if len(node.inputs) < min_count:
        raise OpUsageError(
            node,
            f"Expected at least {min_count} input(s), got {len(node.inputs)}",
        )
    values: list[ir.Value] = []
    for i, name in enumerate(names):
        v = node.inputs[i]
        if v is None:
            raise OpUsageError(
                node,
                f"Required input '{name}' (#{i}) is None",
            )
        values.append(v)
    return tuple(values)


def require_attr(node: ir.Node, name: str) -> ir.Attr:
    """Get a required attribute, raising :class:`OpUsageError` if missing.

    Args:
        node: The node to read from.
        name: Attribute name.

    Returns:
        The attribute.

    Raises:
        OpUsageError: If the attribute does not exist.
    """
    attr = node.attributes.get(name)
    if attr is None:
        raise OpUsageError(node, f"Missing required attribute '{name}'")
    return attr


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

    Tracks constraints, and provides utilities for shape inference functions.

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

        # Recorded errors from shape inference
        self._errors: list[ShapeInferenceError] = []
        # Counter for generating unique symbolic dimension names
        self._dim_counter: int = 0

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

    def new_symbolic_dim(self) -> ir.SymbolicDim:
        """Create a new symbolic dimension with a unique auto-generated name.

        Use this instead of ``ir.SymbolicDim(None)`` so that each unknown
        dimension gets a distinct identity.  Subsequent inference steps can
        then establish relationships between these named dimensions.

        Returns:
            A :class:`ir.SymbolicDim` with a unique name like ``_d0``, ``_d1``, …
        """
        name = f"_d{self._dim_counter}"
        self._dim_counter += 1
        return ir.SymbolicDim(name)

    def name_anonymous_dims(self, value: ir.Value) -> bool:
        """Replace anonymous (``None``) symbolic dims on *value* with unique names.

        This mutates the value's shape in-place.  It is a no-op when the value
        has no shape or all dims are already named/concrete.

        Returns:
            ``True`` if any dim was renamed.
        """
        shape = value.shape
        if shape is None:
            return False

        new_dims: list[int | ir.SymbolicDim] = []
        changed = False
        for dim in shape.dims:
            if isinstance(dim, ir.SymbolicDim) and dim.value is None:
                new_dims.append(self.new_symbolic_dim())
                changed = True
            else:
                new_dims.append(dim)

        if changed:
            value.shape = ir.Shape(new_dims)
        return changed

    def record_error(self, node: ir.Node, message: str) -> None:
        """Record a shape inference error for a node.

        The error is raised immediately unless the merge policy is ``"skip"``,
        in which case it is only logged and appended to :attr:`errors`.

        Args:
            node: The node that caused the error.
            message: Human-readable description of the problem.

        Raises:
            ShapeInferenceError: If the merge policy is not ``"skip"``.
        """
        error = ShapeInferenceError(
            node_name=node.name,
            op_type=node.op_type,
            domain=node.domain,
            message=message,
        )
        self._errors.append(error)
        if self.policy == "skip":
            logger.warning("Shape inference error: %s", error)
            return
        raise error

    @property
    def errors(self) -> Sequence[ShapeInferenceError]:
        """All errors recorded during shape inference."""
        return self._errors

    @staticmethod
    def _check_no_anonymous_dims(shape: ir.Shape) -> None:
        """Raise if *shape* contains any anonymous (``None``) symbolic dims.

        All unknown dimensions must be given a unique name via
        :meth:`new_symbolic_dim` so that downstream inference can establish
        relationships between dimensions.
        """
        for i, dim in enumerate(shape.dims):
            if isinstance(dim, ir.SymbolicDim) and dim.value is None:
                raise ValueError(
                    f"Shape dim {i} is an anonymous SymbolicDim(None). "
                    "Use ctx.new_symbolic_dim() to create a uniquely named dim."
                )

    def set_shape(self, value: ir.Value, shape: ir.Shape) -> bool:
        """Set the shape of a value according to the merge policy.

        Args:
            value: The value to set the shape on.
            shape: The inferred shape.

        Returns:
            True if the shape was updated, False otherwise.

        Raises:
            ValueError: If policy is STRICT and shapes conflict, or if
                the shape contains anonymous (None) symbolic dims.
        """
        self._check_no_anonymous_dims(shape)

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

    def set_type(self, value: ir.Value, type_: ir.TypeProtocol) -> bool:
        """Set the full type of a value according to the merge policy.

        Use this for non-tensor types like :class:`ir.SequenceType` and
        :class:`ir.OptionalType`, where :meth:`set_shape_and_dtype` is not
        appropriate.

        Args:
            value: The value to update.
            type_: The inferred type.

        Returns:
            True if the type was updated, False otherwise.

        Raises:
            ValueError: If policy is ``"strict"`` and types conflict.
        """
        existing = value.type

        if existing is None:
            value.type = type_
            return True

        if self.policy == "skip":
            return False

        if self.policy == "override":
            value.type = type_
            return True

        if self.policy == "strict":
            if type(existing) is not type(type_):
                raise ValueError(
                    f"Type kind mismatch for {value.name}: "
                    f"existing {type(existing).__name__} vs inferred {type(type_).__name__}"
                )
            if existing != type_:
                raise ValueError(
                    f"Type conflict for {value.name}: "
                    f"existing {existing!r} vs inferred {type_!r}"
                )
            return False

        # "refine" policy — only set if not already present
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

    # --- Partial data propagation ---

    def set_symbolic_value(
        self,
        value: ir.Value,
        data: list[int | ir.SymbolicDim],
    ) -> None:
        """Store symbolic element values on a value's metadata.

        This is used to track the known contents of 1-D integer tensors
        (e.g., shape tensors) so that downstream ops like Reshape can read
        them even when the tensor is not a constant.

        Args:
            value: The value to annotate.
            data: A list of known element values (int or SymbolicDim).
        """
        value.meta["symbolic_value"] = data

    def get_symbolic_value(
        self,
        value: ir.Value,
    ) -> list[int | ir.SymbolicDim] | None:
        """Retrieve symbolic element values from a value.

        Checks ``value.meta["symbolic_value"]`` first.  If not set, falls back
        to reading a constant tensor via ``ir.convenience.get_const_tensor()``
        so that initializers and Constant outputs participate in propagation.

        Args:
            value: The value to query.

        Returns:
            A list of ``int | ir.SymbolicDim``, or ``None`` if unavailable.
        """
        sym_val = value.meta.get("symbolic_value")
        if sym_val is not None:
            return sym_val  # type: ignore[return-value]

        # Fall back to constant tensor (only for 0d/1d integer types used in
        # shape computation — higher-rank tensors are data, not shape values)
        const = ir.convenience.get_const_tensor(value)
        if (
            const is not None
            and len(const.shape) <= 1
            and value.dtype in (
                ir.DataType.INT64,
                ir.DataType.INT32,
                ir.DataType.INT16,
                ir.DataType.INT8,
                ir.DataType.UINT64,
                ir.DataType.UINT32,
                ir.DataType.UINT16,
                ir.DataType.UINT8,
            )
        ):
            return [int(x) for x in const.numpy().flatten()]
        return None
