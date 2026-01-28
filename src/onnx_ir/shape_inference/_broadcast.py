# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Broadcasting utilities for shape inference."""

from __future__ import annotations

__all__ = [
    "broadcast_shapes",
]

import onnx_ir as ir


def broadcast_shapes(
    shape1: ir.Shape | None,
    shape2: ir.Shape | None,
) -> ir.Shape | None:
    """Compute the broadcast shape of two shapes following NumPy broadcasting rules.

    Broadcasting rules:
    1. If shapes have different ranks, prepend 1s to the shorter shape.
    2. For each dimension pair:
       - If both are equal, use that value.
       - If one is 1, use the other value.
       - If one is symbolic and one is 1, use the symbolic value.
       - If both are symbolic and equal, use that value.
       - If both are different non-1 values, broadcasting fails.

    Args:
        shape1: First shape (can be None if unknown).
        shape2: Second shape (can be None if unknown).

    Returns:
        The broadcasted shape, or None if shapes are incompatible or unknown.

    Example::

        >>> import onnx_ir as ir
        >>> s1 = ir.Shape([3, 1, 5])
        >>> s2 = ir.Shape([1, 4, 5])
        >>> broadcast_shapes(s1, s2)
        Shape([3, 4, 5])

        >>> s1 = ir.Shape(["batch", 1, 256])
        >>> s2 = ir.Shape([1, "seq_len", 256])
        >>> broadcast_shapes(s1, s2)
        Shape([SymbolicDim('batch'), SymbolicDim('seq_len'), 256])
    """
    if shape1 is None or shape2 is None:
        return None

    # Get dimensions, padding shorter shape with 1s on the left
    dims1 = list(shape1.dims)
    dims2 = list(shape2.dims)

    max_rank = max(len(dims1), len(dims2))

    # Pad with 1s on the left
    while len(dims1) < max_rank:
        dims1.insert(0, 1)
    while len(dims2) < max_rank:
        dims2.insert(0, 1)

    result_dims: list[int | ir.SymbolicDim] = []

    for d1, d2 in zip(dims1, dims2):
        broadcasted = _broadcast_dim(d1, d2)
        if broadcasted is None:
            return None  # Broadcasting failed
        result_dims.append(broadcasted)

    return ir.Shape(result_dims)


def _broadcast_dim(
    dim1: int | ir.SymbolicDim,
    dim2: int | ir.SymbolicDim,
) -> int | ir.SymbolicDim | None:
    """Broadcast two dimensions.

    Returns the broadcasted dimension, or None if incompatible.
    """
    # Both concrete integers
    if isinstance(dim1, int) and isinstance(dim2, int):
        if dim1 == dim2:
            return dim1
        if dim1 == 1:
            return dim2
        if dim2 == 1:
            return dim1
        # Different non-1 values - incompatible
        return None

    # One is concrete 1 - broadcasts to the other
    if dim1 == 1:
        return dim2
    if dim2 == 1:
        return dim1

    # One concrete, one symbolic
    if isinstance(dim1, int) and isinstance(dim2, ir.SymbolicDim):
        # Concrete non-1 wins over symbolic if we assume they're compatible
        return dim1
    if isinstance(dim2, int) and isinstance(dim1, ir.SymbolicDim):
        return dim2

    # Both symbolic
    if isinstance(dim1, ir.SymbolicDim) and isinstance(dim2, ir.SymbolicDim):
        # Check if they're equal (same expression)
        if dim1.expr is not None and dim2.expr is not None:
            if dim1.expr == dim2.expr:
                return dim1
        # If one is unknown (None), use the other
        if dim1.value is None:
            return dim2
        if dim2.value is None:
            return dim1
        # Both are named but different - assume compatible, pick first
        # This is a simplification; in strict mode we might want to fail
        return dim1

    # Fallback: return first dim (assume compatible)
    return dim1
