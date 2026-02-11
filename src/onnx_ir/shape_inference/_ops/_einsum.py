# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for Einsum operator."""

from __future__ import annotations

__all__ = [
    "infer_einsum",
]

import onnx_ir as ir
from onnx_ir.shape_inference import _context, _registry


@_registry.registry.register("", "Einsum", since_version=12)
def infer_einsum(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Einsum operator.

    Parses the equation string to determine output shape from input shapes.
    Supports explicit (``->`` present) and implicit output forms, and ellipsis (``...``).
    """
    if len(node.inputs) < 1 or node.inputs[0] is None:
        raise _context.OpUsageError(node, "Expected at least 1 input")

    output_dtype = node.inputs[0].dtype

    equation_attr = node.attributes.get("equation")
    if equation_attr is None:
        raise _context.OpUsageError(node, "Missing required attribute 'equation'")
    equation = equation_attr.as_string().replace(" ", "")

    # Check all inputs have shapes
    for inp in node.inputs:
        if inp is None or inp.shape is None:
            if len(node.outputs) > 0:
                ctx.set_shape_and_dtype(node.outputs[0], None, output_dtype)
            return

    output_shape = _einsum_shape(ctx, node, equation)
    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, output_dtype)


def _einsum_shape(
    ctx: _context.ShapeInferenceContext,
    node: ir.Node,
    equation: str,
) -> ir.Shape:
    """Compute the output shape for an Einsum equation given input shapes."""

    def _is_letter(c: str) -> bool:
        return c.isalpha()

    mid_index = equation.find("->")
    left_equation = equation[:mid_index] if mid_index != -1 else equation

    # label_dims: maps each unique label char → the dimension it represents
    label_dims: dict[str, int | ir.SymbolicDim] = {}
    repeated_labels: set[str] = set()
    # Ordered list of unique labels (insertion order)
    label_order: list[str] = []

    ellipsis_dims: list[int | ir.SymbolicDim] = []
    num_ellipsis = 0
    num_ellipsis_indices = 0

    terms = left_equation.split(",")
    num_operands = len(terms)

    if num_operands != len(node.inputs):
        raise _context.OpUsageError(
            node,
            f"Number of inputs ({len(node.inputs)}) does not match "
            f"operands in equation ({num_operands})",
        )

    for operand_idx, term in enumerate(terms):
        inp = node.inputs[operand_idx]
        assert inp is not None and inp.shape is not None  # guaranteed by caller
        shape = inp.shape
        rank = shape.rank()

        ellipsis_pos = term.find("...")
        # Count letter indices in the term
        term_letters = sum(1 for c in term if _is_letter(c))

        if ellipsis_pos != -1:
            if rank < term_letters:
                raise _context.OpUsageError(
                    node,
                    f"Ellipsis in operand {operand_idx}: rank {rank} < "
                    f"letter indices {term_letters}",
                )
            local_ellipsis_dims = rank - term_letters

            if num_ellipsis == 0:
                num_ellipsis_indices = local_ellipsis_dims
            elif num_ellipsis_indices != local_ellipsis_dims:
                raise _context.OpUsageError(
                    node, "Ellipsis represents incompatible dimensions"
                )
            num_ellipsis += 1
        else:
            if rank != term_letters:
                raise _context.OpUsageError(
                    node,
                    f"Rank of input {operand_idx} ({rank}) does not match "
                    f"equation indices ({term_letters})",
                )

        # Walk through the term, mapping labels to dims
        shape_idx = 0  # index into the input shape
        i = 0
        while i < len(term):
            if i == ellipsis_pos:
                # Record ellipsis dims
                local_ellipsis_count = rank - term_letters
                if len(ellipsis_dims) == 0:
                    # First time seeing ellipsis — record dims
                    for j in range(local_ellipsis_count):
                        ellipsis_dims.append(shape[shape_idx + j])
                else:
                    # Broadcast: pick the larger of the two
                    for j in range(local_ellipsis_count):
                        existing = ellipsis_dims[j]
                        current = shape[shape_idx + j]
                        if isinstance(existing, int) and isinstance(current, int):
                            if existing == 1:
                                ellipsis_dims[j] = current
                            elif current != 1 and current != existing:
                                ellipsis_dims[j] = ctx.new_symbolic_dim()
                        elif existing == current:
                            pass
                        else:
                            # One is symbolic — keep the non-1 one or create new
                            if isinstance(existing, int) and existing == 1:
                                ellipsis_dims[j] = current
                            elif isinstance(current, int) and current == 1:
                                pass  # keep existing
                            else:
                                ellipsis_dims[j] = ctx.new_symbolic_dim()
                shape_idx += local_ellipsis_count
                i += 3  # skip "..."
                continue

            c = term[i]
            if _is_letter(c):
                dim = shape[shape_idx]
                if c not in label_dims:
                    label_dims[c] = dim
                    label_order.append(c)
                else:
                    repeated_labels.add(c)
                shape_idx += 1
            i += 1

    # Build output shape
    output_dims: list[int | ir.SymbolicDim] = []

    if mid_index != -1:
        # Explicit output
        right_equation = equation[mid_index + 2 :]
        right_ellipsis_pos = right_equation.find("...")

        i = 0
        while i < len(right_equation):
            if i == right_ellipsis_pos:
                output_dims.extend(ellipsis_dims[:num_ellipsis_indices])
                i += 3
                continue
            c = right_equation[i]
            if _is_letter(c):
                if c in label_dims:
                    output_dims.append(label_dims[c])
                else:
                    output_dims.append(ctx.new_symbolic_dim())
            i += 1
    else:
        # Implicit output: ellipsis dims first, then non-repeated labels in
        # alphabetical order (by ASCII: uppercase before lowercase, matching numpy)
        output_dims.extend(ellipsis_dims[:num_ellipsis_indices])
        for label in sorted(label_order):
            if label not in repeated_labels:
                output_dims.append(label_dims[label])

    return ir.Shape(output_dims)
