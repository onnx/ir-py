# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for ai.onnx.preview.training operators."""

from __future__ import annotations

__all__ = [
    "infer_adagrad",
    "infer_adam",
    "infer_momentum",
]

import onnx_ir as ir
from onnx_ir.shape_inference import _context, _registry

_reg = _registry.registry.register


def _infer_training_op(
    ctx: _context.ShapeInferenceContext, node: ir.Node, inputs_per_group: int
) -> None:
    """Generic inference for training optimizers.

    Training ops (Adagrad, Adam, Momentum) share the same pattern:
    - First 2 inputs are R (learning rate, scalar) and T (step, scalar).
    - Remaining inputs are grouped: each group has ``inputs_per_group``
      tensors (e.g., X, G, H for Adagrad).
    - Outputs correspond to the updated state tensors, matching the shapes
      and dtypes of the corresponding inputs in each group.
    """
    # Skip R and T (first 2 inputs)
    group_inputs = [inp for inp in node.inputs[2:] if inp is not None]
    if not group_inputs:
        return

    num_groups = len(group_inputs) // inputs_per_group
    for i, output in enumerate(node.outputs):
        # Map output i to the corresponding input in the group
        group_idx = i // (len(node.outputs) // num_groups) if num_groups > 0 else 0
        output_within_group = i % (len(node.outputs) // num_groups) if num_groups > 0 else 0
        input_idx = group_idx * inputs_per_group + output_within_group
        if input_idx < len(group_inputs):
            src = group_inputs[input_idx]
            ctx.set_shape_and_dtype(output, src.shape, src.dtype)


@_reg("ai.onnx.preview.training", "Adagrad", since_version=1)
def infer_adagrad(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shapes for Adagrad. Outputs match corresponding X and H inputs."""
    # Inputs: R, T, [X1, X2, ...], [G1, G2, ...], [H1, H2, ...]
    # Outputs: [X1_new, X2_new, ...], [H1_new, H2_new, ...]
    # Each output matches the shape of the corresponding X or H
    remaining = [inp for inp in node.inputs[2:] if inp is not None]
    # 3 inputs per param: X, G, H
    n_params = len(remaining) // 3
    for i, output in enumerate(node.outputs):
        if i < n_params:
            # X_new matches X
            src = remaining[i]
        else:
            # H_new matches H
            src = remaining[2 * n_params + (i - n_params)]
        ctx.set_shape_and_dtype(output, src.shape, src.dtype)


@_reg("ai.onnx.preview.training", "Adam", since_version=1)
def infer_adam(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shapes for Adam. Outputs match corresponding X, V, H inputs."""
    # Inputs: R, T, [X1, X2, ...], [G1, G2, ...], [V1, V2, ...], [H1, H2, ...]
    # Outputs: [X1_new, ...], [V1_new, ...], [H1_new, ...]
    remaining = [inp for inp in node.inputs[2:] if inp is not None]
    # 4 inputs per param: X, G, V, H
    n_params = len(remaining) // 4
    for i, output in enumerate(node.outputs):
        if i < n_params:
            src = remaining[i]  # X
        elif i < 2 * n_params:
            src = remaining[2 * n_params + (i - n_params)]  # V
        else:
            src = remaining[3 * n_params + (i - 2 * n_params)]  # H
        ctx.set_shape_and_dtype(output, src.shape, src.dtype)


@_reg("ai.onnx.preview.training", "Momentum", since_version=1)
def infer_momentum(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shapes for Momentum. Outputs match corresponding X, V inputs."""
    # Inputs: R, T, [X1, X2, ...], [G1, G2, ...], [V1, V2, ...]
    # Outputs: [X1_new, ...], [V1_new, ...]
    remaining = [inp for inp in node.inputs[2:] if inp is not None]
    # 3 inputs per param: X, G, V
    n_params = len(remaining) // 3
    for i, output in enumerate(node.outputs):
        if i < n_params:
            src = remaining[i]  # X
        else:
            src = remaining[2 * n_params + (i - n_params)]  # V
        ctx.set_shape_and_dtype(output, src.shape, src.dtype)
