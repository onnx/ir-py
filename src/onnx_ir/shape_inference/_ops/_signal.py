# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for signal processing operators."""

from __future__ import annotations

__all__ = [
    "infer_dft",
    "infer_stft",
]

import onnx_ir as ir
from onnx_ir.shape_inference import _context, _registry

_reg = _registry.registry.register


@_reg("", "DFT", since_version=17)
def infer_dft(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for DFT operator.

    Spec: https://onnx.ai/onnx/operators/onnx__DFT.html
    """
    (input_val,) = _context.check_inputs(node, "input")

    output_dtype = input_val.dtype

    if input_val.shape is None:
        if len(node.outputs) > 0:
            ctx.set_shape_and_dtype(node.outputs[0], None, output_dtype)
        return

    output_dims: list[int | ir.SymbolicDim] = list(input_val.shape.dims)

    inverse_attr = node.attributes.get("inverse")
    inverse = inverse_attr.as_int() if inverse_attr is not None else 0

    onesided_attr = node.attributes.get("onesided")
    onesided = onesided_attr.as_int() if onesided_attr is not None else 0

    # Determine DFT axis: attribute in v17, input[2] in v20+
    axis = 1  # default
    axis_attr = node.attributes.get("axis")
    if axis_attr is not None:
        axis = axis_attr.as_int()
    elif len(node.inputs) > 2 and node.inputs[2] is not None:
        axis_const = ir.convenience.get_const_tensor(node.inputs[2])
        if axis_const is not None:
            axis = int(axis_const.numpy().item())

    # Read optional dft_length (input[1])
    dft_length: int | None = None
    if len(node.inputs) > 1 and node.inputs[1] is not None:
        length_const = ir.convenience.get_const_tensor(node.inputs[1])
        if length_const is not None:
            dft_length = int(length_const.numpy().item())

    if dft_length is not None:
        output_dims[axis] = dft_length
    # For onesided: axis dim becomes n//2+1 (forward) or full (inverse)
    if onesided and not inverse:
        axis_dim = output_dims[axis]
        if isinstance(axis_dim, int):
            output_dims[axis] = axis_dim // 2 + 1
    elif onesided and inverse:
        axis_dim = output_dims[axis]
        if isinstance(axis_dim, int):
            output_dims[axis] = (axis_dim - 1) * 2

    # Last dim: real input (1) -> complex output (2), complex (2) stays 2
    # inverse + onesided: complex input -> real output (1)
    if inverse and onesided:
        output_dims[-1] = 1
    else:
        output_dims[-1] = 2

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], ir.Shape(output_dims), output_dtype)


@_reg("", "STFT", since_version=17)
def infer_stft(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for STFT operator.

    Output: [batch_size, frames, freq_bins, 2]

    Spec: https://onnx.ai/onnx/operators/onnx__STFT.html
    """
    (signal, _frame_step) = _context.check_inputs(node, "signal", "frame_step")

    output_dtype = signal.dtype

    if signal.shape is not None:
        batch_size = signal.shape[0]
        frames = ctx.new_symbolic_dim()
        freq_bins = ctx.new_symbolic_dim()
        output_shape = ir.Shape([batch_size, frames, freq_bins, 2])
    else:
        output_shape = None

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, output_dtype)
