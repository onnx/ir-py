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


@_reg("", "DFT", since_version=20)
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

    # Output has same rank as input with symbolic dims
    rank = input_val.shape.rank()
    output_dims: list[int | ir.SymbolicDim] = [ctx.new_symbolic_dim() for _ in range(rank)]
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
