# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for attention operators."""

from __future__ import annotations

__all__ = [
    "infer_attention",
    "infer_rotary_embedding",
]

import onnx_ir as ir
from onnx_ir.shape_inference import _context, _registry

_reg = _registry.registry.register


@_reg("", "Attention", since_version=23)
def infer_attention(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Attention operator.

    Spec: https://onnx.ai/onnx/operators/onnx__Attention.html
    """
    (q, _k, v) = _context.check_inputs(node, "Q", "K", "V")

    output_dtype = q.dtype

    # Output[0] shape = Q's shape (batch, seq_q, v_hidden approximation)
    output_shape = q.shape
    if output_shape is None and v.shape is not None:
        output_shape = v.shape

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, output_dtype)

    # present_key output
    if len(node.outputs) > 1 and node.outputs[1] is not None:
        ctx.set_shape_and_dtype(node.outputs[1], _k.shape, _k.dtype)

    # present_value output
    if len(node.outputs) > 2 and node.outputs[2] is not None:
        ctx.set_shape_and_dtype(node.outputs[2], v.shape, v.dtype)


@_reg("", "RotaryEmbedding", since_version=24)
def infer_rotary_embedding(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for RotaryEmbedding operator.

    Spec: https://onnx.ai/onnx/operators/onnx__RotaryEmbedding.html
    """
    (input_val,) = _context.check_inputs(node, "input")

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], input_val.shape, input_val.dtype)

    # Optional second output (output_position_ids)
    if len(node.outputs) > 1 and node.outputs[1] is not None:
        position_ids_shape: ir.Shape | None = None
        if len(node.inputs) > 1 and node.inputs[1] is not None:
            position_ids_shape = node.inputs[1].shape
        ctx.set_shape_and_dtype(node.outputs[1], position_ids_shape, ir.DataType.INT64)
