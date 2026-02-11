# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Generic shape inference for binary element-wise operators."""

from __future__ import annotations

__all__ = [
    "infer_binary_elementwise",
]

import logging
from typing import TYPE_CHECKING

import onnx_ir as ir
from onnx_ir.shape_inference import _broadcast
from onnx_ir.shape_inference import _registry

if TYPE_CHECKING:
    from onnx_ir.shape_inference import _context

logger = logging.getLogger(__name__)

_reg = _registry.registry.register


def infer_binary_elementwise(
    ctx: _context.ShapeInferenceContext,
    node: ir.Node,
    *,
    output_dtype_override: ir.DataType | None = None,
) -> None:
    """Infer shape and dtype for a binary element-wise operator.

    Output shape is the broadcast of the two input shapes.
    Output dtype is ``output_dtype_override`` when given, otherwise the dtype of
    the first input that has one.
    """
    if len(node.inputs) < 2:
        ctx.record_error(node, f"Expected at least 2 inputs, got {len(node.inputs)}")
        return

    input_a = node.inputs[0]
    input_b = node.inputs[1]

    if input_a is None or input_b is None:
        return

    output_shape = _broadcast.broadcast_shapes(input_a.shape, input_b.shape)
    output_dtype = output_dtype_override or input_a.dtype or input_b.dtype

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, output_dtype)


# --- Arithmetic binary ops (output dtype = input dtype) ---


@_reg("", "Add", since_version=7)
@_reg("", "Sub", since_version=7)
@_reg("", "Mul", since_version=7)
@_reg("", "Div", since_version=7)
@_reg("", "Mod", since_version=10)
@_reg("", "Pow", since_version=7)
@_reg("", "BitShift", since_version=11)
def _infer_arithmetic(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    infer_binary_elementwise(ctx, node)


# --- Comparison ops (output dtype = BOOL) ---


@_reg("", "Equal", since_version=7)
@_reg("", "Less", since_version=7)
@_reg("", "Greater", since_version=7)
@_reg("", "LessOrEqual", since_version=12)
@_reg("", "GreaterOrEqual", since_version=12)
def _infer_comparison(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    infer_binary_elementwise(ctx, node, output_dtype_override=ir.DataType.BOOL)


# --- Logical binary ops (output dtype = BOOL) ---


@_reg("", "And", since_version=7)
@_reg("", "Or", since_version=7)
@_reg("", "Xor", since_version=7)
def _infer_logical(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    infer_binary_elementwise(ctx, node, output_dtype_override=ir.DataType.BOOL)
