# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Generic shape inference for unary element-wise operators."""

from __future__ import annotations

__all__ = [
    "infer_unary",
]

from typing import TYPE_CHECKING

import onnx_ir as ir
from onnx_ir.shape_inference import _registry

if TYPE_CHECKING:
    from onnx_ir.shape_inference import _context

_reg = _registry.registry.register


@_reg("", "Identity", since_version=1)
@_reg("", "Neg", since_version=6)
@_reg("", "Abs", since_version=6)
@_reg("", "Ceil", since_version=6)
@_reg("", "Floor", since_version=6)
@_reg("", "Round", since_version=11)
@_reg("", "Reciprocal", since_version=6)
@_reg("", "Sqrt", since_version=6)
@_reg("", "Exp", since_version=6)
@_reg("", "Log", since_version=6)
@_reg("", "Sigmoid", since_version=6)
@_reg("", "Relu", since_version=6)
@_reg("", "LeakyRelu", since_version=6)
@_reg("", "Tanh", since_version=6)
@_reg("", "Erf", since_version=9)
@_reg("", "Sign", since_version=9)
@_reg("", "Sin", since_version=7)
@_reg("", "Cos", since_version=7)
@_reg("", "Tan", since_version=7)
@_reg("", "Asin", since_version=7)
@_reg("", "Acos", since_version=7)
@_reg("", "Atan", since_version=7)
@_reg("", "Sinh", since_version=9)
@_reg("", "Cosh", since_version=9)
@_reg("", "Asinh", since_version=9)
@_reg("", "Acosh", since_version=9)
@_reg("", "Atanh", since_version=9)
@_reg("", "Clip", since_version=6)
@_reg("", "BitwiseNot", since_version=18)
@_reg("", "Celu", since_version=12)
@_reg("", "Elu", since_version=6)
@_reg("", "Selu", since_version=6)
@_reg("", "Gelu", since_version=20)
@_reg("", "HardSigmoid", since_version=6)
@_reg("", "HardSwish", since_version=14)
@_reg("", "Softplus", since_version=1)
@_reg("", "Softsign", since_version=1)
@_reg("", "ThresholdedRelu", since_version=10)
def infer_unary(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for a unary element-wise operator.

    Output shape and dtype are identical to the first input.
    """
    if len(node.inputs) < 1:
        ctx.record_error(node, f"Expected at least 1 input, got {len(node.inputs)}")
        return

    input_val = node.inputs[0]
    if input_val is None:
        return

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], input_val.shape, input_val.dtype)


@_reg("", "Not", since_version=1)
@_reg("", "IsNaN", since_version=9)
@_reg("", "IsInf", since_version=10)
def infer_logical_unary(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape for a logical unary operator (output dtype = BOOL)."""
    if len(node.inputs) < 1:
        ctx.record_error(node, f"Expected at least 1 input, got {len(node.inputs)}")
        return

    input_val = node.inputs[0]
    if input_val is None:
        return

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], input_val.shape, ir.DataType.BOOL)
