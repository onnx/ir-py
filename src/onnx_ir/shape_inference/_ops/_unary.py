# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Generic shape inference for unary element-wise operators."""

from __future__ import annotations

__all__ = [
    "infer_cumsum",
    "infer_prelu",
    "infer_unary",
]

import math
from collections.abc import Callable

import onnx_ir as ir
from onnx_ir.shape_inference import _context, _registry

_reg = _registry.registry.register


@_reg("", "ReverseSequence", since_version=10)
@_reg("", "MeanVarianceNormalization", since_version=9)
@_reg("", "LpNormalization", since_version=1)
@_reg("", "Mish", since_version=22)
@_reg("", "Shrink", since_version=9)
@_reg("", "Swish", since_version=22)
@_reg("", "Trilu", since_version=14)
@_reg("", "Abs", since_version=6)
@_reg("", "Acos", since_version=7)
@_reg("", "Acosh", since_version=9)
@_reg("", "Asin", since_version=7)
@_reg("", "Asinh", since_version=9)
@_reg("", "Atan", since_version=7)
@_reg("", "Atanh", since_version=9)
@_reg("", "BitwiseNot", since_version=18)
@_reg("", "Celu", since_version=12)
@_reg("", "Clip", since_version=6)
@_reg("", "Cos", since_version=7)
@_reg("", "Cosh", since_version=9)
@_reg("", "Elu", since_version=6)
@_reg("", "Erf", since_version=9)
@_reg("", "Exp", since_version=6)
@_reg("", "Gelu", since_version=20)
@_reg("", "HardSigmoid", since_version=6)
@_reg("", "HardSwish", since_version=14)
@_reg("", "Identity", since_version=1)
@_reg("", "LeakyRelu", since_version=6)
@_reg("", "Log", since_version=6)
@_reg("", "Reciprocal", since_version=6)
@_reg("", "Relu", since_version=6)
@_reg("", "Round", since_version=11)
@_reg("", "Selu", since_version=6)
@_reg("", "Sigmoid", since_version=6)
@_reg("", "Sign", since_version=9)
@_reg("", "Sin", since_version=7)
@_reg("", "Sinh", since_version=9)
@_reg("", "Softplus", since_version=1)
@_reg("", "Softsign", since_version=1)
@_reg("", "Sqrt", since_version=6)
@_reg("", "Tan", since_version=7)
@_reg("", "Tanh", since_version=6)
@_reg("", "ThresholdedRelu", since_version=10)
def infer_unary(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for a unary element-wise operator.

    Output shape and dtype are identical to the first input.
    """
    (input_val,) = _context.check_inputs(node, "X")

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], input_val.shape, input_val.dtype)


_UNARY_VALUE_OPS: dict[str, Callable[[object], object]] = {
    "Floor": math.floor,  # type: ignore[dict-item]
    "Ceil": math.ceil,  # type: ignore[dict-item]
    "Neg": lambda x: -x,  # type: ignore[operator]
}


@_reg("", "Floor", since_version=6)
@_reg("", "Ceil", since_version=6)
@_reg("", "Neg", since_version=6)
def _infer_unary_with_value(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape/dtype and propagate symbolic_value for Floor, Ceil, Neg."""
    (input_val,) = _context.check_inputs(node, "X")

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], input_val.shape, input_val.dtype)

        op_func = _UNARY_VALUE_OPS.get(node.op_type)
        if op_func is not None:
            sym_val = ctx.get_symbolic_value(input_val)
            if sym_val is not None:
                ctx.set_symbolic_value(
                    node.outputs[0],
                    [op_func(v) for v in sym_val],  # type: ignore[misc]
                )


@_reg("", "PRelu", since_version=16)
def infer_prelu(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for PRelu operator.

    Output shape and dtype are identical to the first input (X).
    """
    (input_val, _slope) = _context.check_inputs(node, "X", "slope")

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], input_val.shape, input_val.dtype)


@_reg("", "CumSum", since_version=14)
@_reg("", "CumProd", since_version=22)
def infer_cumsum(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for CumSum/CumProd operator.

    Output shape and dtype are identical to the first input.
    """
    (input_val,) = _context.check_inputs(node, "x")

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], input_val.shape, input_val.dtype)


@_reg("", "RegexFullMatch", since_version=20)
@_reg("", "Not", since_version=1)
@_reg("", "IsNaN", since_version=9)
@_reg("", "IsInf", since_version=10)
def infer_logical_unary(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape for a logical unary operator (output dtype = BOOL)."""
    (input_val,) = _context.check_inputs(node, "X")

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], input_val.shape, ir.DataType.BOOL)
