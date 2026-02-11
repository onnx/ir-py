# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for Transpose operator."""

from __future__ import annotations

__all__ = [
    "infer_transpose",
]

import onnx_ir as ir
from onnx_ir.shape_inference import _context
from onnx_ir.shape_inference._registry import registry


@registry.register("", "Transpose", since_version=1)
def infer_transpose(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Transpose operator.

    Transpose permutes the dimensions of the input tensor according to the `perm` attribute.
    If `perm` is not provided, the dimensions are reversed.
    Output dtype is the same as input dtype.

    Spec: https://onnx.ai/onnx/operators/onnx__Transpose.html
    """
    (input_tensor,) = _context.check_inputs(node, "data")

    input_shape = input_tensor.shape
    input_dtype = input_tensor.dtype

    output_shape = None
    if input_shape is not None:
        rank = input_shape.rank()

        # Get perm attribute (optional)
        perm_attr = node.attributes.get("perm")
        if perm_attr is not None:
            perm = perm_attr.as_ints()
        else:
            # Default: reverse dimensions
            perm = tuple(range(rank - 1, -1, -1))

        # Validate perm
        if len(perm) == rank and sorted(perm) == list(range(rank)):
            # Permute dimensions
            output_dims: list[int | ir.SymbolicDim] = []
            valid = True
            for i in perm:
                if 0 <= i < rank:
                    output_dims.append(input_shape[i])
                else:
                    valid = False
                    break

            if valid:
                output_shape = ir.Shape(output_dims)

    if len(node.outputs) > 0:
        output = node.outputs[0]
        ctx.set_shape_and_dtype(output, output_shape, input_dtype)
