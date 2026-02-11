# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for Reshape operator."""

from __future__ import annotations

__all__ = [
    "infer_reshape",
]

import logging
import math
from typing import TYPE_CHECKING

import onnx_ir as ir
from onnx_ir.shape_inference import _registry

if TYPE_CHECKING:
    from onnx_ir.shape_inference import _context

logger = logging.getLogger(__name__)


@_registry.registry.register("", "Reshape", since_version=5)
def infer_reshape(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Reshape operator.

    Spec: https://onnx.ai/onnx/operators/onnx__Reshape.html
    """
    if len(node.inputs) < 2:
        ctx.record_error(node, f"Expected 2 inputs, got {len(node.inputs)}")
        return

    data = node.inputs[0]
    shape_input = node.inputs[1]
    if data is None or shape_input is None:
        return

    input_dtype = data.dtype
    input_shape = data.shape

    # Try to read the target shape from const_value
    shape_const = shape_input.const_value
    if shape_const is None:
        # Shape is dynamic — we can try to infer rank from the shape input's shape
        if shape_input.shape is not None and shape_input.shape.rank() == 1:
            dim0 = shape_input.shape[0]
            if isinstance(dim0, int):
                output_shape = ir.Shape([ir.SymbolicDim(None)] * dim0)
                if len(node.outputs) > 0:
                    ctx.set_shape_and_dtype(node.outputs[0], output_shape, input_dtype)
        else:
            if len(node.outputs) > 0:
                ctx.set_shape_and_dtype(node.outputs[0], None, input_dtype)
        return

    target_dims = [int(x) for x in shape_const.numpy().flatten()]

    allowzero_attr = node.attributes.get("allowzero")
    allowzero = allowzero_attr.as_int() if allowzero_attr is not None else 0

    # Process target dims: handle 0 and -1
    output_dims: list[int | ir.SymbolicDim] = []
    inferred_idx: int | None = None

    for i, dim_val in enumerate(target_dims):
        if dim_val == 0 and not allowzero:
            # Copy from input shape
            if input_shape is not None and i < input_shape.rank():
                output_dims.append(input_shape[i])
            else:
                output_dims.append(ir.SymbolicDim(None))
        elif dim_val == -1:
            if inferred_idx is not None:
                ctx.record_error(node, "At most one dimension can be -1 in Reshape")
                return
            inferred_idx = i
            output_dims.append(-1)  # placeholder
        else:
            output_dims.append(dim_val)

    # Try to compute the inferred dimension
    if inferred_idx is not None and input_shape is not None and input_shape.is_static():
        total_input = math.prod(
            d if isinstance(d, int) else 0 for d in input_shape.dims
        )
        known_output = 1
        all_known = True
        for i, d in enumerate(output_dims):
            if i == inferred_idx:
                continue
            if isinstance(d, int) and d > 0:
                known_output *= d
            else:
                all_known = False
                break

        if all_known and known_output > 0 and total_input > 0:
            output_dims[inferred_idx] = total_input // known_output
        else:
            output_dims[inferred_idx] = ir.SymbolicDim(None)
    elif inferred_idx is not None:
        output_dims[inferred_idx] = ir.SymbolicDim(None)

    output_shape = ir.Shape(output_dims)
    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, input_dtype)
