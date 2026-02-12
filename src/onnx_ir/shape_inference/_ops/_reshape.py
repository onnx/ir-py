# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for Reshape operator."""

from __future__ import annotations

__all__ = [
    "infer_reshape",
]


import math

import onnx_ir as ir
from onnx_ir.shape_inference import _context, _registry


@_registry.registry.register("", "Reshape", since_version=5)
def infer_reshape(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for Reshape operator.

    Spec: https://onnx.ai/onnx/operators/onnx__Reshape.html
    """
    (data, shape_input) = _context.check_inputs(node, "data", "shape")

    input_dtype = data.dtype
    input_shape = data.shape

    # Try to read target dims from const_value, then symbolic_value
    shape_const = ir.convenience.get_const_tensor(shape_input)
    if shape_const is not None:
        target_dims: list[int | ir.SymbolicDim] = [
            int(x) for x in shape_const.numpy().flatten()
        ]
    else:
        sym_val = ctx.get_symbolic_value(shape_input)
        if sym_val is not None:
            target_dims = list(sym_val)
        else:
            # Shape is fully dynamic — try to infer rank from shape input's shape
            if shape_input.shape is not None and shape_input.shape.rank() == 1:
                dim0 = shape_input.shape[0]
                if isinstance(dim0, int):
                    output_shape = ir.Shape([ctx.new_symbolic_dim() for _ in range(dim0)])
                    if len(node.outputs) > 0:
                        ctx.set_shape_and_dtype(node.outputs[0], output_shape, input_dtype)
            else:
                if len(node.outputs) > 0:
                    ctx.set_shape_and_dtype(node.outputs[0], None, input_dtype)
            return

    allowzero_attr = node.attributes.get("allowzero")
    allowzero = allowzero_attr.as_int() if allowzero_attr is not None else 0

    # Process target dims: handle 0 and -1
    output_dims: list[int | ir.SymbolicDim] = []
    inferred_idx: int | None = None

    for i, dim_val in enumerate(target_dims):
        if isinstance(dim_val, int) and dim_val == 0 and not allowzero:
            # Copy from input shape
            if input_shape is not None and i < input_shape.rank():
                output_dims.append(input_shape[i])
            else:
                output_dims.append(ctx.new_symbolic_dim())
        elif isinstance(dim_val, int) and dim_val == -1:
            if inferred_idx is not None:
                ctx.record_error(node, "At most one dimension can be -1 in Reshape")
                return
            inferred_idx = i
            output_dims.append(-1)  # placeholder
        else:
            output_dims.append(dim_val)

    # Try to compute the inferred dimension
    if inferred_idx is not None and input_shape is not None and input_shape.is_static():
        total_input = math.prod(d if isinstance(d, int) else 0 for d in input_shape.dims)
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
            output_dims[inferred_idx] = ctx.new_symbolic_dim()
    elif inferred_idx is not None:
        output_dims[inferred_idx] = ctx.new_symbolic_dim()

    output_shape = ir.Shape(output_dims)
    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, input_dtype)
