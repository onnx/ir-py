# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference for window function operators."""

from __future__ import annotations

__all__ = [
    "infer_window",
]

import onnx_ir as ir
from onnx_ir.shape_inference import _context, _registry

_reg = _registry.registry.register


@_reg("", "BlackmanWindow", since_version=17)
@_reg("", "HammingWindow", since_version=17)
@_reg("", "HannWindow", since_version=17)
def infer_window(ctx: _context.ShapeInferenceContext, node: ir.Node) -> None:
    """Infer shape and dtype for BlackmanWindow, HammingWindow, HannWindow.

    Output is a 1-D tensor whose length equals the scalar ``size`` input.
    """
    (size,) = _context.check_inputs(node, "size")

    dtype_attr = node.attributes.get("output_datatype")
    output_dtype = (
        ir.DataType(dtype_attr.as_int()) if dtype_attr is not None else ir.DataType.FLOAT
    )

    size_const = ir.convenience.get_const_tensor(size)
    if size_const is not None:
        size_val = int(size_const.numpy().item())
        output_shape: ir.Shape = ir.Shape([size_val])
    else:
        output_shape = ir.Shape([ctx.new_symbolic_dim()])

    if len(node.outputs) > 0:
        ctx.set_shape_and_dtype(node.outputs[0], output_shape, output_dtype)
