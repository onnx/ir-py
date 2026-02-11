# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference operators.

This module imports all operator shape inference functions to ensure
they are registered with the global registry.
"""

# Import to trigger registration
from onnx_ir.shape_inference._ops import _cast
from onnx_ir.shape_inference._ops import _concat
from onnx_ir.shape_inference._ops import _constant
from onnx_ir.shape_inference._ops import _constant_of_shape
from onnx_ir.shape_inference._ops import _conv
from onnx_ir.shape_inference._ops import _elementwise
from onnx_ir.shape_inference._ops import _expand
from onnx_ir.shape_inference._ops import _gather
from onnx_ir.shape_inference._ops import _gemm
from onnx_ir.shape_inference._ops import _matmul
from onnx_ir.shape_inference._ops import _reduce
from onnx_ir.shape_inference._ops import _reshape
from onnx_ir.shape_inference._ops import _shape_ops
from onnx_ir.shape_inference._ops import _slice
from onnx_ir.shape_inference._ops import _softmax
from onnx_ir.shape_inference._ops import _split
from onnx_ir.shape_inference._ops import _squeeze
from onnx_ir.shape_inference._ops import _unary
from onnx_ir.shape_inference._ops import _where
from onnx_ir.shape_inference._ops._add import infer_add
from onnx_ir.shape_inference._ops._transpose import infer_transpose

__all__ = [
    "infer_add",
    "infer_transpose",
    # Modules (imported to trigger registration)
    "_cast",
    "_concat",
    "_constant",
    "_constant_of_shape",
    "_conv",
    "_elementwise",
    "_expand",
    "_gather",
    "_gemm",
    "_matmul",
    "_reduce",
    "_reshape",
    "_shape_ops",
    "_slice",
    "_softmax",
    "_split",
    "_squeeze",
    "_unary",
    "_where",
]
