# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference operators.

This module imports all operator shape inference functions to ensure
they are registered with the global registry.
"""

# Import to trigger registration
from onnx_ir.shape_inference._ops import (
    _cast,
    _concat,
    _constant,
    _constant_of_shape,
    _control_flow,
    _conv,
    _dropout,
    _elementwise,
    _expand,
    _gather,
    _gemm,
    _matmul,
    _reduce,
    _reshape,
    _sequence,
    _shape_ops,
    _slice,
    _softmax,
    _split,
    _squeeze,
    _transpose,
    _unary,
    _where,
)

__all__ = [
    "_cast",
    "_concat",
    "_constant",
    "_constant_of_shape",
    "_control_flow",
    "_conv",
    "_dropout",
    "_elementwise",
    "_expand",
    "_gather",
    "_gemm",
    "_matmul",
    "_reduce",
    "_reshape",
    "_sequence",
    "_shape_ops",
    "_slice",
    "_softmax",
    "_split",
    "_squeeze",
    "_transpose",
    "_unary",
    "_where",
]
