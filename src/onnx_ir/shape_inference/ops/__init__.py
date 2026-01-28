# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shape inference operators.

This module imports all operator shape inference functions to ensure
they are registered with the global registry.
"""

# Import to trigger registration
from onnx_ir.shape_inference.ops._add import infer_add
from onnx_ir.shape_inference.ops._transpose import infer_transpose

__all__ = [
    "infer_add",
    "infer_transpose",
]
