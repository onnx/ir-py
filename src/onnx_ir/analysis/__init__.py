# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Analysis utilities for ONNX IR graphs."""

from __future__ import annotations

__all__ = [
    "analyze_implicit_usage",
]

from onnx_ir.analysis._implicit_usage import analyze_implicit_usage
