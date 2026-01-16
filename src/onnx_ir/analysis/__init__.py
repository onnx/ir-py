# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Analysis utilities for ONNX IR graphs."""

from __future__ import annotations

__all__ = [
    "analyze_implicit_usage",
]

from onnx_ir.analysis._implicit_usage import analyze_implicit_usage


def __set_module() -> None:
    """Set the module of all functions in this module to this public module."""
    global_dict = globals()
    for name in __all__:
        global_dict[name].__module__ = __name__


__set_module()
