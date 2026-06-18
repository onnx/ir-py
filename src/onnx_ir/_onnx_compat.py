# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Compatibility shim: import onnx or onnx_light.onnx based on USE_ONNX_LIGHT.

When the environment variable ``USE_ONNX_LIGHT`` is set to ``1``, this module
re-exports everything from ``onnx_light.onnx`` (and its submodules) so that the
rest of the codebase never imports ``onnx`` directly.  Otherwise it falls back
to the standard ``onnx`` package.

Usage in other modules::

    from onnx_ir._onnx_compat import onnx  # noqa: TID251

This single import line replaces ``import onnx``.
"""

from __future__ import annotations

import importlib
import os

_USE_ONNX_LIGHT = os.environ.get("USE_ONNX_LIGHT", "0") == "1"

if _USE_ONNX_LIGHT:
    onnx = importlib.import_module("onnx_light.onnx")
    # Ensure commonly used submodules are importable via attribute access
    onnx.external_data_helper = importlib.import_module(  # type: ignore[attr-defined]
        "onnx_light.onnx.external_data_helper"
    )
    onnx.checker = importlib.import_module("onnx_light.onnx.checker")  # type: ignore[attr-defined]
    onnx.shape_inference = importlib.import_module(  # type: ignore[attr-defined]
        "onnx_light.onnx.shape_inference"
    )
    onnx.defs = importlib.import_module("onnx_light.onnx.defs")  # type: ignore[attr-defined]
    onnx.helper = importlib.import_module("onnx_light.onnx.helper")  # type: ignore[attr-defined]
    onnx.backend = importlib.import_module("onnx_light.onnx.backend")  # type: ignore[attr-defined]
    onnx.numpy_helper = importlib.import_module(  # type: ignore[attr-defined]
        "onnx_light.onnx.numpy_helper"
    )
else:
    onnx = importlib.import_module("onnx")  # type: ignore[no-redef]
    # Ensure submodules are loaded
    importlib.import_module("onnx.external_data_helper")
    importlib.import_module("onnx.checker")
    importlib.import_module("onnx.shape_inference")
    importlib.import_module("onnx.defs")
