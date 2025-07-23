# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Version utils for testing."""

# pylint: disable=import-outside-toplevel
from __future__ import annotations

import packaging.version


def onnx_older_than(version: str) -> bool:
    """Returns True if the ONNX version is older than the given version."""
    import onnx  # noqa: TID251

    return (
        packaging.version.parse(onnx.__version__).release
        < packaging.version.parse(version).release
    )


def torch_older_than(version: str) -> bool:
    """Returns True if the torch version is older than the given version."""
    import torch

    return (
        packaging.version.parse(torch.__version__).release
        < packaging.version.parse(version).release
    )


def onnxruntime_older_than(version: str) -> bool:
    """Returns True if the onnxruntime version is older than the given version."""
    import onnxruntime

    return (
        packaging.version.parse(onnxruntime.__version__).release
        < packaging.version.parse(version).release
    )


def numpy_older_than(version: str) -> bool:
    """Returns True if the numpy version is older than the given version."""
    import numpy

    return (
        packaging.version.parse(numpy.__version__).release
        < packaging.version.parse(version).release
    )
