# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Test IR roundtrip with ONNX model zoo.

NOTE: This test is disabled because onnx.hub is deprecated and no longer available.
"""

from __future__ import annotations

import sys


def main():
    """onnx.hub is deprecated and no longer available."""
    print("model_zoo_test is skipped: onnx.hub is deprecated")
    sys.exit(0)


if __name__ == "__main__":
    main()
