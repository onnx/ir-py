# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Test display() methods in various classes."""

import contextlib
import unittest

import numpy as np

import onnx_ir as ir


class DisplayTest(unittest.TestCase):
    def test_tensor_display_does_not_raise_on_nan_values(self):
        array_with_nan = np.array([np.inf, -np.inf, np.nan, 5, -10], dtype=np.float32)
        tensor = ir.Tensor(array_with_nan, dtype=ir.DataType.FLOAT)
        with contextlib.redirect_stdout(None):
            tensor.display()

    def test_display_graph(self):
        graph = ir.Graph([], [], nodes=[ir.node("TestOp", inputs=[])], name="test_graph")

        with contextlib.redirect_stdout(None):
            graph.display()


if __name__ == "__main__":
    unittest.main()
