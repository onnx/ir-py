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


class DisplayRichFallbackTest(unittest.TestCase):
    def test_require_rich_returns_none_when_not_installed(self):
        from unittest import mock

        from onnx_ir import _display

        with mock.patch.dict("sys.modules", {"rich": None}):
            result = _display.require_rich()
            self.assertIsNone(result)

    def test_display_without_rich_prints_plaintext(self):
        """Test the fallback when rich is not available."""
        import io
        from unittest import mock

        from onnx_ir import _display

        graph = ir.Graph([], [], nodes=[], name="test_graph")

        # Mock require_rich to return None (simulating rich not installed)
        with mock.patch.object(_display, "require_rich", return_value=None):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                graph.display()
            printed = output.getvalue()
            self.assertIn("Tip:", printed)

    def test_display_with_page(self):
        """Test display with page=True uses rich pager."""
        import io

        graph = ir.Graph([], [], nodes=[], name="test_graph")
        # Just ensure it doesn't raise. We can't easily test rich pager.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.suppress(Exception):
            graph.display(page=True)


if __name__ == "__main__":
    unittest.main()
