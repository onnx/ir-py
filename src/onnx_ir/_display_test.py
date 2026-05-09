# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Test display() methods in various classes."""

import contextlib
import io
import unittest
from unittest import mock

import numpy as np

import onnx_ir as ir
from onnx_ir import _display


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
        with mock.patch.dict("sys.modules", {"rich": None}):
            result = _display.require_rich()
            self.assertIsNone(result)

    def test_display_without_rich_prints_plaintext(self):
        """Test the fallback when rich is not available."""
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
        graph = ir.Graph([], [], nodes=[], name="test_graph")
        mock_console = mock.MagicMock()
        with mock.patch("rich.console.Console", return_value=mock_console):
            graph.display(page=True)
        mock_console.pager.assert_called_once()
        mock_console.print.assert_called_once()


if __name__ == "__main__":
    unittest.main()
