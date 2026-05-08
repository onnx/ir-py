# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

from onnx_ir._version_utils import numpy_older_than, onnx_older_than


class VersionUtilsTest(unittest.TestCase):
    def test_onnx_older_than(self):
        # ONNX is installed, so this should work
        result = onnx_older_than("0.0.1")
        self.assertFalse(result)
        result = onnx_older_than("99.99.99")
        self.assertTrue(result)

    def test_numpy_older_than(self):
        result = numpy_older_than("0.0.1")
        self.assertFalse(result)
        result = numpy_older_than("99.99.99")
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
