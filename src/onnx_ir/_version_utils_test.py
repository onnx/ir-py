# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest
from unittest import mock

from onnx_ir._version_utils import numpy_older_than, onnx_older_than


class VersionUtilsTest(unittest.TestCase):
    def test_onnx_older_than_returns_false_for_old_version(self):
        with mock.patch("onnx.__version__", "1.16.0"):
            self.assertFalse(onnx_older_than("0.0.1"))

    def test_onnx_older_than_returns_true_for_newer_version(self):
        with mock.patch("onnx.__version__", "1.16.0"):
            self.assertTrue(onnx_older_than("99.99.99"))

    def test_onnx_older_than_exact_version_is_not_older(self):
        with mock.patch("onnx.__version__", "1.16.0"):
            self.assertFalse(onnx_older_than("1.16.0"))

    def test_onnx_older_than_ignores_prerelease(self):
        # 2.0.0rc1 should NOT be older than 2.0.0 because .release is compared
        with mock.patch("onnx.__version__", "2.0.0rc1"):
            self.assertFalse(onnx_older_than("2.0.0"))

    def test_onnx_older_than_ignores_dev_suffix(self):
        with mock.patch("onnx.__version__", "1.16.0.dev20240101"):
            self.assertFalse(onnx_older_than("1.16.0"))

    def test_numpy_older_than_returns_false_for_old_version(self):
        with mock.patch("numpy.__version__", "1.26.0"):
            self.assertFalse(numpy_older_than("0.0.1"))

    def test_numpy_older_than_returns_true_for_newer_version(self):
        with mock.patch("numpy.__version__", "1.26.0"):
            self.assertTrue(numpy_older_than("99.99.99"))

    def test_numpy_older_than_exact_version_is_not_older(self):
        with mock.patch("numpy.__version__", "1.26.0"):
            self.assertFalse(numpy_older_than("1.26.0"))

    def test_numpy_older_than_ignores_prerelease(self):
        with mock.patch("numpy.__version__", "2.0.0rc1"):
            self.assertFalse(numpy_older_than("2.0.0"))


if __name__ == "__main__":
    unittest.main()
