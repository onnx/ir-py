# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest
from unittest import mock, skipIf

from onnx_ir._onnx_compat import use_onnx_light
from onnx_ir._version_utils import numpy_older_than, onnx_older_than


@skipIf(use_onnx_light, "irrelevant")
class VersionUtilsTest(unittest.TestCase):
    def test_onnx_older_than_handles_exact_prerelease_dev_and_local_versions(self):
        cases = [
            ("1.16.0", "1.16.0", False),
            ("2.0.0rc1", "2.0.0", True),
            ("1.16.0.dev20240101", "1.16.0", True),
            ("1.16.0+cpu", "1.16.0", False),
        ]
        for current_version, compare_version, expected in cases:
            with (
                self.subTest(current_version=current_version, compare_version=compare_version),
                mock.patch("onnx.__version__", current_version),
            ):
                self.assertEqual(onnx_older_than(compare_version), expected)

    def test_numpy_older_than_handles_exact_prerelease_dev_and_local_versions(self):
        cases = [
            ("1.26.0", "1.26.0", False),
            ("2.0.0rc1", "2.0.0", True),
            ("2.0.0.dev0", "2.0.0", True),
            ("2.0.0+cpu", "2.0.0", False),
        ]
        for current_version, compare_version, expected in cases:
            with (
                self.subTest(current_version=current_version, compare_version=compare_version),
                mock.patch("numpy.__version__", current_version),
            ):
                self.assertEqual(numpy_older_than(compare_version), expected)


if __name__ == "__main__":
    unittest.main()
