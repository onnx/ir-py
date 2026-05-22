# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

from onnx_ir import _metadata


class MetadataStoreTest(unittest.TestCase):
    def test_repr(self):
        store = _metadata.MetadataStore()
        store["key"] = "value"
        r = repr(store)
        self.assertIn("MetadataStore", r)
        self.assertIn("key", r)

    def test_bool_true_with_data(self):
        store = _metadata.MetadataStore()
        store["key"] = "value"
        self.assertTrue(bool(store))

    def test_bool_true_with_invalid_keys(self):
        store = _metadata.MetadataStore()
        store.invalidate("some_key")
        self.assertTrue(bool(store))

    def test_bool_false_when_empty(self):
        store = _metadata.MetadataStore()
        self.assertFalse(bool(store))


if __name__ == "__main__":
    unittest.main()
