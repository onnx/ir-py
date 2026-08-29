# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
import unittest

import numpy as np
import parameterized

from onnx_ir import _type_casting


class TypeCastingTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("signed", np.int8),
            ("unsigned", np.uint8),
        ]
    )
    def test_pack_4bitx2_even_sized_array(self, _: str, dtype):
        array = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=dtype)
        expected = np.array([0x21, 0x43, 0x65, 0x87], dtype=np.uint8)
        actual = _type_casting.pack_4bitx2(array)
        np.testing.assert_array_equal(actual, expected)

    @parameterized.parameterized.expand(
        [
            ("signed", np.int8),
            ("unsigned", np.uint8),
        ]
    )
    def test_pack_4bitx2_odd_sized_array(self, _: str, dtype):
        array = np.array([1, 2, 3, 4, 5], dtype=dtype)
        expected = np.array([0x21, 0x43, 0x5], dtype=np.uint8)
        actual = _type_casting.pack_4bitx2(array)
        np.testing.assert_array_equal(actual, expected)

    @parameterized.parameterized.expand(
        [
            ("signed", np.int8),
            ("unsigned", np.uint8),
        ]
    )
    def test_pack_4bitx2_returns_flatten_array(self, _: str, dtype):
        array = np.array([[[1, 2, 3, 4, 5]]], dtype=dtype)
        expected = np.array([0x21, 0x43, 0x5], dtype=np.uint8)
        actual = _type_casting.pack_4bitx2(array)
        np.testing.assert_array_equal(actual, expected)

    def test_pack_6bit_uses_lsb_first_bit_order(self):
        expected = np.array([0x81, 0x30, 0x10], dtype=np.uint8)
        for dtype in (np.uint8, np.uint16):
            with self.subTest(dtype=dtype):
                array = np.array([1, 2, 3, 4], dtype=dtype)
                np.testing.assert_array_equal(_type_casting.pack_6bit(array), expected)

    def test_pack_unpack_6bit_round_trip_with_padding(self):
        for size in range(8):
            with self.subTest(size=size):
                array = np.arange(size, dtype=np.uint8)
                packed = _type_casting.pack_6bit(array)
                self.assertEqual(packed.size, (size * 6 + 7) // 8)
                np.testing.assert_array_equal(_type_casting.unpack_6bit(packed, [size]), array)

    def test_unpack_6bit_raises_for_truncated_data(self):
        with self.assertRaisesRegex(ValueError, "too small"):
            _type_casting.unpack_6bit(np.array([0, 0], dtype=np.uint8), [4])


if __name__ == "__main__":
    unittest.main(verbosity=2)
