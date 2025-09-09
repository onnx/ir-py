# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the _constructors module."""

import unittest

import numpy as np

import onnx_ir as ir
from onnx_ir._convenience import _constructors


class ConstructorsTest(unittest.TestCase):
    def test_tensor_accepts_torch_tensor(self):
        import torch as some_random_name  # pylint: disable=import-outside-toplevel

        torch_tensor = some_random_name.tensor([1, 2, 3])
        tensor = _constructors.tensor(torch_tensor)
        np.testing.assert_array_equal(tensor, torch_tensor.numpy())

    def test_tensor_raises_value_error_for_empty_sequence_without_dtype(self):
        with self.assertRaises(ValueError):
            _constructors.tensor([])

    def test_tensor_handles_empty_sequence_with_dtype(self):
        tensor = _constructors.tensor([], dtype=ir.DataType.FLOAT)
        np.testing.assert_array_equal(tensor.numpy(), np.array([], dtype=np.float32))


class ValueConstructorTest(unittest.TestCase):
    def test_value_minimal_creation(self):
        """Test creating a value with just a name."""
        value = _constructors.val("minimal")

        self.assertEqual(value.name, "minimal")
        self.assertIsNone(value.type)
        self.assertIsNone(value.shape)
        self.assertIsNone(value.const_value)

    def test_value_creation_with_sequence_shape(self):
        """Test that shape is correctly converted from sequence to Shape object."""
        value = _constructors.val("test", ir.DataType.INT32, [1, 2, 3])

        self.assertEqual(value.name, "test")
        self.assertIsInstance(value.shape, ir.Shape)
        self.assertEqual(value.shape, ir.Shape([1, 2, 3]))

    def test_value_creation_with_explicit_type(self):
        """Test value creation with explicit type parameter."""
        tensor_type = ir.TensorType(ir.DataType.DOUBLE)
        value = _constructors.val("y", type=tensor_type, shape=[10])

        self.assertEqual(value.name, "y")
        self.assertEqual(value.type, tensor_type)
        self.assertEqual(value.shape, ir.Shape([10]))

    def test_value_creation_with_const_value(self):
        """Test value creation with const_value (initializer)."""
        const_tensor = ir.Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32), name="const")
        value = _constructors.val("initializer", const_value=const_tensor)

        self.assertEqual(value.name, "initializer")
        self.assertEqual(value.type, ir.TensorType(ir.DataType.FLOAT))
        self.assertEqual(value.shape, ir.Shape([3]))
        self.assertEqual(value.const_value, const_tensor)

    def test_value_creation_with_dtype_only(self):
        """Test value creation with only dtype specified."""
        value = _constructors.val("float_value", dtype=ir.DataType.FLOAT)

        self.assertEqual(value.name, "float_value")
        self.assertEqual(value.type, ir.TensorType(ir.DataType.FLOAT))
        self.assertIsNone(value.shape)

    def test_value_const_value_type_mismatch_error(self):
        """Test that providing mismatched type with const_value raises ValueError."""
        const_tensor = ir.tensor([1, 2, 3], dtype=ir.DataType.INT32)
        wrong_type = ir.TensorType(ir.DataType.FLOAT)

        with self.assertRaisesRegex(ValueError, "The type does not match the const_value"):
            _constructors.val("test", type=wrong_type, const_value=const_tensor)

    def test_value_const_value_dtype_mismatch_error(self):
        """Test that providing mismatched dtype with const_value raises ValueError."""
        const_tensor = ir.tensor([1.0, 2.0], dtype=ir.DataType.FLOAT)

        with self.assertRaisesRegex(ValueError, "The dtype does not match the const_value"):
            _constructors.val("test", dtype=ir.DataType.INT32, const_value=const_tensor)

    def test_value_const_value_shape_mismatch_error(self):
        """Test that providing mismatched shape with const_value raises ValueError."""
        const_tensor = ir.tensor([[1, 2], [3, 4]], dtype=ir.DataType.INT32)  # Shape: [2, 2]

        with self.assertRaisesRegex(ValueError, "The shape does not match the const_value"):
            _constructors.val("test", shape=[3, 3], const_value=const_tensor)

    def test_value_initialize_with_const_value(self):
        const_tensor = ir.tensor(np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64))
        value = _constructors.val("test", const_value=const_tensor)

        self.assertEqual(value.name, "test")
        self.assertEqual(value.type, ir.TensorType(ir.DataType.DOUBLE))
        self.assertEqual(value.shape, ir.Shape([2, 2]))
        self.assertEqual(value.const_value, const_tensor)

    def test_value_creation_with_string_dimensions(self):
        """Test value creation with string dimensions in shape."""
        value = _constructors.val("dynamic", ir.DataType.FLOAT, ["batch", "seq_len", 768])

        self.assertEqual(value.name, "dynamic")
        self.assertEqual(value.shape, ir.Shape(["batch", "seq_len", 768]))

    def test_value_creation_with_none_dimensions(self):
        """Test value creation with None dimensions in shape."""
        value = _constructors.val("unknown", ir.DataType.INT64, [None, 10, None])

        self.assertEqual(value.name, "unknown")
        self.assertEqual(value.shape, ir.Shape([None, 10, None]))


if __name__ == "__main__":
    unittest.main()
